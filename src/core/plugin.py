"""Plugin protocol, scoped context and DAG loader."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .events import EventBus
from .meta import MetaRegistry
from .services import ServiceRegistry
from .streams import StreamHub
from .ui import Side, UIManager


class PluginError(RuntimeError):
    """Raised when a plugin cannot be loaded or its DAG is invalid."""


class Plugin:
    """Base class for behavior and low-level plugins.

    ``requires`` lists plugin ids, not service names.  The manager loads
    dependencies first and detects cycles.  ``optional_requires`` are soft
    edges: they are ordered before the plugin when present, but a missing
    optional dependency is not an error.  ``provides`` declares the services
    the plugin publishes through :meth:`PluginContext.register_service`.
    """

    id: str = ""
    requires: tuple[str, ...] = ()
    optional_requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    # UI action ids owned by other plugins that this plugin drives.
    uses_actions: tuple[str, ...] = ()

    def setup(self, ctx: "PluginContext") -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def teardown(self) -> None:  # pragma: no cover - optional
        pass


# Scoped registries share manager state and attach plugin ownership for teardown.


class _ScopedUI(UIManager):
    __slots__ = ("_scope_plugin_id",)

    def __init__(self, target: UIManager, plugin_id: str) -> None:
        self.__dict__ = target.__dict__
        self._scope_plugin_id = plugin_id

    def add_tab(
        self,
        side: Side,
        id: str,
        title: str,
        *,
        icon: str | None = None,
        order: float = 0,
        plugin_id: str = "",
    ) -> Any:
        return super().add_tab(
            side,
            id,
            title,
            icon=icon,
            order=order,
            plugin_id=plugin_id or self._scope_plugin_id,
        )

    def add_toolbox_item(
        self,
        tab_id: str,
        id: str,
        title: str,
        *,
        description: str = "",
        order: float = 0,
        render: Any = None,
        plugin_id: str = "",
        page_id: str | None = None,
    ) -> Any:
        return super().add_toolbox_item(
            tab_id,
            id,
            title,
            description=description,
            order=order,
            render=render,
            plugin_id=plugin_id or self._scope_plugin_id,
            page_id=page_id,
        )

    def add_page(
        self,
        id: str,
        title: str,
        *,
        render: Any = None,
        icon: str | None = None,
        order: float = 0,
        closable: bool = True,
        plugin_id: str = "",
    ) -> Any:
        return super().add_page(
            id,
            title,
            render=render,
            icon=icon,
            order=order,
            closable=closable,
            plugin_id=plugin_id or self._scope_plugin_id,
        )

    def register_action(
        self, action_id: str, handler: Any, *, plugin_id: str = ""
    ) -> None:
        owner = self._scope_plugin_id
        if plugin_id and plugin_id != owner:
            raise PluginError(
                f"Plugin '{owner}' cannot register UI actions for '{plugin_id}'"
            )
        if not action_id.startswith(f"{owner}."):
            raise PluginError(
                f"Plugin '{owner}' must register UI actions under "
                f"'{owner}.*', not {action_id!r}"
            )
        super().register_action(action_id, handler, plugin_id=owner)

    def unregister_action(
        self, action_id: str, *, plugin_id: str | None = None
    ) -> None:
        owner = self._scope_plugin_id
        if plugin_id and plugin_id != owner:
            raise PluginError(
                f"Plugin '{owner}' cannot unregister UI actions for '{plugin_id}'"
            )
        super().unregister_action(action_id, plugin_id=owner)

    def register_command(
        self,
        command_id: str,
        title: str,
        *,
        category: str = "",
        order: float = 0,
        page_id: str | None = None,
        action: str | None = None,
        keywords: str = "",
        plugin_id: str = "",
    ) -> None:
        super().register_command(
            command_id,
            title,
            category=category,
            order=order,
            page_id=page_id,
            action=action,
            keywords=keywords,
            plugin_id=plugin_id or self._scope_plugin_id,
        )

    def register_task_provider(self, provider: Any, *, plugin_id: str = "") -> None:
        super().register_task_provider(
            provider, plugin_id=plugin_id or self._scope_plugin_id
        )

    def set_status_provider(self, provider: Any, *, plugin_id: str = "") -> None:
        super().set_status_provider(
            provider, plugin_id=plugin_id or self._scope_plugin_id
        )


class _ScopedEvents(EventBus):
    __slots__ = ("_scope_plugin_id",)

    def __init__(self, target: EventBus, plugin_id: str) -> None:
        self.__dict__ = target.__dict__
        self._scope_plugin_id = plugin_id

    def on(
        self,
        event_type: Any,
        handler: Any,
        *,
        plugin_id: str = "",
        priority: int = 0,
        once: bool = False,
    ) -> Any:
        return super().on(
            event_type,
            handler,
            plugin_id=plugin_id or self._scope_plugin_id,
            priority=priority,
            once=once,
        )

    def once(self, event_type: Any, handler: Any, **kwargs: Any) -> Any:
        kwargs["once"] = True
        return self.on(event_type, handler, **kwargs)


class _ScopedMeta(MetaRegistry):
    __slots__ = ("_scope_plugin_id",)

    def __init__(self, target: MetaRegistry, plugin_id: str) -> None:
        self.__dict__ = target.__dict__
        self._scope_plugin_id = plugin_id

    def register(self, name: str, provider: Any, *, plugin_id: str = "") -> None:
        super().register(name, provider, plugin_id=plugin_id or self._scope_plugin_id)


class _ScopedServices(ServiceRegistry):
    __slots__ = ("_scope_ctx",)

    def __init__(self, target: ServiceRegistry, ctx: "PluginContext") -> None:
        self.__dict__ = target.__dict__
        self._scope_ctx = ctx

    def get(self, name: str) -> Any | None:
        self._scope_ctx._check_access(name)
        return super().get(name)

    def require(self, name: str) -> Any:
        self._scope_ctx._check_access(name)
        return super().require(name)

    def call(self, name: str, method: str, *args: Any, **kwargs: Any) -> Any:
        self._scope_ctx._check_access(name)
        return super().call(name, method, *args, **kwargs)

    def register(self, name: str, obj: Any, *, plugin_id: str = "") -> None:
        if not plugin_id or plugin_id == self._scope_ctx.plugin_id:
            self._scope_ctx.register_service(name, obj)
            return
        super().register(name, obj, plugin_id=plugin_id)


def _scoped_ui(ui: Any, plugin_id: str) -> Any:
    if not plugin_id or ui is None or isinstance(ui, _ScopedUI):
        return ui
    if isinstance(ui, UIManager):
        return _ScopedUI(ui, plugin_id)
    return ui


def _scoped_events(bus: Any, plugin_id: str) -> Any:
    if not plugin_id or bus is None or isinstance(bus, _ScopedEvents):
        return bus
    if isinstance(bus, EventBus):
        return _ScopedEvents(bus, plugin_id)
    return bus


def _scoped_meta(meta: Any, plugin_id: str) -> Any:
    if not plugin_id or meta is None or isinstance(meta, _ScopedMeta):
        return meta
    if isinstance(meta, MetaRegistry):
        return _ScopedMeta(meta, plugin_id)
    return meta


@dataclass(slots=True)
class PluginContext:
    plugin_id: str
    data_dir: Path
    events: EventBus
    streams: StreamHub
    llm: Any
    ui: Any
    services: ServiceRegistry
    meta: MetaRegistry
    logger: logging.Logger
    # Filled in by the manager from the plugin declaration.  ``None`` disables
    # the DAG boundary check (used by unit tests that build a context by hand).
    allowed_dependencies: set[str] | None = None
    declared_provides: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.ui = _scoped_ui(self.ui, self.plugin_id)
        self.events = _scoped_events(self.events, self.plugin_id)
        self.meta = _scoped_meta(self.meta, self.plugin_id)
        if (
            self.plugin_id
            and self.services is not None
            and isinstance(self.services, ServiceRegistry)
            and not isinstance(self.services, _ScopedServices)
        ):
            self.services = _ScopedServices(self.services, self)

    def _check_access(self, name: str) -> None:
        allowed = self.allowed_dependencies
        if allowed is None:
            return
        owner = self.services.owner(name)
        if not owner or owner in ("<injected>", self.plugin_id):
            return
        if owner in allowed or name in allowed:
            return
        raise PluginError(
            f"Plugin '{self.plugin_id}' cannot access service '{name}' owned by "
            f"'{owner}' without declaring it in requires or optional_requires"
        )

    def service(self, name: str) -> Any | None:
        self._check_access(name)
        return ServiceRegistry.get(self.services, name)

    def require(self, name: str) -> Any:
        self._check_access(name)
        return ServiceRegistry.require(self.services, name)

    def register_service(self, name: str, obj: Any) -> None:
        declared = tuple(self.declared_provides)
        if declared and name not in declared:
            raise PluginError(
                f"Plugin '{self.plugin_id}' declared provides={declared!r} but tried "
                f"to register service {name!r}"
            )
        if declared and isinstance(obj, Plugin):
            raise PluginError(
                "Plugin must register a narrow service facade, not the Plugin instance itself"
            )
        ServiceRegistry.register(self.services, name, obj, plugin_id=self.plugin_id)

    def log(self, level: int, message: str, **fields: Any) -> None:
        extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
        self.logger.log(level, f"{message} {extra}".rstrip())

    def session_dir(self, session_id: str | None = None) -> Path:
        sid = session_id or self.streams.current.session_id
        path = self.data_dir / "sessions" / sid
        path.mkdir(parents=True, exist_ok=True)
        return path


ContextFactory = Callable[[str], PluginContext]


def _plugin_classes(module: Any) -> list[type[Plugin]]:
    """Plugin subclasses visible in *module* (preferring its own definitions)."""

    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, Plugin) and obj is not Plugin
    ]
    own = [cls for cls in classes if getattr(cls, "__module__", "") == module.__name__]
    return own or classes


def _instantiate(klass: type[Plugin], where: str) -> Plugin:
    try:
        plugin = klass()
    except Exception as exc:  # noqa: BLE001 - surface the spec rather than a trace
        raise PluginError(f"cannot instantiate plugin class {where}: {exc}") from exc
    if not isinstance(plugin, Plugin):
        raise PluginError(f"{where} did not produce a Plugin instance")
    if not plugin.id:
        raise PluginError(f"plugin {where} has no id")
    return plugin


def _module_from_file(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PluginError(f"plugin file not found: {resolved}")
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    module_name = f"_lecture_agent_ext_{digest}"
    # A file module is always executed afresh: that *is* the hot reload, and
    # ``importlib.reload`` cannot re-find a module created from a bare path.
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise PluginError(f"cannot load plugin file {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - a broken file must not half-register
        sys.modules.pop(module_name, None)
        raise PluginError(f"importing plugin file {resolved} failed: {exc}") from exc
    return module


def resolve_plugin_spec(
    spec: Plugin | type[Plugin] | str,
    *,
    reload_module: bool = False,
) -> tuple[Plugin, str | None]:
    """Turn a plugin instance/class/module spec/path into ``(plugin, origin)``.

    ``origin`` is the string the plugin was resolved from, so the manager can
    reload it later; instances and classes have no reloadable origin.
    """

    if isinstance(spec, Plugin):
        return spec, None
    if isinstance(spec, type) and issubclass(spec, Plugin):
        return _instantiate(spec, spec.__name__), None
    if not isinstance(spec, str) or not spec.strip():
        raise PluginError(f"{spec!r} is not a plugin spec")

    text = spec.strip()
    head, _, attr = text.partition(":")
    candidate = Path(head).expanduser()
    looks_like_path = (
        candidate.suffix == ".py" or "/" in head or "\\" in head or candidate.is_file()
    )

    if looks_like_path:
        module = _module_from_file(candidate)
        if attr:
            klass = getattr(module, attr, None)
            if not (isinstance(klass, type) and issubclass(klass, Plugin)):
                raise PluginError(f"{text!r}: {attr!r} is not a Plugin subclass")
            return _instantiate(klass, text), text
        classes = _plugin_classes(module)
        if not classes:
            raise PluginError(f"{text!r} defines no Plugin subclass")
        if len(classes) > 1:
            names = ", ".join(cls.__name__ for cls in classes)
            raise PluginError(
                f"{text!r} defines several Plugin classes ({names}); use 'file.py:Class'"
            )
        return _instantiate(classes[0], text), text

    module_name = head
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PluginError(
            f"cannot import plugin module {module_name!r}: {exc}"
        ) from exc
    if reload_module:
        module = importlib.reload(module)
    if attr:
        klass = getattr(module, attr, None)
        if not (isinstance(klass, type) and issubclass(klass, Plugin)):
            raise PluginError(f"{text!r}: {attr!r} is not a Plugin subclass")
        return _instantiate(klass, text), text
    classes = _plugin_classes(module)
    if not classes:
        raise PluginError(f"module {module_name!r} defines no Plugin subclass")
    if len(classes) > 1:
        names = ", ".join(cls.__name__ for cls in classes)
        raise PluginError(
            f"module {module_name!r} defines several Plugin classes ({names}); "
            "use 'module:Class'"
        )
    return _instantiate(classes[0], module_name), text


class PluginManager:
    """Topologically loads plugins declared with ``requires``."""

    def __init__(
        self,
        *,
        bus: EventBus,
        services: ServiceRegistry,
        meta: MetaRegistry,
        context_factory: ContextFactory,
        ui: UIManager | None = None,
        streams: StreamHub | None = None,
    ) -> None:
        self._bus = bus
        self._services = services
        self._meta = meta
        self._context_factory = context_factory
        self._ui = ui
        self._streams = streams
        self._plugins: dict[str, Plugin] = {}
        self._origins: dict[str, str] = {}
        self._order: list[str] = []
        self._loaded: list[str] = []
        self._lock = threading.RLock()

    def register(
        self,
        plugin: Plugin | type[Plugin],
        *,
        origin: str | None = None,
    ) -> None:
        if isinstance(plugin, type):
            plugin = plugin()
        if not isinstance(plugin, Plugin):
            raise PluginError(f"{plugin!r} is not a Plugin")
        if not plugin.id:
            raise PluginError(f"plugin {plugin!r} has no id")
        with self._lock:
            if plugin.id in self._plugins:
                raise PluginError(f"duplicate plugin id {plugin.id!r}")
            self._plugins[plugin.id] = plugin
            if origin:
                self._origins[plugin.id] = origin

    def register_all(self, plugins: Iterable[Plugin | type[Plugin]]) -> None:
        for plugin in plugins:
            self.register(plugin)

    def _make_context(self, plugin: Plugin) -> PluginContext:
        ctx = self._context_factory(plugin.id)
        if getattr(ctx, "allowed_dependencies", None) is None:
            ctx.allowed_dependencies = {
                plugin.id,
                *plugin.requires,
                *plugin.optional_requires,
            }
        if not getattr(ctx, "declared_provides", ()):
            ctx.declared_provides = tuple(plugin.provides)
        return ctx

    @staticmethod
    def _predecessors(plugin: Plugin, plugins: dict[str, Plugin]) -> list[str]:
        """Hard dependencies plus *registered* optional ones."""

        deps = list(plugin.requires)
        for dep in plugin.optional_requires:
            if dep in plugins and dep not in deps and dep != plugin.id:
                deps.append(dep)
        return deps

    def _check_action_usage(self, plugin: Plugin, loaded: set[str]) -> None:
        """Every declared UI action must exist and belong to a loaded plugin."""

        for action_id in plugin.uses_actions:
            owner = self._ui.action_owner(action_id) if self._ui is not None else None
            if owner is None:
                raise PluginError(
                    f"plugin {plugin.id!r} declares uses_actions={action_id!r} but "
                    "no plugin registered that UI action"
                )
            if owner not in loaded:
                raise PluginError(
                    f"plugin {plugin.id!r} uses UI action {action_id!r} owned by "
                    f"unloaded plugin {owner!r}"
                )

    def _sort(self) -> list[str]:
        plugins = self._plugins
        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done
        order: list[str] = []
        stack: list[str] = []

        def visit(pid: str) -> None:
            status = state.get(pid, 0)
            if status == 2:
                return
            if status == 1:
                cycle = " -> ".join(stack[stack.index(pid) :] + [pid])
                raise PluginError(f"plugin dependency cycle: {cycle}")
            state[pid] = 1
            stack.append(pid)
            plugin = plugins[pid]
            for dep in plugin.requires:
                if dep not in plugins:
                    raise PluginError(f"plugin {pid!r} requires missing plugin {dep!r}")
            for dep in self._predecessors(plugin, plugins):
                visit(dep)
            stack.pop()
            state[pid] = 2
            order.append(pid)

        for pid in sorted(plugins):
            visit(pid)
        return order

    def _check_cycle_from(self, pid: str) -> None:
        """Reject a cycle introduced by adding/loading a single plugin."""

        plugins = self._plugins
        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(current: str) -> None:
            status = state.get(current, 0)
            if status == 2:
                return
            if status == 1:
                cycle = " -> ".join(stack[stack.index(current) :] + [current])
                raise PluginError(f"plugin dependency cycle: {cycle}")
            state[current] = 1
            stack.append(current)
            plugin = plugins.get(current)
            if plugin is not None:
                for dep in self._predecessors(plugin, plugins):
                    visit(dep)
            stack.pop()
            state[current] = 2

        visit(pid)
        # A new plugin may also close a cycle through plugins that depend on it.
        for other, plugin in plugins.items():
            if other == pid:
                continue
            if pid in self._predecessors(plugin, plugins):
                visit(other)

    @property
    def order(self) -> list[str]:
        with self._lock:
            return list(self._order)

    @property
    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._loaded)

    def get(self, plugin_id: str) -> Plugin | None:
        return self._plugins.get(plugin_id)

    def origin(self, plugin_id: str) -> str | None:
        return self._origins.get(plugin_id)

    def load(self) -> list[str]:
        order = self._sort()
        with self._lock:
            self._order = order
            self._loaded = []
        for pid in order:
            plugin = self._plugins[pid]
            ctx = self._make_context(plugin)
            try:
                plugin.setup(ctx)
            except Exception as exc:  # noqa: BLE001 - surface plugin id
                try:
                    plugin.teardown()
                except Exception:  # noqa: BLE001 - best effort rollback
                    logging.getLogger("lecture_agent.plugins").exception(
                        "plugin %s teardown after setup failure failed", pid
                    )
                self.teardown()
                self._cleanup_plugin(pid)
                raise PluginError(f"plugin {pid!r} setup failed: {exc}") from exc
            with self._lock:
                self._loaded.append(pid)
        with self._lock:
            loaded = set(self._loaded)
        for pid in order:
            try:
                self._check_action_usage(self._plugins[pid], loaded)
            except PluginError:
                self.teardown()
                raise
        return order

    def load_plugin(self, plugin_or_spec: Plugin | type[Plugin] | str) -> Plugin:
        """Register and load one plugin while the runtime is already running."""

        plugin, origin = resolve_plugin_spec(plugin_or_spec)
        pid = plugin.id
        with self._lock:
            existing = self._plugins.get(pid)
            if existing is not None:
                if existing is not plugin:
                    raise PluginError(f"duplicate plugin id {pid!r}")
                if pid in self._loaded:
                    return existing
            else:
                self._plugins[pid] = plugin
                plugin = self._plugins[pid]
                if origin:
                    self._origins[pid] = origin

        def _rollback() -> None:
            with self._lock:
                self._plugins.pop(pid, None)
                self._origins.pop(pid, None)
                self._loaded = [item for item in self._loaded if item != pid]
            try:
                self._order = self._sort()
            except PluginError:
                self._order = [item for item in self._order if item != pid]

        with self._lock:
            loaded = set(self._loaded)
            missing = [dep for dep in plugin.requires if dep not in loaded]
        if missing:
            _rollback()
            raise PluginError(
                f"plugin {pid!r} requires plugins that are not loaded: {missing}"
            )
        try:
            self._check_cycle_from(pid)
        except PluginError:
            _rollback()
            raise

        ctx = self._make_context(plugin)
        try:
            plugin.setup(ctx)
        except Exception as exc:  # noqa: BLE001 - never leave a half-loaded plugin
            try:
                plugin.teardown()
            except Exception:  # noqa: BLE001 - best effort
                logging.getLogger("lecture_agent.plugins").exception(
                    "plugin %s teardown after setup failure failed", pid
                )
            self._cleanup_plugin(pid)
            _rollback()
            raise PluginError(f"plugin {pid!r} setup failed: {exc}") from exc

        unregistered = [
            name for name in plugin.provides if self._services.get(name) is None
        ]
        if unregistered:
            try:
                plugin.teardown()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._cleanup_plugin(pid)
            _rollback()
            raise PluginError(
                f"plugin {pid!r} declared provides={plugin.provides!r} but never "
                f"registered {unregistered!r}"
            )

        with self._lock:
            loaded = set(self._loaded) | {pid}
        try:
            self._check_action_usage(plugin, loaded)
        except PluginError:
            try:
                plugin.teardown()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._cleanup_plugin(pid)
            _rollback()
            raise

        with self._lock:
            if pid not in self._loaded:
                self._loaded.append(pid)
            try:
                self._order = self._sort()
            except PluginError:
                pass
        return plugin

    def unload(self, plugin_id: str, *, cascade: bool = False) -> list[str]:
        """Unload a loaded plugin (optionally with its hard dependents)."""

        with self._lock:
            if plugin_id not in self._plugins:
                return []
            dependents = [
                pid
                for pid in self._loaded
                if pid != plugin_id and plugin_id in self._plugins[pid].requires
            ]
            order = list(self._order)
        if dependents:
            if not cascade:
                raise PluginError(
                    f"cannot unload plugin '{plugin_id}': required by {dependents}"
                )
            # Reverse topological order: dependents before their dependency.
            dependents.sort(
                key=lambda pid: order.index(pid) if pid in order else 0, reverse=True
            )

        removed: list[str] = []
        for pid in dependents:
            removed.extend(self.unload(pid, cascade=True))

        plugin = self._plugins.get(plugin_id)
        if plugin is not None and plugin_id in self._loaded:
            try:
                plugin.teardown()
            except Exception:  # noqa: BLE001 - teardown must not block unload
                logging.getLogger("lecture_agent.plugins").exception(
                    "plugin %s teardown failed", plugin_id
                )
        self._cleanup_plugin(plugin_id)
        with self._lock:
            self._plugins.pop(plugin_id, None)
            self._origins.pop(plugin_id, None)
            self._loaded = [pid for pid in self._loaded if pid != plugin_id]
            try:
                self._order = self._sort()
            except PluginError:
                self._order = [pid for pid in self._order if pid != plugin_id]
        removed.append(plugin_id)
        return removed

    def reload(self, plugin_id: str) -> Plugin:
        """Hot-reload one plugin from the spec it was resolved from."""

        with self._lock:
            plugin = self._plugins.get(plugin_id)
            if plugin is None:
                raise PluginError(f"unknown plugin '{plugin_id}'")
            origin = self._origins.get(plugin_id)
            if not origin:
                raise PluginError(
                    f"plugin '{plugin_id}' was registered programmatically and "
                    "cannot be reloaded"
                )
            dependents = [
                pid
                for pid in self._loaded
                if pid != plugin_id and plugin_id in self._plugins[pid].requires
            ]
            order = list(self._order)
        dependents.sort(key=lambda pid: order.index(pid) if pid in order else 0)
        dependent_plugins = [
            self._plugins[pid] for pid in dependents if pid in self._plugins
        ]

        self.unload(plugin_id, cascade=True)

        new_plugin, _ = resolve_plugin_spec(origin, reload_module=True)
        loaded = self.load_plugin(new_plugin)
        for dependent in dependent_plugins:
            try:
                self.load_plugin(dependent)
            except PluginError:
                logging.getLogger("lecture_agent.plugins").exception(
                    "failed to restore plugin %s after reloading %s",
                    dependent.id,
                    plugin_id,
                )
        return loaded

    def _cleanup_plugin(self, pid: str) -> None:
        self._bus.remove_plugin(pid)
        self._meta.remove_plugin(pid)
        self._services.remove_plugin(pid)
        if self._ui is not None:
            self._ui.remove_plugin(pid)
        if self._streams is not None:
            self._streams.remove_plugin(pid)
        settings = self._services.get("settings")
        unregister = getattr(settings, "unregister", None)
        if callable(unregister):
            try:
                unregister(plugin_id=pid)
            except Exception:  # noqa: BLE001 - settings cleanup is best effort
                logging.getLogger("lecture_agent.plugins").exception(
                    "settings cleanup failed for plugin %s", pid
                )
        reader = self._services.get("reader")
        drop_action = getattr(reader, "unregister_selection_action", None)
        if callable(drop_action):
            try:
                drop_action(owner=pid)
            except Exception:  # noqa: BLE001 - reader cleanup is best effort
                logging.getLogger("lecture_agent.plugins").exception(
                    "selection action cleanup failed for plugin %s", pid
                )
        drop_activity = getattr(reader, "unregister_activity_provider", None)
        if callable(drop_activity):
            try:
                drop_activity(pid)
            except Exception:  # noqa: BLE001 - reader cleanup is best effort
                logging.getLogger("lecture_agent.plugins").exception(
                    "activity provider cleanup failed for plugin %s", pid
                )

    def teardown(self) -> None:
        with self._lock:
            loaded = list(reversed(self._loaded))
            self._loaded = []
        for pid in loaded:
            plugin = self._plugins.get(pid)
            if plugin is None:
                continue
            try:
                plugin.teardown()
            except Exception:  # noqa: BLE001 - best effort shutdown
                logging.getLogger("lecture_agent.plugins").exception(
                    "plugin %s teardown failed", pid
                )
            self._cleanup_plugin(pid)
