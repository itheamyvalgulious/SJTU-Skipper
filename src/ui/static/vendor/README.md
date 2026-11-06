# 前端第三方库

本目录存放前端运行需要的第三方库，全部以静态文件形式随仓库分发，页面不访问任何 CDN。

| 目录 | 库 | 版本 | 许可证 | 用途 |
| --- | --- | --- | --- | --- |
| `katex/` | KaTeX | 见 `katex/LICENSE` | MIT | 数学公式渲染（含 `contrib/auto-render` 与字体） |
| `hls/` | hls.js | 1.5.20 | Apache-2.0 | 播放 `.m3u8` 录播/直播流（Safari 回退原生 HLS） |
| `mpegts/` | mpegts.js | 1.7.3 | Apache-2.0 | 播放 `.flv` 直播流 |

`hls/` 与 `mpegts/` 中的文件是上游 `dist/` 构建产物，未做修改；上游许可证全文见
<https://github.com/video-dev/hls.js> 和 <https://github.com/xqq/mpegts.js>。

新增或升级这些库时：

1. 只保留运行所需的 `.min.js` / `.min.css` / 字体，不提交 `src/` 与 sourcemap；
2. 同步更新上表的版本号；
3. 修改 `src/ui/static/index.html` 里对应资源的 `?v=` 查询参数，避免浏览器继续用旧缓存。
