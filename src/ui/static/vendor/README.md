# 前端第三方库

本目录存放前端运行需要的第三方库，全部随应用本地分发，页面不访问 CDN．

| 目录 | 库 | 版本 | 许可证 | 用途 |
| --- | --- | --- | --- | --- |
| `katex/` | KaTeX | 见 `katex/LICENSE` | MIT | 数学公式渲染（含 `contrib/auto-render` 与字体） |
| `hls/` | hls.js | 1.5.20 | Apache-2.0 | 播放 `.m3u8` 录播/直播流（Safari 回退原生 HLS） |
| `mpegts/` | mpegts.js | 1.7.3 | Apache-2.0 | 播放 `.flv` 直播流 |
| `bootstrap/` | Bootstrap | 6.0.0-alpha.1 (`cca152095afcf92b9defb5deae8304a2a84f61eb`) | MIT | 组件样式与交互 |

`hls/` 与 `mpegts/` 使用上游 `dist/` 构建产物．上游许可证见 <https://github.com/video-dev/hls.js> 和 <https://github.com/xqq/mpegts.js>．

库维护步骤：

1. 只保留运行所需的 `.min.js` / `.min.css` / 字体，不提交 `src/` 与 sourcemap；
2. 同步更新上表的版本号．
