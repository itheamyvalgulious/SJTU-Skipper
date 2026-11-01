"""Synthetic classroom script used by the offline demo.

The demo deliberately prefixes each line with a category marker so the mocked
LLM can route it to ``knowledge.md`` / ``gossip.md`` / ``notice.md`` without
any network access.  Real lessons do not need these markers; the summarize
prompt asks the model to classify the content itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoLine:
    text: str
    category: str
    source: str = "stt"


DEMO_LINES: tuple[DemoLine, ...] = (
    DemoLine("【知识】今天我们讲复分析中的留数定理。设 $f$ 在区域 $D$ 内除有限个孤立奇点外解析。", "knowledge"),
    DemoLine("【知识】定义：若 $f$ 在 $z_0$ 的某去心邻域内解析，则称 $z_0$ 为 $f$ 的孤立奇点。", "knowledge"),
    DemoLine("【知识】根据 Laurent 展开，$f(z)=\\sum_{n=-\\infty}^{\\infty} a_n (z-z_0)^n$，其中 $a_{-1}$ 称为留数。", "knowledge"),
    DemoLine("【闲聊】老师顺便提到，最近 AI 对数学研究方式的冲击很大，很多计算可以由模型辅助完成。", "gossip"),
    DemoLine("【知识】定理：设 $f$ 在简单闭曲线 $C$ 及其内部除有限个孤立奇点外解析，则 $\\int_C f(z)\\,dz = 2\\pi i \\sum \\operatorname{Res}(f,z_k)$。", "knowledge"),
    DemoLine("【知识】证明思路：围绕每个奇点作小圆，利用柯西积分定理把大围道化为小圆之和。", "knowledge"),
    DemoLine("【通知】作业：习题 3.1 第 1、3、5 题，下周三课前交。", "notice"),
    DemoLine("【知识】注意：留数只与 Laurent 展开的 $-1$ 次项系数有关，与更高次项无关。", "knowledge"),
    DemoLine("【知识】例如 $f(z)=e^{1/z}$ 在 $z=0$ 处的留数为 $1$，因为展开式中 $z^{-1}$ 的系数是 $1$。", "knowledge"),
    DemoLine("【闲聊】他还说大家不要只刷题，要多想想定义背后的几何直觉。", "gossip"),
    DemoLine("【知识】计算留数时，若 $z_0$ 是一阶极点，则 $\\operatorname{Res}(f,z_0)=\\lim_{z\\to z_0}(z-z_0)f(z)$。", "knowledge"),
    DemoLine("【通知】期中考试定在第十周周五晚上，闭卷，范围到留数定理。", "notice"),
    DemoLine("【知识】若 $z_0$ 是 $m$ 阶极点，则 $\\operatorname{Res}(f,z_0)=\\frac{1}{(m-1)!}\\lim_{z\\to z_0}\\frac{d^{m-1}}{dz^{m-1}}\\left[(z-z_0)^m f(z)\\right]$。", "knowledge"),
    DemoLine("【闲聊】课间聊到学校食堂新开了一个窗口，味道还不错。", "gossip"),
    DemoLine("【知识】下面用留数定理计算 $\\int_0^{2\\pi}\\frac{d\\theta}{a+\\cos\\theta}$，其中 $a>1$。", "knowledge"),
    DemoLine("【知识】令 $z=e^{i\\theta}$，则 $\\cos\\theta=\\frac{z+z^{-1}}{2}$，$d\\theta=\\frac{dz}{iz}$，积分化为单位圆上的围道积分。", "knowledge"),
    DemoLine("【通知】下周的习题课改到周四下午，地点仍是东中院 2-106。", "notice"),
    DemoLine("【知识】最终得到 $\\int_0^{2\\pi}\\frac{d\\theta}{a+\\cos\\theta}=\\frac{2\\pi}{\\sqrt{a^2-1}}$。", "knowledge"),
    DemoLine("【闲聊】老师最后说，希望同学们保持好奇心，不要被考试牵着走。", "gossip"),
    DemoLine("【知识】作业之外，建议自己推导一遍 $\\int_{-\\infty}^{\\infty}\\frac{dx}{1+x^2}=\\pi$ 的围道证明。", "knowledge"),
)


def demo_transcript(total_chars: int = 900) -> list[DemoLine]:
    """Return enough lines to exceed the demo summarize threshold."""

    lines: list[DemoLine] = []
    chars = 0
    while chars < total_chars:
        for line in DEMO_LINES:
            lines.append(line)
            chars += len(line.text)
            if chars >= total_chars:
                break
    return lines
