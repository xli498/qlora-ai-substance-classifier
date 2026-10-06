#!/usr/bin/env python3
"""年报 AI 实质性识别 Web Demo

⚠️ 重要声明：
本 Demo 后端调用的是 teacher 模型（Jev, typesafe/jev-1.13）按论文原始口径实时打分，
**不是**已发布的 QLoRA 蒸馏权重。QLoRA 微调权重尚未发布，本地无模型可跑。
打分口径与论文 Phase 5b（f05b_substance.py）完全一致，不得冒充为蒸馏模型输出。

打分口径（真实，来自论文管线）：
  1分：纯套话/蹭热点，无实质内容
  2分：提及AI但无具体行动
  3分：有一般性AI布局或规划描述
  4分：有具体AI项目、产品或研发活动
  5分：有具体投入金额、项目、研发人员、专利或产品落地
判定规则（来自 build_data.py）：
  score <= 1.5 → 非实质
  score >= 2.5 → 实质
  (1.5, 2.5) → 中间带（存疑）

启动：
  ./venv/bin/python app.py
  浏览器打开 http://127.0.0.1:7860
"""
import json
import os
import re
import subprocess
import sys
import tempfile

# 清理 no_proxy 中的 IPv6 方括号地址，避免 gradio/httpx 解析失败
# 必须在 import gradio 之前执行
_no_proxy = os.environ.get("no_proxy", "")
if "[" in _no_proxy:
    _parts = [p for p in _no_proxy.split(",") if "[" not in p]
    os.environ["no_proxy"] = ",".join(_parts)
    os.environ["NO_PROXY"] = ",".join(_parts)

JEV = "/home/hatch/workspace/skills/openrouter/bin/jev_score.py"

# 真实打分口径（来自 ~/workspace/goals/ai/full/scripts/f05b_substance.py）
QUESTION = {
    "substance": {
        "type": "score",
        "instructions": (
            "以下是一家上市公司年报中关于人工智能的表述，请按实质性打分："
            "1=纯套话/蹭热点、无实质内容；"
            "5=有具体投入（金额、项目、研发人员、专利、产品落地）。只返回分数。"
        ),
        "criteria": [
            "1分：纯套话/蹭热点，无实质内容",
            "2分：提及AI但无具体行动",
            "3分：有一般性AI布局或规划描述",
            "4分：有具体AI项目、产品或研发活动",
            "5分：有具体投入金额、项目、研发人员、专利或产品落地",
        ],
    }
}

# AI 关键词（来自 f05b_substance.py AI_DICT，含"智能"用于高亮）
AI_KEYWORDS = [
    "人工智能", "机器学习", "深度学习", "强化学习", "神经网络",
    "自然语言处理", "计算机视觉", "语音识别", "图像识别", "人脸识别",
    "知识图谱", "大模型", "生成式", "AIGC", "算法",
    "自动驾驶", "无人驾驶", "智能机器人", "智能制造", "数字孪生",
    "边缘计算", "智能客服", "智能投顾", "智能",
]

CRITERIA_TEXT = """**打分口径**（论文 Phase 5b 真实标准）：
- 1分：纯套话/蹭热点，无实质内容
- 2分：提及AI但无具体行动
- 3分：有一般性AI布局或规划描述
- 4分：有具体AI项目、产品或研发活动
- 5分：有具体投入金额、项目、研发人员、专利或产品落地

**判定规则**：≤1.5分 → 非实质；≥2.5分 → 实质；(1.5, 2.5) → 存疑（中间带）
"""


def jev_score_1to5(text):
    """调用 Jev 打分，返回 1-5 分制的分数和概率分布。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as qf:
        json.dump(QUESTION, qf, ensure_ascii=False)
        qfile = qf.name
    try:
        out = subprocess.run(
            [sys.executable, JEV, "--state", text, "--questions", "@" + qfile],
            capture_output=True, text=True, timeout=90,
        )
        if out.returncode != 0:
            raise RuntimeError(f"Jev 调用失败: {out.stderr[:500]}")
        data = json.loads(out.stdout)
        sub = data["substance"]
        # Jev 返回 0-4 分制（criteria 下标），换算为 1-5
        score_1to5 = sub["score"] + 1
        probs = sub.get("probabilities", {})
        # probs 的 key 是 "0".."4"，对应 1..5 分
        prob_1to5 = {int(k) + 1: v for k, v in probs.items()}
        return score_1to5, prob_1to5
    finally:
        os.unlink(qfile)


def extract_key_sentences(text, max_sentences=5):
    """抽取含 AI 关键词的关键句（用于展示依据），复用论文上下文抽取思路。"""
    # 按句号/分号/换行切分
    sentences = re.split(r"[。；\n]+", text)
    pat = re.compile("|".join(re.escape(k) for k in AI_KEYWORDS))
    scored = []
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        hits = pat.findall(s)
        if hits:
            # 去重关键词计数，命中越多越关键
            scored.append((len(set(hits)), s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:max_sentences]]


def highlight_keywords(text):
    """高亮 AI 关键词（HTML）。"""
    # 按长度降序避免子串先被替换（如"智能"先于"人工智能"）
    kws = sorted(set(AI_KEYWORDS), key=len, reverse=True)
    # 转义 HTML
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for kw in kws:
        text = text.replace(
            kw, f'<mark style="background:#fff3b0;padding:1px 3px;border-radius:3px;">{kw}</mark>'
        )
    return text


def classify(score):
    if score <= 1.5:
        return "非实质", "#d32f2f"
    if score >= 2.5:
        return "实质", "#2e7d32"
    return "存疑（中间带）", "#f57c00"


EXAMPLE_FUFF = (
    "公司积极响应国家人工智能发展战略，大力推进智能化转型升级，"
    "以智能化赋能高质量发展，不断提升核心竞争力，为股东创造更大价值。"
)

EXAMPLE_SUBSTANTIAL = (
    "报告期内，公司研发投入2.3亿元，其中人工智能专项投入8000万元。"
    "建成AI实验室，引进算法工程师45人，申请发明专利12项。"
    "自主研发的'智能投顾2.0'产品已上线，服务客户超10万户，实现收入1500万元。"
    "公司与高校共建联合实验室，开展大模型微调技术攻关。"
)


def predict(text):
    text = (text or "").strip()
    if not text:
        return (
            "请输入年报文本",
            "",
            "",
            "⚠️ 未输入文本",
        )
    if len(text) < 20:
        return (
            "文本过短（需≥20字）",
            "",
            "",
            "⚠️ 文本过短，无法打分",
        )
    try:
        score, probs = jev_score_1to5(text[:3000])
    except Exception as e:
        return (
            "打分失败",
            "",
            "",
            f"❌ 后端调用失败：{e}",
        )

    label, color = classify(score)

    # 结果卡片
    result_md = (
        f"## 判定：<span style='color:{color}'>{label}</span>\n\n"
        f"**实质性得分：{score:.2f} / 5.0**\n\n"
    )
    # 概率分布条
    dist_lines = ["**各分档概率**："]
    for k in range(1, 6):
        p = probs.get(k, 0.0)
        bar = "█" * int(p * 20)
        dist_lines.append(f"- {k}分：{p:.0%} {bar}")
    result_md += "\n".join(dist_lines)

    # 依据：关键句
    key_sents = extract_key_sentences(text)
    if key_sents:
        ev_md = "### 关键句（命中 AI 关键词）\n\n"
        for i, s in enumerate(key_sents, 1):
            ev_md += f"{i}. {s}\n\n"
    else:
        ev_md = "### 关键句\n\n未命中 AI 关键词（注意：若文本仅含“智能”泛词，论文管线直接记 1 分）。"

    # 高亮原文
    highlighted = highlight_keywords(text[:3000])

    note = (
        "⚠️ 本次打分为 **teacher 模型（Jev）实时打分**，非 QLoRA 蒸馏权重输出。"
        "口径与论文 Phase 5b 一致。"
    )
    return result_md, ev_md, highlighted, note


def build_ui():
    import gradio as gr

    with gr.Blocks(title="年报 AI 实质性识别 Demo") as demo:
        gr.Markdown("# 年报 AI 实质性识别 Demo")
        gr.Markdown(
            "> ⚠️ **演示声明**：本 Demo 后端调用 teacher 模型（Jev, typesafe/jev-1.13）"
            "按论文原始口径实时打分，**不是**已发布的 QLoRA 蒸馏权重。"
            "QLoRA 微调权重尚未发布，本地无模型可跑。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                txt = gr.Textbox(
                    label="年报文本（粘贴上市公司年报中 AI 相关表述）",
                    placeholder="粘贴年报文本，或点击下方示例…",
                    lines=10,
                )
                with gr.Row():
                    btn_fluff = gr.Button("示例：套话型", variant="secondary")
                    btn_sub = gr.Button("示例：实质型", variant="secondary")
                    btn_run = gr.Button("开始打分", variant="primary")
            with gr.Column(scale=2):
                out_result = gr.Markdown(label="判定结果")
                out_note = gr.Markdown(label="说明")
        with gr.Row():
            out_evidence = gr.Markdown(label="判定依据")
        with gr.Row():
            out_highlight = gr.HTML(label="关键词高亮原文")

        gr.Markdown("---")
        gr.Markdown(CRITERIA_TEXT)

        btn_run.click(
            fn=predict,
            inputs=[txt],
            outputs=[out_result, out_evidence, out_highlight, out_note],
        )
        btn_fluff.click(fn=lambda: EXAMPLE_FUFF, outputs=[txt])
        btn_sub.click(fn=lambda: EXAMPLE_SUBSTANTIAL, outputs=[txt])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)
