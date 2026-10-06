# 年报 AI 实质性识别 Web Demo

## 这是什么

给 `qlora-ai-substance-classifier` 项目配的可视化演示：粘贴一段上市公司年报文本，
输出"实质 / 非实质"判定 + 1-5 分打分 + 判定依据（关键句抽取 + 关键词高亮）。

## ⚠️ 重要声明（必读）

本 Demo 后端调用的是 **teacher 模型（Jev, `typesafe/jev-1.13`）按论文原始口径实时打分**，
**不是已发布的 QLoRA 蒸馏权重**。QLoRA 微调权重尚未发布，本地无模型可跑。
界面内已明确标注，不得冒充为蒸馏模型输出。

## 打分口径（真实，来自论文管线）

原始 1-5 分制（`~/workspace/goals/ai/full/scripts/f05b_substance.py` Phase 5b）：

| 分数 | 含义 |
|------|------|
| 1分 | 纯套话/蹭热点，无实质内容 |
| 2分 | 提及AI但无具体行动 |
| 3分 | 有一般性AI布局或规划描述 |
| 4分 | 有具体AI项目、产品或研发活动 |
| 5分 | 有具体投入金额、项目、研发人员、专利或产品落地 |

二分类判定（`scripts/build_data.py`）：
- score ≤ 1.5 → **非实质**
- score ≥ 2.5 → **实质**
- (1.5, 2.5) → 存疑（中间带，训练时丢弃）

## 启动方法

```bash
cd ~/workspace/repos/qlora-ai-substance-classifier/demo
./venv/bin/python app.py
```

浏览器打开 http://127.0.0.1:7860

首次启动会加载 Gradio，约需 10-20 秒。

## 环境要求

- Python 3.8+
- 已安装的 venv（`demo/venv`，含 gradio）
- 可访问 OpenRouter 的网络（后端经 `~/workspace/skills/openrouter/bin/jev_score.py`
  调用 Jev，凭据走 `custom.openrouter` 保险库通道，无需手动配置 key）
- 每次打分约消耗一次 Jev API 调用（走用户已授权的 OpenRouter 额度）

如需重建环境：

```bash
cd demo
python3 -m venv venv
./venv/bin/pip install gradio
```

## 界面说明

- **年报文本**：粘贴待判定的年报片段（≥20字，超长自动截断前 3000 字）
- **示例按钮**："套话型"（预期 1 分左右→非实质）、"实质型"（预期 4-5 分→实质）
- **判定结果**：实质/非实质/存疑 + 1-5 分得分 + 各分档概率分布
- **判定依据**：命中 AI 关键词的关键句（按关键词命中数排序）
- **关键词高亮原文**：AI 关键词黄色高亮显示

## 文件结构

```
demo/
├── app.py      # Gradio 主程序（含打分口径、Jev 调用、关键句抽取）
├── venv/       # Python 虚拟环境（含 gradio）
└── README.md   # 本文件
```

## 学术诚实

- 分数为 LLM 辅助判断，仅作演示用途，不构成投资建议
- 与论文一致：`ai_nogeneric==0`（文本仅含"智能"泛词、无其他 AI 实词）
  的情形，论文管线直接记 1 分；本 Demo 若未命中 AI 关键词会在依据栏提示
