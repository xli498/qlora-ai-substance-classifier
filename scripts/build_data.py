#!/usr/bin/env python3
"""构建年报 AI 实质性识别指令微调数据集。

数据来源（真实）：
- ~/workspace/goals/ai/full/data/reports_txt/{code}_{year}.txt：39,142 份年报文本
- ~/workspace/goals/ai/full/data/ai_substance.csv：33,948 条 Jev/规则实质打分（1~5分）

任务：二分类 —— 年报 AI 相关片段是否属于实质性投入。
标签口径（与论文 Jev 打分问题一致）：
  score <= 1.5 → 非实质（纯套话/蹭热点/仅泛词）
  score >= 2.5 → 实质（有具体技术、项目、研发或业务落地）
  (1.5, 2.5) 中间带丢弃，保证标签干净。

输出：~/workspace/goals/qlora/data/{train,val,test}.jsonl（instruction/input/output）
"""
import json
import os
import random
import sys

sys.path.insert(0, ".")
import pandas as pd
from f05b_substance import extract_context  # 复用论文管线的上下文抽取

BASE = "./thesis_data"
OUT = "./data"
os.makedirs(OUT, exist_ok=True)

INSTRUCTION = (
    "阅读以下上市公司年报片段，判断其中提及的 AI 相关内容是否属于实质性投入。"
    "实质性投入指有具体 AI 技术、研发项目、产品或业务落地；"
    "仅出现\"智能\"等泛词、纯套话或蹭热点的表述不算实质。"
    "只回答\"实质\"或\"非实质\"，不要输出其他内容。"
)

def main():
    df = pd.read_csv(f"{BASE}/ai_substance.csv", dtype={"code": str})
    print(f"打分记录总数: {len(df)}")

    def label(s):
        if s <= 1.5:
            return 0
        if s >= 2.5:
            return 1
        return None

    df["label"] = df["substance_score"].map(label)
    df = df[df["label"].notna()]
    print(f"去中间带后: {len(df)} (label=0: {(df['label']==0).sum()}, label=1: {(df['label']==1).sum()})")

    items = []
    skipped_noctx = 0
    for r in df.itertuples():
        fn = f"{BASE}/reports_txt/{r.code}_{r.year}.txt"
        if not os.path.exists(fn):
            continue
        with open(fn, encoding="utf-8") as f:
            text = f.read()
        ctx = extract_context(text)
        if not ctx or len(ctx.strip()) < 50:
            skipped_noctx += 1
            continue
        # 截断过长输入，微调时只用前 1500 字
        if len(ctx) > 1500:
            ctx = ctx[:1500]
        items.append({
            "instruction": INSTRUCTION,
            "input": ctx,
            "output": "实质" if r.label == 1 else "非实质",
            "meta": {"code": r.code, "year": int(r.year), "score": float(r.substance_score)},
        })
    print(f"有效样本: {len(items)} (无上下文跳过: {skipped_noctx})")

    # 类别平衡：每类最多取 1500 条
    random.seed(42)
    by_label = {0: [], 1: []}
    for it in items:
        by_label[0 if it["output"] == "非实质" else 1].append(it)
    n_per_class = min(1500, min(len(by_label[0]), len(by_label[1])))
    balanced = []
    for k in (0, 1):
        balanced += random.sample(by_label[k], n_per_class)
    random.shuffle(balanced)
    print(f"平衡后每类 {n_per_class} 条，共 {len(balanced)} 条")

    # 分层 8/1/1
    random.seed(7)
    splits = {"train": [], "val": [], "test": []}
    for k in (0, 1):
        cls = [it for it in balanced if (it["output"] == "实质") == bool(k)]
        random.shuffle(cls)
        n = len(cls)
        splits["train"] += cls[: int(n * 0.8)]
        splits["val"] += cls[int(n * 0.8): int(n * 0.9)]
        splits["test"] += cls[int(n * 0.9):]
    for name, lst in splits.items():
        random.shuffle(lst)
        # 去掉 meta 再落盘（训练不需要）
        out = [{kk: v for kk, v in it.items() if kk != "meta"} for it in lst]
        with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
            for it in out:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        n1 = sum(1 for it in lst if it["output"] == "实质")
        avg_len = sum(len(it["input"]) for it in lst) / len(lst)
        print(f"{name}: {len(lst)} 条 (实质 {n1}, 非实质 {len(lst)-n1}), 平均输入 {avg_len:.0f} 字")

if __name__ == "__main__":
    main()
