#!/usr/bin/env python3
"""数据集 v2（数据规模对照）：val/test 与 v1 完全一致（复用原文件），
train 从有效样本池中扩容采样（每类 3200 条，共 6400），并严格排除
与原 val/test 输入相同的样本，防止泄漏。

输出：~/workspace/goals/qlora/data_v2/train_v2.jsonl（val/test 直接用 ../data/ 的）
"""
import json
import os
import random
import sys

sys.path.insert(0, ".")
import pandas as pd
from f05b_substance import extract_context

BASE = "./thesis_data"
OLD = "./data"
OUT = "./data_v2"
os.makedirs(OUT, exist_ok=True)

INSTRUCTION = (
    "阅读以下上市公司年报片段，判断其中提及的 AI 相关内容是否属于实质性投入。"
    "实质性投入指有具体 AI 技术、研发项目、产品或业务落地；"
    "仅出现\"智能\"等泛词、纯套话或蹭热点的表述不算实质。"
    "只回答\"实质\"或\"非实质\"，不要输出其他内容。"
)

TRAIN_PER_CLASS = 3200

def load_inputs(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s.add(json.loads(line)["input"])
    return s

def main():
    holdout = load_inputs(f"{OLD}/val.jsonl") | load_inputs(f"{OLD}/test.jsonl")
    print(f"holdout (val+test) 输入数: {len(holdout)}")

    df = pd.read_csv(f"{BASE}/ai_substance.csv", dtype={"code": str})

    def label(s):
        if s <= 1.5:
            return 0
        if s >= 2.5:
            return 1
        return None

    df["label"] = df["substance_score"].map(label)
    df = df[df["label"].notna()]
    print(f"去中间带后: {len(df)}")

    by_label = {0: [], 1: []}
    skipped_noctx = skipped_holdout = 0
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
        if len(ctx) > 1500:
            ctx = ctx[:1500]
        if ctx in holdout:
            skipped_holdout += 1
            continue
        by_label[int(r.label)].append({
            "instruction": INSTRUCTION,
            "input": ctx,
            "output": "实质" if r.label == 1 else "非实质",
        })
    print(f"池: 非实质 {len(by_label[0])}, 实质 {len(by_label[1])} "
          f"(无上下文跳过 {skipped_noctx}, holdout 排除 {skipped_holdout})")

    random.seed(123)
    train = []
    for k in (0, 1):
        n = min(TRAIN_PER_CLASS, len(by_label[k]))
        train += random.sample(by_label[k], n)
        print(f"label={k} 取 {n} 条")
    random.shuffle(train)
    with open(f"{OUT}/train_v2.jsonl", "w", encoding="utf-8") as f:
        for it in train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n1 = sum(1 for it in train if it["output"] == "实质")
    print(f"train_v2: {len(train)} 条 (实质 {n1}, 非实质 {len(train)-n1}) -> {OUT}/train_v2.jsonl")

if __name__ == "__main__":
    main()
