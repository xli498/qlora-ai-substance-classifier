#!/usr/bin/env python3
"""QLoRA 数据泄漏自查: 对 data(2400) 与 data_v2(6400) 两套, 按 input 文本哈希查
train/val/test 两两重叠 + 集内重复 + 标签分布。用法: leakage_check.py <dir1> <dir2>"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

def load(p: Path):
    items = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items

def key(item):
    text = item.get("input", "") + "||" + item.get("instruction", "")[:30]
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def label(item):
    out = str(item.get("output", ""))
    return "实质" if out.startswith("实质") else ("非实质" if out else "?")

for d in sys.argv[1:]:
    d = Path(d)
    print(f"== {d}")
    splits = {}
    for name in ("train", "val", "test"):
        items = load(d / f"{name}.jsonl")
        keys = [key(i) for i in items]
        splits[name] = set(keys)
        dup = len(keys) - len(set(keys))
        bal = Counter(label(i) for i in items)
        print(f"  {name}: n={len(items)} 集内重复={dup} 标签={dict(bal)}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = splits[a] & splits[b]
        print(f"  重叠 {a}∩{b}: {len(ov)}")
