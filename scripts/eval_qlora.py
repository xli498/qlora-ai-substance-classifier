#!/usr/bin/env python3
"""在 test.jsonl 上评估微调后的模型：准确率 / 精确率 / 召回率 / F1。

用法（Kaggle，训练同一 notebook）：
  !python eval_qlora.py --adapter ./qlora-annual-report-adapter --data /kaggle/input/qlora-data/test.jsonl
也可加 --base-only 做基座模型对照（量化前/微调前的零样本表现）。
"""
import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "{instruction}\n\n年报片段：\n{input}\n\n判断："

def load_model(adapter=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok

def predict(model, tok, instruction, inp, max_new=8):
    prompt = PROMPT.format(instruction=instruction, input=inp)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"(实质|非实质)", gen)
    return m.group(1) if m else "未知"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-new", type=int, default=8)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    model, tok = load_model(a.adapter)

    tp = fp = fn = tn = unknown = 0
    for r in rows:
        pred = predict(model, tok, r["instruction"], r["input"], a.max_new)
        gold = r["output"]
        if pred == "未知":
            unknown += 1
            continue
        if pred == "实质" and gold == "实质":
            tp += 1
        elif pred == "实质":
            fp += 1
        elif gold == "实质":
            fn += 1
        else:
            tn += 1
    n = len(rows) - unknown
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"样本数: {len(rows)}（未知/拒答: {unknown}）")
    print(f"准确率: {acc:.4f}  精确率: {prec:.4f}  召回率: {rec:.4f}  F1: {f1:.4f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

if __name__ == "__main__":
    main()
