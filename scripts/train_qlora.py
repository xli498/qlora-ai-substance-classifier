#!/usr/bin/env python3
"""年报 AI 实质性识别 QLoRA 微调 —— Kaggle 免费 GPU (T4 16GB) 一键运行版。

任务：二分类（实质 / 非实质），数据见 ../data/{train,val,test}.jsonl（build_data.py 生成）。
基座：Qwen2.5-1.5B-Instruct；4-bit NF4 量化 + LoRA (r=16)，单 T4 约 20~40 分钟跑完 1 epoch。

Kaggle 用法：
  1. 新建 Notebook，挂 GPU (T4)，把本目录 data/*.jsonl 以 Dataset 形式挂载到 /kaggle/input/qlora-data
  2. !pip install -q transformers peft bitsandbytes trl accelerate datasets
  3. 运行本脚本（或把下面代码粘进 cell）
输出：./qlora-annual-report-adapter/（LoRA adapter + 训练日志 + val 评估）
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = os.environ.get("QLORA_DATA_DIR", "/kaggle/input/qlora-data")  # 云端运行时用环境变量覆盖
OUT_DIR = os.environ.get("QLORA_OUT_DIR", "./qlora-annual-report-adapter")

PROMPT = (
    "{instruction}\n\n年报片段：\n{input}\n\n判断："
)

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def to_text(ex):
    return PROMPT.format(instruction=ex["instruction"], input=ex["input"]) + ex["output"]

def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list([{"text": to_text(e)} for e in load_jsonl(f"{DATA_DIR}/train.jsonl")])
    val_ds = Dataset.from_list([{"text": to_text(e)} for e in load_jsonl(f"{DATA_DIR}/val.jsonl")])

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=train_ds, eval_dataset=val_ds,
        args=args, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    print("训练完成，adapter 已保存到", OUT_DIR)

if __name__ == "__main__":
    main()
