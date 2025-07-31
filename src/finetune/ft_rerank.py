# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: ft_rerank.py
# @time: 2024/6/19 11:30
# Python script for ReRank model fine-tuning using Sentence Transformers
import os
import logging
import pandas as pd
import tiktoken
from modelscope import snapshot_download
# model_dir = snapshot_download('Xorbits/bge-reranker-large')
# print(model_dir)
import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

# logger
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/nonexistent_dir"
os.environ["TRANSFORMERS_USE_SLOW_TOKENIZERS"] = "1"
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.5) 

# First, we define the transformer model we want to fine-tune
model_path = "ms-marco-MiniLM-L-6-v2"
train_batch_size = 1
num_epochs = 5
model_save_path = "ft_" + os.path.basename(model_path)

device_id = torch.cuda.current_device()
total_memory, free_memory = torch.cuda.mem_get_info(device_id)

# # 手动加载分词器（禁用 fast tokenizer）
# tokenizer = AutoTokenizer.from_pretrained(
#     "/root/.cache/modelscope/hub/models/BAAI/bge-reranker-base",
#     use_fast=False  # 关键：禁用 fast tokenizer
# )

model = CrossEncoder("/home/ms-marco-MiniLM-L-6-v2", num_labels=1, max_length=512)# We set num_labels=1, which predicts a continuous score between 0 and 1

# Prepare datasets for model training and evaluation
train_samples = []
dev_samples = {}

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

print(project_dir)

train_df = pd.read_csv(os.path.join(project_dir, "data/ft_rerank_train.csv"))
print(train_df.shape)
for i, row in train_df.iterrows():
    train_samples.append(InputExample(texts=[row["queries"], row["passages"]], label=row["labels"]))

val_df = pd.read_csv(os.path.join(project_dir, "data/ft_rerank_val.csv"))
for i, row in val_df.iterrows():
    query_id = row["query_id"]
    if query_id not in dev_samples:
        dev_samples[query_id] = {"query": row["queries"], "positive": set(), "negative": set()}
    
    if row["labels"]:
        dev_samples[query_id]["positive"].add(row["passages"])
    else:
        dev_samples[query_id]["negative"].add(row["passages"])

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

# Configure the training
warmup_steps = 1000
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=2000,
    optimizer_params={'lr': 1e-5},
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True
)

# Save the model
model.save(model_save_path)
