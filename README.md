# embedding_model_finetunning
本项目用于Embedding模型的相关实验，包括Embedding模型评估、ReRank模型微调、Embedding模型微调、Embedding模型量化。

该项目需要GPU，测试的时候用的是NVIDIA GeForce RTX 3080

### 1. 启动容器
#### 1.1 制作容器
```commandline
docker build -t embedding_model_finetunning_image .
```

#### 1.2 启动容器
```commandline
docker run --gpus all -itd -v $PWD:/home --name emdedding_model_finetunning_test --network=host embedding_model_finetunning_image:latest
```

### 1. Embedding模型评估

参考脚本: `src/baseline_eval`目录：

- bge_base_zh_eval.py: BGE-base-zh-v1.5模型评估，作为基线评估（baseline）

评估结果参考 `docs/result.md` 文档。
