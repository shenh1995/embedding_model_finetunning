不同模型的评估指标如下：

| 模型                        | accuracy@5 | accuracy@10 | map@100 | mrr@10 | ndcg@10  | cost time |
|---------------------------|------------|-------------|---------|--------|----------|-----------|
| bge-base-zh-v1.5          | 0.8224     | 0.87850      | 0.69640  | 0.69089 | 0.7364   | 4.66s    |
| ft_bge-base-zh-v1.5       | 0.85981    | 0.91277      | 0.75273  | 0.74875 | 0.7886  | 4.64s    |

注意：

- ft_bge-base-zh-v1.5是对基准模型bge-base-zh-v1.5进行微调得到的模型，使用sentence-transformers微调。模型微调本文采用的是NVIDIA GeForce RTX 3080
- 评估脚本为 src/baseline_eval/bge_base_zh_eval.py，使用GPU测试，GPU为NVIDIA GeForce RTX 3080

