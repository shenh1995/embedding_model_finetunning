不同模型的评估指标如下：

| 模型                        | accuracy@5 | accuracy@10 | map@100 | mrr@10 | ndcg@10  | cost time |
|---------------------------|------------|-------------|---------|--------|----------|-----------|
| bge-base-zh-v1.5          | 0.8224     | 0.87850      | 0.69640  | 0.69089 | 0.7364   | 4.66s    |
| ft_bge-base-zh-v1.5       | 0.85981    | 0.91277      | 0.75273  | 0.74875 | 0.7886  | 4.64s    |

注意：

- ft_bge-base-zh-v1.5是对基准模型bge-base-zh-v1.5进行微调得到的模型，使用sentence-transformers微调。模型微调本文采用的是NVIDIA GeForce RTX 3080
- 评估脚本为 src/baseline_eval/bge_base_zh_eval.py，使用GPU测试，GPU为NVIDIA GeForce RTX 3080

不同Rerank模型的评估指标如下：

bge-rerank-base:

| retrievers                          | hit_rate | mrr    |
|-------------------------------------|----------|--------|
| ensemble_bge_base_rerank_top_1_eval | 0.8255   | 0.8255 |
| ensemble_bge_base_rerank_top_2_eval | 0.8785   | 0.8489 |
| ensemble_bge_base_rerank_top_3_eval | 0.9346   | 0.8686 |
| ensemble_bge_base_rerank_top_4_eval | 0.947    | 0.872  |
| ensemble_bge_base_rerank_top_5_eval | 0.9564   | 0.8693 |

bge-rerank-large:

| retrievers                           | hit_rate | mrr    |
|--------------------------------------|----------|--------|
| ensemble_bge_large_rerank_top_1_eval | 0.8224   | 0.8224 |
| ensemble_bge_large_rerank_top_2_eval | 0.8847   | 0.8364 |
| ensemble_bge_large_rerank_top_3_eval | 0.9377   | 0.8572 |
| ensemble_bge_large_rerank_top_4_eval | 0.9502   | 0.8564 |
| ensemble_bge_large_rerank_top_5_eval | 0.9626   | 0.8537 |

ft-bge-rerank-base:

| retrievers                             | hit_rate | mrr      | 
|----------------------------------------|----------|----------|
| ensemble_ft_bge_base_rerank_top_1_eval | 0.8474   | 0.8474   |
| ensemble_ft_bge_base_rerank_top_2_eval | 0.9003   | 0.8816   |
| ensemble_ft_bge_base_rerank_top_3_eval | 0.9408   | 0.9102   | 
| ensemble_ft_bge_base_rerank_top_4_eval | 0.9533   | 0.9180   | 
| ensemble_ft_bge_base_rerank_top_5_eval | 0.9657   | 0.9240   | 


ft-bge-rerank-large:

| retrievers                              | hit_rate | mrr     |
|-----------------------------------------|----------|---------|
| ensemble_ft_bge_large_rerank_top_1_eval | 0.8474   | 0.8474  |
| ensemble_ft_bge_large_rerank_top_2_eval | 0.9003   | 0.8769  |
| ensemble_ft_bge_large_rerank_top_3_eval | 0.9439   | 0.9024  |
| ensemble_ft_bge_large_rerank_top_4_eval | 0.9564   | 0.9029  |
| ensemble_ft_bge_large_rerank_top_5_eval | 0.9688   | 0.9028  |