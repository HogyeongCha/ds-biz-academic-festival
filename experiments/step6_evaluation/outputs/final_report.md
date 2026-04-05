# Hyper-Persona Evaluation Report

## Data split
- Method: global temporal split (80/20)
- Train interactions: 7,197
- Test interactions: 1,800
- Eval users: 20

## Model summary
- CF baseline: Matrix Factorization (NMF fallback)
- Content baseline: SBERT cosine similarity (paraphrase-mpnet-base-v2)
- Ranker: LightGBM ranker
- Best Precision@10 model: Matrix Factorization (SVD-style)

## Metrics

| Model                            |   Precision@10 |   Recall@10 |   NDCG@10 |   Coverage (%) |   ILS |   Novelty |
|:---------------------------------|---------------:|------------:|----------:|---------------:|------:|----------:|
| Most Popular                     |           0    |         0   |     0     |            0.4 | 0.279 |     11.23 |
| IBCF                             |           0    |         0   |     0     |            0.8 | 0.297 |     11.23 |
| Matrix Factorization (SVD-style) |           0.08 |         0.8 |     0.782 |            1.7 | 0.329 |     11.32 |
| Content-Based                    |           0.04 |         0.4 |     0.247 |            6.2 | 0.503 |     11.26 |
| Simple Hybrid                    |           0.08 |         0.8 |     0.782 |            6.3 | 0.495 |     11.29 |
| Hyper-Persona Engine             |           0.08 |         0.8 |     0.8   |            6.7 | 0.442 |     11.33 |
