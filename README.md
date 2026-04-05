# Hyper-Persona 기반 3대 프로파일링 통합 추천 시스템

> 제1회 D&B 2025 학술연구경진대회 — 팀 In5ight

## 논문 요약

본 연구는 이커머스 환경에서 고객 데이터를 활용하여 **마케팅 전략적 사고가 내재된 Hyper-Persona 통합 추천 시스템**을 설계·구현한다. 기존 추천 시스템이 단순 협업 필터링이나 콘텐츠 기반 필터링에 머무르는 한계를 극복하기 위해, 세 가지 프로파일링 파이프라인을 통합한다:

1. **고객 프로파일링** — 인구통계 필터링 → RFM+관여도 K-Means 클러스터링 → LLM 기반 페르소나 생성
2. **상품 프로파일링** — SBERT 임베딩 → PCA+UMAP 차원축소 → HDBSCAN 클러스터링 → c-TF-IDF 키워드 → LLM 속성 태깅
3. **여정 프로파일링** — AIDA 퍼널 기반 상태 전이 모델로 고객의 현재 구매 여정 단계를 추론

이 세 프로파일을 AIDA 기반 동적 가중치로 결합한 하이브리드 추천 후보군을 생성하고, LightGBM LambdaRank로 전략적 재랭킹을 수행하여 최종 개인화 추천 리스트를 생성한다.

## 재현 과정

본 저장소는 논문의 전체 실험 과정을 실제로 재현한 결과물이다. 재현 시 사용된 기술 스택:

| 구성 요소 | 논문 원본 | 재현 시 사용 | 비고 |
|-----------|-----------|-------------|------|
| LLM | Gemini 2.0 Flash | Gemini 2.5 Flash | 2.0 Flash 서비스 종료로 후속 모델 사용 |
| 텍스트 임베딩 | SBERT (paraphrase-mpnet-base-v2) | 동일 | 768차원 벡터 |
| CF 모델 | Surprise SVD | NMF (scikit-learn) | numpy 2.x 호환성 문제로 fallback |
| 재랭킹 | LightGBM LambdaRank | 동일 | |
| 데이터 증강 | 가우시안 노이즈 ×3배 | 동일 | |
| 클러스터링 | HDBSCAN 그리드서치 | 동일 | |
| 여정 모델 | AIDA 상태 전이 | 동일 | |


## 프로젝트 구조

```
├── experiments/                    # 실험 포트폴리오 (6단계 + Root README)
│   ├── README.md                   # 전체 실험 개요 및 네비게이션
│   ├── step1_preprocessing/        # 데이터 전처리·증강 (§3.2–3.3)
│   ├── step2_customer_profiling/   # 고객 프로파일링 (§4.1)
│   ├── step3_product_profiling/    # 상품 프로파일링 (§4.2)
│   ├── step4_journey_profiling/    # 여정 프로파일링 (§4.3)
│   ├── step5_recommendation/       # 하이브리드 추천·재랭킹 (§4.4)
│   └── step6_evaluation/           # 오프라인 평가 (§5)
├── pipeline/
│   ├── scripts/                    # 원본 파이프라인 스크립트 (phase1~6)
│   └── data/                       # 원본 입력 데이터 (CSV)
├── requirements.txt                # Python 의존성
├── run_all.py                      # 전체 파이프라인 실행 스크립트
├── run_all.bat                     # Windows 실행 스크립트
└── run_paper_repro.sh              # 논문 재현 모드 실행 스크립트
```

각 step 디렉토리는 `code/` (스크립트), `outputs/` (산출물), `README.md` (회고)로 구성된다.

## 최종 평가 결과

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage (%) | ILS | Novelty |
|-------|-------------|-----------|---------|-------------|-----|---------|
| Most Popular | 0.000 | 0.000 | 0.000 | 0.4 | 0.279 | 11.23 |
| IBCF | 0.000 | 0.000 | 0.000 | 0.8 | 0.297 | 11.23 |
| Matrix Factorization | 0.080 | 0.800 | 0.782 | 1.7 | 0.329 | 11.32 |
| Content-Based | 0.040 | 0.400 | 0.247 | 6.2 | 0.503 | 11.26 |
| Simple Hybrid | 0.080 | 0.800 | 0.782 | 6.3 | 0.495 | 11.29 |
| **Hyper-Persona Engine** | **0.080** | **0.800** | **0.800** | **6.7** | **0.442** | **11.33** |

## 데이터셋

KaggleHub *E-commerce Sales Data 2024* 기반. 가우시안 노이즈 증강(×3배) 적용.

| 데이터 | 원본 | 증강 후 |
|--------|------|---------|
| 고객 | 3,900 | 11,700 |
| 상호작용 | 3,294 | 9,000 |
| 상품 | 10,002 | 30,006 |

## 실행 방법

```bash
# 환경 설정
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 빠른 실행 (LLM API 불필요)
FAST_MODE=1 AUG_COPIES=1 EMBEDDING_BACKEND=tfidf python run_all.py

# 논문 재현 (Gemini API 키 필요)
bash run_paper_repro.sh
```

## 팀원

| 학번 | 학과 | 이름 |
|------|------|------|
| 2022****** | 경영학부 | 김*빈 |
| 2022****** | 경영학부 | 김*현 |
| 2023****** | 경영학부 | 황*서 |
| 2024****** | 데이터사이언스학부 | 차*경 |
| 2024****** | 데이터사이언스학부 | 최*석 |
