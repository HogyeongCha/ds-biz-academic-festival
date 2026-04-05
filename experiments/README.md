# Hyper-Persona 기반 3대 프로파일링 통합 추천 시스템 — 실험 포트폴리오

> **LLM과 마케팅 전략의 융합적 접근**

| 항목 | 내용 |
|------|------|
| 논문 제목 | Hyper-Persona 기반 3대 프로파일링 통합 추천 시스템 |
| 팀명 | In5ight |
| 대회 | 제1회 D&B 2025 학술연구경진대회 |

## 프로젝트 개요

이커머스 환경에서 고객 데이터를 활용하여 마케팅 전략적 사고가 내재된 **Hyper-Persona 통합 추천 시스템** 아키텍처를 설계 및 구현하는 것을 목표로 한다. 고객·상품·여정의 3대 프로파일링을 LLM 기반 페르소나 생성과 결합하고, 하이브리드 추천 + LightGBM 재랭킹 파이프라인을 통해 개인화 추천 성능을 극대화한다.

본 포트폴리오는 논문의 전체 실험 과정을 6단계로 분리하여, 각 단계의 코드·산출물·회고를 독립적으로 탐색할 수 있도록 구성하였다.

---

## 파이프라인 개요

| 단계 | 논문 섹션 | 핵심 내용 | Step Directory |
|------|-----------|-----------|----------------|
| 1. 데이터 전처리·증강 | §3.2–3.3 | 결측치 처리, 이상치 클리핑, 가우시안 노이즈 증강 (×3배) | [step1_preprocessing](./step1_preprocessing/) |
| 2. 고객 프로파일링 | §4.1 | 인구통계 필터링 → RFM+관여도 K-Means → LLM 페르소나 | [step2_customer_profiling](./step2_customer_profiling/) |
| 3. 상품 프로파일링 | §4.2 | SBERT 임베딩 → PCA+UMAP → HDBSCAN → c-TF-IDF → LLM 태깅 | [step3_product_profiling](./step3_product_profiling/) |
| 4. 여정 프로파일링 | §4.3 | AIDA 퍼널 상태 전이 모델 (30분 세션 기반) | [step4_journey_profiling](./step4_journey_profiling/) |
| 5. 하이브리드 추천·재랭킹 | §4.4 | CF + 시맨틱 유사도 → 동적 가중치 → LightGBM LambdaRank | [step5_recommendation](./step5_recommendation/) |
| 6. 오프라인 평가 | §5 | Temporal split → 6개 모델 × 6개 지표 비교 | [step6_evaluation](./step6_evaluation/) |

---

## 데이터셋 정보

- **출처**: KaggleHub — *E-commerce Sales Data 2024*

| 데이터 | 원본 | 증강 후 (×3배) |
|--------|------|----------------|
| 고객 정보 | 3,900 | 11,700 |
| 상호작용 | 3,294 | 9,000 |
| 상품 정보 | 10,002 | 30,006 |

---

## 최종 평가 결과 요약

6개 비교 모델에 대해 global temporal split (80/20, 평가 사용자 20명 overlap) 기반 오프라인 평가를 수행한 결과:

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage (%) | ILS | Novelty |
|-------|-------------|-----------|---------|-------------|-----|---------|
| Most Popular | 0.000 | 0.000 | 0.000 | 0.4 | 0.279 | 11.23 |
| IBCF | 0.000 | 0.000 | 0.000 | 0.8 | 0.297 | 11.23 |
| Matrix Factorization | 0.080 | 0.800 | 0.782 | 1.7 | 0.329 | 11.32 |
| Content-Based | 0.040 | 0.400 | 0.247 | 6.2 | 0.503 | 11.26 |
| Simple Hybrid | 0.080 | 0.800 | 0.782 | 6.3 | 0.495 | 11.29 |
| **Hyper-Persona Engine** | **0.080** | **0.800** | **0.800** | **6.7** | **0.442** | **11.33** |

Hyper-Persona Engine이 NDCG@10(0.800)과 Coverage(6.7%)에서 최고 성능을 달성하였으며, Precision@10과 Recall@10에서도 MF/Simple Hybrid와 동률로 최고 수준을 기록하였다.

---

## 실행 환경 설정

### Python 가상환경

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 빠른 실행 (Fast Mode)

LLM API 없이 로컬 fallback으로 빠르게 전체 파이프라인을 실행한다:

```bash
export FAST_MODE=1
export AUG_COPIES=1
export EMBEDDING_BACKEND=tfidf
python run_all.py
```

### 논문 재현 (Paper Reproduction)

논문과 동일한 설정으로 전체 파이프라인을 재현한다 (Gemini API 키 필요):

```bash
# Linux/Mac
bash run_paper_repro.sh

# 또는 수동 설정
export PAPER_REPRO=1
export USE_GEMINI=1
export EMBEDDING_BACKEND=sbert
export AUG_COPIES=2
python run_all.py
```

### Windows 실행

```cmd
run_all.bat
```

또는 환경변수를 수동으로 설정한 후:

```cmd
set FAST_MODE=1
set AUG_COPIES=1
set EMBEDDING_BACKEND=tfidf
python run_all.py
```
