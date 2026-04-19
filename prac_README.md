# 신용카드 사기 탐지 파이프라인 — `credit_fraud_pipeline_prac.py`

Scikit-learn `Pipeline` + `ColumnTransformer`를 활용한 신용카드 사기 탐지 실습 코드 문서입니다. RandomForest의 5가지 하이퍼파라미터 조합을 비교하여 최적 모델을 자동 선택합니다.

---

## 목차
1. [전체 개요](#1-전체-개요)
2. [데이터셋](#2-데이터셋)
3. [코드 구조 — 섹션별 상세 설명](#3-코드-구조--섹션별-상세-설명)
4. [핵심 ML 개념](#4-핵심-ml-개념)
5. [실행 방법](#5-실행-방법)
6. [실행 결과 예시](#6-실행-결과-예시)
7. [개선 아이디어](#7-개선-아이디어)

---

## 1. 전체 개요

### 목적
신용카드 거래 데이터에서 **사기(fraud) 거래를 이진 분류**하는 모델을 만들고, 하이퍼파라미터 탐색을 통해 최적 모델을 선정합니다.

### 파이프라인 흐름
```
원본 CSV
   ↓
[데이터 로드 & 분할]
   ↓
[전처리]  ── 수치: 결측치 대체 → 표준화
           ── 범주: 결측치 대체 → One-Hot 인코딩
   ↓
[RandomForest 학습] × 5가지 파라미터 조합
   ↓
[평가]   ── accuracy, classification_report, ROC-AUC
   ↓
[최고 모델 선정 & 재사용]
   ↓
[신규 데이터 예측]
```

### 평가 지표 용어 설명

#### 기본 카운트 (이진 분류에서 예측 vs 실제)
| 용어 | 의미 |
|------|------|
| **TP(True Positive, 진양성)** | 실제 사기를 사기로 올바르게 예측 |
| **TN(True Negative, 진음성)** | 실제 정상을 정상으로 올바르게 예측 |
| **FP(False Positive, 위양성)** | 실제 정상을 사기로 잘못 예측 (**오탐**) |
| **FN(False Negative, 위음성)** | 실제 사기를 정상으로 잘못 예측 (**놓침**) |

#### 주요 평가 지표

- **accuracy(정확도)**: `(TP + TN) / 전체`
  전체 예측 중 맞춘 비율. 가장 직관적이지만, 클래스 불균형이 심하면 오해 소지 있음 (예: 사기 1%일 때 모두 "정상"이라고 찍어도 99%).

- **precision(정밀도)**: `TP / (TP + FP)`
  "사기라고 예측한 것 중 실제로 사기인 비율". 오탐(FP)을 줄이는 데 초점. 사기 경보가 울릴 때마다 조사 비용이 드는 상황에서 중요.

- **recall(재현율)** = **sensitivity(민감도)** = **TPR(True Positive Rate, 진양성률)**: `TP / (TP + FN)`
  "실제 사기 중에 잡아낸 비율". 놓침(FN)을 줄이는 데 초점. **사기 탐지에서 가장 중요한 지표** — 사기를 놓치면 실제 금전 손실이 발생.

- **F1-score(F1 점수)**: `2 × (precision × recall) / (precision + recall)`
  정밀도와 재현율의 조화평균(harmonic mean). 두 지표가 모두 높아야 F1도 높음 → 둘의 **균형**을 보고 싶을 때 사용.

- **support(지지도)**: 해당 클래스의 실제 샘플 수. 지표 해석 시 신뢰도 참고용.

- **macro avg(매크로 평균)**: 클래스별 지표를 **단순 평균**. 소수 클래스에 동등한 가중치를 부여할 때 사용.

- **weighted avg(가중 평균)**: 클래스별 지표를 **샘플 수(support) 기준 가중 평균**. 전체 데이터 분포를 반영.

- **classification_report(분류 리포트)**: scikit-learn이 위 precision / recall / F1-score / support / macro avg / weighted avg를 **클래스별로 한 번에 출력**해 주는 함수.

- **confusion matrix(혼동 행렬)**: TP/TN/FP/FN을 2×2 표로 정리한 것. classification_report의 기반이 되는 원천 정보.

- **ROC curve(ROC 곡선, Receiver Operating Characteristic curve)**: 분류 임계값(threshold)을 0~1로 변화시키며 **FPR(False Positive Rate, 위양성률)** 을 x축, **TPR(재현율)** 을 y축에 그린 곡선. 임계값에 따른 모델의 탐지 능력 변화를 시각화.

- **AUC(Area Under the Curve, 곡선 아래 면적)**: ROC 곡선 아래 면적.
  - 1.0 = 완벽한 분류기
  - 0.5 = 무작위 (동전 던지기)
  - < 0.5 = 예측이 반대로 나옴
  **임계값에 독립적**이라 모델 자체의 분별력을 평가할 때 유용.

- **ROC-AUC**: 위 두 개념의 합성어. 실무에서 가장 흔한 **종합 성능 지표** 중 하나.

- **predict_proba(확률 예측)**: 0/1 클래스 라벨이 아니라 **양성 클래스일 확률**을 반환하는 메서드. ROC-AUC 계산에 필요.

#### 어떤 지표를 써야 하나? (사기 탐지 맥락)
| 상황 | 우선 지표 |
|------|----------|
| 사기를 절대 놓치면 안 됨 (손실 회피) | **recall(재현율)** |
| 오탐으로 인한 고객 불편·조사 비용이 큼 | **precision(정밀도)** |
| 둘 사이 균형 | **F1-score** |
| 임계값 조정 전 전반적 모델 품질 비교 | **ROC-AUC** |
| 클래스가 매우 불균형(사기 < 1%) | accuracy ❌, **F1 / PR-AUC** 권장 |

### 사용 라이브러리
| 라이브러리 | 역할 |
|-----------|------|
| `pandas` | CSV 로드 / DataFrame 조작 |
| `scikit-learn` | 전처리·모델·평가 |
| `joblib` | 학습된 모델 직렬화 (`.pkl`) |

---

## 2. 데이터셋

**파일**: `data/credit_fraud_dataset.csv`
**크기**: 300행 × 9컬럼 (피처 8 + 타겟 1)
**사기 비율**: 약 45% (클래스 균형이 상대적으로 양호함)

### 컬럼 구성

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `amount` | 수치 | 거래 금액 |
| `hour` | 수치 | 거래 시각 (0~23) |
| `transaction_count_1h` | 수치 | 직전 1시간 내 거래 횟수 |
| `distance_from_home_km` | 수치 | 집으로부터 거리(km) |
| `age` | 수치 | 카드 소유자 나이 |
| `merchant_category` | 범주 | 가맹점 업종 (online, grocery, restaurant, ...) |
| `card_type` | 범주 | 카드 종류 (credit, debit) |
| `country` | 범주 | 거래 국가 (domestic, foreign) |
| **`is_fraud`** | 타겟 | 0=정상, 1=사기 |

---

## 3. 코드 구조 — 섹션별 상세 설명

### 섹션 0: 설정 및 모델 디렉토리 생성 (L1-24)
```python
import warnings
warnings.filterwarnings("ignore")

model_dir = "credit_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
```
- 학습된 모델을 저장할 `credit_model/` 디렉토리를 생성
- `warnings`를 무시하여 출력 가독성 확보

---

### 섹션 1: 데이터 로드 및 분할 (L26-35)
```python
df = pd.read_csv("data/credit_fraud_dataset.csv")
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

| 파라미터 | 값 | 의미 |
|---------|---|------|
| `test_size=0.2` | 20% | 테스트 세트 비율 |
| `random_state=42` | 고정 | 재현성 확보 |
| **`stratify=y`** | 타겟 기준 | **클래스 비율 유지** (불균형 데이터에서 필수) |

> **왜 `stratify`가 중요한가?**
> 사기 비율이 45%이므로 균형이 나쁘진 않지만, 무작위 분할 시 train/test의 클래스 비율이 어긋날 수 있습니다. `stratify=y`는 분할 후에도 양쪽 모두 사기 비율 ≈ 45%를 유지합니다.

---

### 섹션 2: 컬럼 분류 (L37-40)
```python
numeric_features     = ["amount", "hour", "transaction_count_1h",
                        "distance_from_home_km", "age"]
categorical_features = ["merchant_category", "card_type", "country"]
```
수치형과 범주형을 명시적으로 분리 → 각기 다른 전처리를 적용하기 위함.

---

### 섹션 3: 전처리 서브-파이프라인 (L42-56)

#### 수치형 파이프라인
```python
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # 결측치 → 중앙값
    ("scaler",  StandardScaler()),                   # 평균 0, 분산 1로 표준화
])
```
- **median 사용 이유**: `amount`, `distance_from_home_km`처럼 이상치가 있는 변수는 평균보다 중앙값이 robust
- **StandardScaler**: RandomForest 자체는 스케일에 민감하지 않지만, 파이프라인 일반화를 위해 포함

#### 범주형 파이프라인
```python
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
```
- **`fill_value="missing"`**: 결측치를 "missing" 카테고리로 처리
- **`handle_unknown="ignore"`**: 테스트 데이터에 train에 없던 값이 나와도 에러 없이 0벡터로 처리
- **`sparse_output=False`**: 밀집 행렬 반환 (RandomForest 호환성 확보)

#### ColumnTransformer로 통합
```python
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
```
- 하나의 `fit/transform` 호출로 수치/범주 전처리를 동시에 수행

---

### 섹션 4: 하이퍼파라미터 조합 (L58-90)

| # | n_estimators | max_depth | min_samples_split | class_weight | 특징 |
|---|-----|------|----|------|------|
| 1 | 100 | None | 2 | balanced | 깊이 제한 없음 (과적합 위험) |
| 2 | 200 | 10 | 5 | balanced | 중간 복잡도 |
| 3 | 50 | 5 | 10 | None | 단순/빠름 (weight 보정 없음) |
| 4 | 300 | 20 | 2 | balanced_subsample | 트리별 재샘플링 |
| 5 | 150 | 8 | 4 | balanced | 중간 조합 |

#### 주요 파라미터 의미
- **`n_estimators`**: 앙상블 트리 개수. 많을수록 안정적이지만 학습 시간 증가
- **`max_depth`**: 각 트리의 최대 깊이. 작을수록 일반화, 클수록 과적합 경향
- **`min_samples_split`**: 내부 노드 분할에 필요한 최소 샘플 수. 클수록 단순해짐
- **`class_weight`**:
  - `"balanced"`: 전체 데이터 기준 역비율 가중
  - `"balanced_subsample"`: 부트스트랩 샘플 기준 역비율 가중
  - `None`: 가중치 없음

---

### 섹션 5: 학습 & 평가 루프 (L92-120)

```python
for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(**params, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)

    acc       = accuracy_score(y_test,  pipe.predict(X_test))
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    model_uri = f"{model_dir}/pipeline_{run_name}.pkl"
    joblib.dump(pipe, model_uri)

    run_results.append({
        "run_name":  run_name,
        "accuracy":  acc,
        "train_acc": train_acc,
        "model_uri": model_uri,
    })
```

#### 핵심 포인트
1. **전체 파이프라인 저장**: `joblib.dump(pipe, ...)` — 전처리기까지 포함된 파이프라인 객체를 통째로 저장 → 예측 시 raw 데이터만 넣으면 자동으로 전처리됨
2. **`n_jobs=-1`**: 모든 CPU 코어 사용 (대규모 앙상블에서 속도 향상)
3. **train_acc vs test_acc 비교**: 차이가 크면 과적합, 작으면 일반화 잘됨
4. **`**params`**: dict 언패킹으로 파라미터를 키워드 인자로 전달

---

### 섹션 6: 최고 모델 선정 (L122-125)
```python
best = max(run_results, key=lambda x: x["accuracy"])
```
- `test accuracy` 기준으로 가장 높은 모델 선택
- 실무에서는 목적에 따라 **F1**, **recall**, **ROC-AUC**를 기준으로 삼기도 함

---

### 섹션 7: 최고 모델 상세 평가 (L127-138)
```python
best_pipeline_model = joblib.load(best["model_uri"])

y_pred  = best_pipeline_model.predict(X_test)
y_proba = best_pipeline_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["정상(0)", "사기(1)"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

#### 세부 지표 설명

**`classification_report`** 출력 항목:
- **precision** = TP / (TP + FP) — "사기라고 예측한 것 중 실제 사기인 비율"
- **recall** = TP / (TP + FN) — "실제 사기 중에 잡아낸 비율"
- **f1-score** = precision과 recall의 조화평균
- **support** = 해당 클래스의 실제 샘플 수

**ROC-AUC**:
- 분류 임계값과 독립적인 모델 성능 지표
- 1.0: 완벽, 0.5: 무작위, `predict_proba`의 양성 확률(`[:, 1]`) 사용

---

### 섹션 8: 예제 데이터 예측 (L140-157)
```python
example_data = pd.DataFrame({ ... })
predictions = best_pipeline_model.predict(example_data)
```

- **중요**: raw DataFrame을 그대로 넣으면 **파이프라인이 자동 전처리 후 예측** — 이것이 `Pipeline` 사용의 가장 큰 장점
- 두 샘플:
  - 의심스러운 거래 (고액, 새벽, 해외) → 1 (사기)
  - 일상적 거래 (소액, 오후, 국내) → 0 (정상)

---

## 4. 핵심 ML 개념

### ① 왜 Pipeline을 사용하는가?
**Pipeline 없이**:
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # fit 금지!
model.fit(X_train_scaled, y_train)
# 예측 시 매번 scaler.transform 호출 필요 → 실수 여지 많음
```

**Pipeline 사용 시**:
- **데이터 누수(leakage) 방지**: `fit_transform`은 오직 train에만, test에는 자동으로 `transform`만 적용
- **재사용성**: `joblib.dump(pipe)`로 전처리 + 모델을 하나로 배포
- **교차검증 친화**: `cross_val_score(pipe, X, y)` 한 줄로 가능

### ② ColumnTransformer의 필요성
이 데이터는 수치/범주 혼합 → 각기 다른 전처리 필요. `ColumnTransformer`는 이를 병렬로 처리한 뒤 결과를 **가로로 결합(hstack)** 해 최종 feature matrix 생성.

### ③ 과적합 진단법
```
train_acc = 1.0000, test_acc = 0.7833  → 과적합 심각
train_acc = 0.9375, test_acc = 0.8167  → 일반화 양호
```
→ `max_depth`, `min_samples_split` 등으로 모델 복잡도 제어

### ④ class_weight로 불균형 대응
- 이 실습 데이터는 사기 45%로 비교적 균형
- 실제 사기 탐지는 사기 비율이 **0.1~1%** 수준 → `class_weight="balanced"`가 recall 향상에 크게 기여

---

## 5. 실행 방법

### 환경 준비
```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 확인
uv pip list | grep -iE "scikit-learn|pandas|joblib"
```

### 실행
```bash
python credit_fraud_pipeline_prac.py
```

### 출력물
- **콘솔**: 각 파라미터별 accuracy, 최고 모델, 분류 리포트, 예측 결과
- **`credit_model/` 디렉토리**: 5개의 `.pkl` 파이프라인 파일
  ```
  credit_model/
  ├── pipeline_n100_dNone.pkl
  ├── pipeline_n200_d10.pkl
  ├── pipeline_n50_d5.pkl
  ├── pipeline_n300_d20.pkl
  └── pipeline_n150_d8.pkl
  ```

---

## 6. 실행 결과 예시

```
데이터 크기: (300, 8)  |  사기 비율: 45.0%

=================================================================
  run_name                        test_acc  train_acc  model_uri
=================================================================
  n100_dNone                        0.7833     1.0000  credit_model/pipeline_n100_dNone.pkl
  n200_d10                          0.8000     1.0000  credit_model/pipeline_n200_d10.pkl
  n50_d5                            0.8167     0.9375  credit_model/pipeline_n50_d5.pkl  ← 최고
  n300_d20                          0.8000     1.0000  credit_model/pipeline_n300_d20.pkl
  n150_d8                           0.8000     0.9958  credit_model/pipeline_n150_d8.pkl

🏆 최고 모델: n50_d5 | accuracy: 0.8167

테스트셋 분류 리포트 — n50_d5
              precision    recall  f1-score   support
       정상(0)       0.89      0.76      0.82        33
       사기(1)       0.75      0.89      0.81        27
    accuracy                           0.82        60
ROC-AUC: 0.8833

--- 예측 결과 ---
[1 0]  # 첫 번째 거래는 사기, 두 번째는 정상
```

### 결과 해석
- **n50_d5가 승리한 이유**: `max_depth=5`로 트리 복잡도가 낮아 과적합이 덜함 (train 0.94 vs test 0.82)
- **반면 n300_d20**: train 1.0 / test 0.80 → 학습 데이터 완벽 암기했지만 일반화 실패
- **사기 클래스 recall = 0.89**: 실제 사기 27건 중 24건을 잡아냄 → 실무적으로 중요한 지표

---

## 7. 개선 아이디어

| 개선 방향 | 방법 |
|----------|------|
| **교차검증 도입** | `GridSearchCV` / `StratifiedKFold`로 더 안정적인 성능 추정 |
| **실험 추적** | `mlflow.sklearn.log_model()`로 하이퍼파라미터·메트릭 자동 기록 |
| **데이터 버저닝** | `dvc add data/credit_fraud_dataset.csv`로 데이터셋 스냅샷 관리 |
| **추가 피처 엔지니어링** | `hour`를 `is_night` 같은 이진 피처로, `distance`에 로그 변환 등 |
| **모델 다양성** | LightGBM, XGBoost, LogisticRegression과 비교 |
| **불균형 처리** | SMOTE(`imbalanced-learn`), 임계값 튜닝 |
| **성능 지표** | accuracy 대신 **F1** 또는 **Recall@fixed-precision**으로 최적화 |
| **서빙** | 최고 모델을 FastAPI로 감싸 REST API 엔드포인트 제공 |

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `credit_fraud_pipeline_prac.py` | 본 실습 코드 |
| `data/credit_fraud_dataset.csv` | 입력 데이터셋 |
| `credit_model/*.pkl` | 학습된 파이프라인 모델 |
| `train.py` | iris 데이터 기반 간단 파이프라인 (비교용 레퍼런스) |
