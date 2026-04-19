# =============================================================================
# [AI] 전체 동작 원리 (Overview)
# =============================================================================
# [AI] 이 스크립트는 Iris(붓꽃) 데이터셋을 사용해 RandomForest 분류 모델의
# [AI] 하이퍼파라미터 4가지 조합을 순차적으로 학습·비교하고, 가장 성능이 좋은
# [AI] 모델을 파일로 저장해 재사용(예측)하는 "간단한 실험 파이프라인"입니다.
# [AI] (MLflow 없이 joblib만으로 모델 persistence를 구현한 형태)
#
# [AI] ──────────────── 전체 흐름 ────────────────
# [AI]  ① model/ 폴더 준비
# [AI]       └─ 학습된 .pkl 파일들을 저장할 디렉토리 생성
# [AI]
# [AI]  ② 데이터 로드 & 분할
# [AI]       └─ data/iris_data.csv 읽기
# [AI]       └─ 피처(X) / 타겟(y) 분리
# [AI]       └─ train(80%) / test(20%)로 나누기 (stratify=y로 클래스 비율 유지)
# [AI]
# [AI]  ③ 하이퍼파라미터 그리드 정의
# [AI]       └─ (n_estimators, max_depth) 4가지 조합을 리스트로 준비
# [AI]
# [AI]  ④ 조합별 학습 루프
# [AI]       └─ Pipeline(StandardScaler → RandomForest)을 구성
# [AI]       └─ fit → 테스트 정확도·학습 정확도 계산
# [AI]       └─ joblib.dump로 .pkl 저장
# [AI]       └─ 결과를 run_results 리스트에 누적
# [AI]
# [AI]  ⑤ 최고 모델 선정
# [AI]       └─ run_results 중 test accuracy가 가장 높은 항목을 max()로 선택
# [AI]
# [AI]  ⑥ 저장된 최고 모델을 다시 로드하여 예측 수행
# [AI]       └─ X_test의 앞 5개 샘플로 예측 → 실제 레이블과 비교
# [AI]
# [AI] ──────────────── 핵심 개념 ────────────────
# [AI]  • Pipeline: 전처리(scaler)와 모델(classifier)을 하나의 객체로 묶어
# [AI]    fit/predict 시 자동으로 연쇄 호출 → data leakage 방지 및 코드 간결화
# [AI]  • joblib: pickle의 대용량 numpy 배열 최적화 버전. scikit-learn 모델
# [AI]    저장의 사실상 표준.
# [AI]  • stratify: 분할 시 타겟 클래스 비율을 train/test 양쪽에 동일하게 유지.
# [AI]    불균형 데이터·다중 클래스에서 편향된 평가를 방지.
# [AI]  • train_acc vs test_acc: 두 값의 격차로 과적합 여부를 가늠.
# [AI]    train은 높은데 test가 낮으면 과적합, 둘 다 낮으면 과소적합(underfitting).
# [AI]
# [AI] ──────────────── 실행 요건 ────────────────
# [AI]  • data/iris_data.csv 존재 필요 (DVC 사용 시 `dvc pull` 선행)
# [AI]  • 의존 패키지: pandas, scikit-learn, joblib
# [AI]  • 출력물: model/pipeline_n{...}_d{...}.pkl 4개 파일
# =============================================================================

import pandas as pd                                          # [AI] CSV 로드·DataFrame 조작
#from sklearn.datasets import load_iris
# [AI] 원래는 sklearn 내장 iris 데이터를 쓰려던 코드지만 주석 처리됨
# [AI] → CSV 파일을 직접 읽어 쓰도록 변경된 것 (DVC 관리 대상으로 만들기 위함)
from sklearn.model_selection import train_test_split         # [AI] 데이터를 train/test로 분리
from sklearn.preprocessing import StandardScaler             # [AI] 평균0·표준편차1로 정규화 (특성 스케일 통일)
from sklearn.ensemble import RandomForestClassifier          # [AI] 여러 결정 트리를 앙상블한 분류기
from sklearn.pipeline import Pipeline                        # [AI] 전처리+모델을 하나로 묶어 파이프라인화
from sklearn.metrics import accuracy_score                   # [AI] 정확도 = (맞힌 예측 수) / (전체 샘플 수)
import joblib                                                # [AI] 학습된 모델을 .pkl 파일로 저장/로드
# [AI] joblib은 pickle과 유사하지만 numpy 배열 직렬화가 훨씬 빠르고 압축 효율적
# [AI] → scikit-learn 공식 권장 방식
import os                                                    # [AI] 파일·디렉토리 조작용 (존재 여부 확인·생성)

# ── 모델 저장 디렉토리 준비 ──────────────────────────────────
model_dir = "model"
# [AI] 상대 경로 "model" → 스크립트 실행 위치의 현재 작업 디렉토리 기준
# [AI] 절대 경로가 필요하면 os.path.abspath(model_dir) 사용
if not os.path.exists(model_dir):                            # [AI] 디렉토리가 없으면
    os.makedirs(model_dir)                                   # [AI] 중간 경로까지 포함해 재귀적으로 생성
    # [AI] os.makedirs는 os.mkdir와 달리 여러 단계 경로(a/b/c)도 한 번에 생성 가능
    print(f"'{model_dir}' 폴더가 생성되었습니다.")
else:                                                         # [AI] 이미 존재하면 생성 시도 시 FileExistsError 발생하므로 분기
    print(f"'{model_dir}' 폴더가 이미 존재합니다.")
# [AI] 위 패턴의 대안: os.makedirs(model_dir, exist_ok=True) — 존재해도 에러 없음 (더 간결)

# ── 1. 데이터 로드 및 분할 ────────────────────────────────────
try:
    df = pd.read_csv("data/iris_data.csv")
    # [AI] pandas가 CSV를 읽어 DataFrame으로 변환
    # [AI] 첫 행을 자동으로 컬럼명으로 인식 (header=0 기본값)
    X = df.drop('target', axis=1)
    # [AI] axis=1 → 컬럼 방향 drop (행이 아닌 'target' 컬럼을 제거)
    # [AI] 결과: X에는 피처 컬럼들만 남음 (sepal_length, sepal_width, petal_length, petal_width)
    y = df['target']
    # [AI] 타겟만 Series 형태로 추출 (0=setosa, 1=versicolor, 2=virginica)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    # [AI] test_size=0.2  → 테스트 20%, 학습 80%
    # [AI] random_state=42 → seed 고정으로 실행할 때마다 같은 분할이 나오도록 보장 (재현성)
    # [AI] stratify=y      → train과 test 모두에서 클래스 비율(0:1:2)을 원본 y와 동일하게 유지
    # [AI]                   (150개 샘플 중 각 클래스 50개씩이면 train에 40/40/40, test에 10/10/10이 들어감)
    print("✅ 데이터 로드 및 전처리 완료")
except FileNotFoundError:
    # [AI] pd.read_csv가 파일을 못 찾으면 FileNotFoundError 발생 → 여기서 캐치
    # [AI] ※ 주의: 이 스크립트는 에러 시 exit()를 호출하지 않아서 이후 코드(X 미정의)에서
    # [AI]         NameError가 발생함. 안전하게 하려면 sys.exit(1) 추가 권장.
    print("❌ 데이터를 찾을 수 없음. ")

# ── 2. 하이퍼파라미터 조합 정의 ───────────────────────────────
run_results=[]
# [AI] 각 학습 결과(run)를 dict로 담아 누적할 리스트 — 루프 종료 후 최고 모델 선정에 사용
param_list = [
    {"n_estimators": 50,  "max_depth": 2},                   # [AI] 트리 50개, 깊이 2 → 단순/빠름/과소적합 위험
    {"n_estimators": 100, "max_depth": 3},                   # [AI] 기본값에 가까운 조합
    {"n_estimators": 200, "max_depth": 5},                   # [AI] 중간 복잡도
    {"n_estimators": 300, "max_depth": 4},                   # [AI] 트리 많음 + 중간 깊이 → 안정성 ↑, 비용 ↑
]
# [AI] 수동 grid search 패턴. 조합 수가 많아지면 GridSearchCV/Optuna 등으로 자동화 권장.

# ── 3. 파라미터별 학습 & 평가 루프 ────────────────────────────
for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"
    # [AI] f-string으로 실행명 생성 (예: "n100_d3") — 로그·파일명 식별용

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # [AI] StandardScaler: (x - mean) / std 로 각 피처를 표준화
        # [AI] RandomForest 자체는 스케일에 민감하지 않지만, 파이프라인을 일반화해 두면
        # [AI] 나중에 분류기를 LogisticRegression/SVM 등으로 교체해도 그대로 동작.
        ("clf", RandomForestClassifier(**params, random_state=42))
        # [AI] **params: dict 언패킹 — {"n_estimators":50,"max_depth":2}가
        # [AI]           n_estimators=50, max_depth=2 키워드 인자로 전개됨
        # [AI] random_state=42: 부트스트랩 샘플링과 피처 무작위 선택의 seed 고정 → 재현성 확보
    ])
    pipe.fit(X_train, y_train)
    # [AI] Pipeline.fit 내부:
    # [AI]   1) scaler.fit_transform(X_train)        ← 평균·표준편차를 X_train에서만 학습
    # [AI]   2) clf.fit(scaled_X_train, y_train)     ← 변환된 입력으로 랜덤포레스트 학습
    # [AI] ※ test 데이터에는 transform만 적용됨 → data leakage 방지

    acc = accuracy_score(y_test, pipe.predict(X_test))
    # [AI] 테스트 정확도 = 모델이 처음 보는 데이터에 대한 성능 → "일반화(generalization)" 지표
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    # [AI] 학습 정확도 = 학습에 사용한 데이터에 대한 성능
    # [AI] 비교 용도:
    # [AI]   train↑ test↓ → 과적합(overfitting)
    # [AI]   train↓ test↓ → 과소적합(underfitting)
    # [AI]   train≈test   → 균형 잘 맞음

    joblib.dump(pipe, f"model/pipeline_n{params['n_estimators']}_d{params['max_depth']}.pkl")
    # [AI] 전체 Pipeline 객체(스케일러+분류기)를 통째로 직렬화하여 .pkl로 저장
    # [AI] 나중에 joblib.load로 복원하면 X raw 데이터만 넣어도 자동 스케일링→예측이 됨
    # [AI] ⚠ pickle은 scikit-learn 버전에 민감 — 로드 시 저장 당시와 동일 버전 권장
    # [AI] ⚠ 경로 문자열을 두 번 쓰고 있음 → f-string 변수로 뽑아두면 타이핑 오류 여지 줄어듦

    run_results.append({
        "run_name": run_name,
        "accuracy": acc,                                     # [AI] test 정확도
        "train_acc": train_acc,                              # [AI] train 정확도 (과적합 진단용)
        "model_uri": f"model/pipeline_n{params['n_estimators']}_d{params['max_depth']}.pkl"
        # [AI] 저장된 파일 경로를 함께 기록 → 루프 종료 후 최고 모델을 "경로로" 다시 찾아 로드하기 위함
    })

    #결과 출력
    print(f"  {run_name}: {acc:.4f}/ {train_acc:.4f} |  pipeline_n{params['n_estimators']}_d{params['max_depth']}.pkl")
    # [AI] "{acc:.4f}" → 소수점 4자리까지 표시 (예: 0.9667)

# ── 4. 최고 모델 자동 선정 ────────────────────────────────────
# 실험 후 — 가장 좋은 모델 자동 선택
best = max(run_results, key=lambda x: x["accuracy"])
# [AI] max(iterable, key=func): 각 원소에 func 적용 결과가 가장 큰 원소를 반환
# [AI] lambda x: x["accuracy"] → 각 dict의 "accuracy" 값으로 비교
# [AI] 동률 시 "먼저 등장한" 원소가 선택됨 (Python max의 안정적 규칙)
# [AI] 선정 기준을 바꾸려면 "accuracy" 대신 "f1", "roc_auc" 등으로 교체
print(f"🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")

# ── 5. 저장된 최고 모델 다시 로드 ─────────────────────────────
# 이전 실험에서 가장 좋은 모델 불러오기
# 여기서는 파이프라인 모델 'pipeline_{'n_estimators': 50, 'max_depth': 2}.pkl'을 사용합니다.
# (파이프라인이 스케일링을 자동으로 처리하므로 더 간편합니다)
model_path = best[ "model_uri"]
# [AI] best에서 최고 모델의 파일 경로만 추출
# [AI] (메모리 상 pipe 객체를 그대로 써도 되지만, "저장→로드" 과정을 시연하려는 의도)
best_pipeline_model = joblib.load(model_path)
# [AI] joblib.load: .pkl 파일을 역직렬화해 원래 Pipeline 객체로 복원
# [AI] 이 객체는 방금 학습한 pipe와 동일한 내부 상태(학습된 가중치, 스케일러 통계)를 가짐

# ── 6. 최고 모델로 예제 예측 ──────────────────────────────────
# 예제 데이터 준비 (X_test의 일부 사용)
example_data = X_test.head(5)
# [AI] DataFrame.head(n): 앞에서 n행을 DataFrame으로 반환 (n 생략 시 기본 5)
# [AI] 여기서는 테스트 세트의 앞 5개 샘플로 예측을 시연

print("--- 예제 데이터 ---")
print(example_data)

# 예측 수행
predictions = best_pipeline_model.predict(example_data)
# [AI] Pipeline.predict 내부:
# [AI]   1) scaler.transform(example_data)   ← 학습 시 계산된 평균·표준편차로 변환 (fit 아님!)
# [AI]   2) clf.predict(scaled_example)      ← 변환된 입력으로 클래스 예측
# [AI] 반환값: numpy 배열 (예: array([0, 1, 2, 0, 1]))

print("\n--- 예측 결과 ---")
print(predictions)

# 실제 값 (비교를 위해 y_test에서 해당 부분 가져오기)
actual_labels = y_test.head(5)
# [AI] X_test.head(5)와 같은 인덱스의 y_test 레이블 → 예측과 1:1 비교 가능
# [AI] (train_test_split이 X, y를 같은 순서로 나누기 때문에 인덱스 정합성 보장)
print("\n--- 실제 레이블 ---")
print(actual_labels)
# [AI] 예측(predictions)과 실제(actual_labels)가 같으면 그 샘플은 맞힌 것
# [AI] 전체 일치도를 정량화하려면 위에서 이미 계산한 accuracy_score를 참고
