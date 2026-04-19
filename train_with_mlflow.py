# =============================================================================
# 초금: 이 코드가 뭘 하는 프로그램인지 — 초보자를 위한 한 장 요약
# =============================================================================
# 초금: "컴퓨터한테 붓꽃(Iris) 사진 보고 종류를 맞히도록 4번 다른 방법으로 가르친 뒤,
# 초금:  그 중 제일 잘 맞히는 녀석을 '공식 선수(production)'로 뽑아서 보관해 두는 프로그램"입니다.
#
# 초금: 일상 비유로 설명하면:
# 초금:   ① 학생 4명(n50_d2, n100_d3, n200_d5, n300_d4)에게 각기 다른 공부법으로 시험 준비시킴
# 초금:   ② 모의고사(test set)로 누가 제일 잘 맞히는지 점수 매김 (= accuracy)
# 초금:   ③ 매 시험마다 "이름/공부법/점수/답안지"를 MLflow 노트에 꼬박꼬박 기록
# 초금:   ④ 최고 득점자를 "공식 대표 선수"로 등록 (= Model Registry)
# 초금:   ⑤ 그 선수에게 "production"이라는 배지(별명)를 달아 줌 (= alias)
# 초금:   ⑥ 나중에 실제 예측이 필요하면 "production 배지 단 선수 불러와!" 로 찾기
#
# 초금: 왜 MLflow를 쓰나?
# 초금:   → 엑셀에 손으로 "이번엔 n=100, 점수 0.93" 적다 보면 금방 혼란스러워짐.
# 초금:     MLflow는 이 모든 걸 자동으로 기록·비교·재현·배포까지 도와주는 "ML용 공책 + 창고"입니다.
#
# 초금: 이 파일에서 쓰이는 핵심 단어 사전:
# 초금:   • Experiment(실험)  = 프로젝트 이름 (예: "iris_classification")
# 초금:   • Run(실행)         = 한 번의 학습 시도 (예: "n100_d3")
# 초금:   • Parameter(파라미터)= 내가 고른 설정값 (트리 100개, 깊이 3)
# 초금:   • Metric(지표)      = 결과 점수 (accuracy=0.93)
# 초금:   • Artifact(결과물)  = 실험 부산물 (학습된 모델 파일)
# 초금:   • Registry(등록소)   = 공식 선수 보관함
# 초금:   • Alias(별명)       = 공식 선수에 붙이는 배지 (production, staging 등)
# =============================================================================

import pandas as pd                                          # 데이터프레임 처리용
# 초금: pandas = "엑셀을 파이썬에서 다루는 도구". df는 엑셀 시트 한 장이라고 생각하세요.
from sklearn.model_selection import train_test_split         # 데이터를 train/test로 분리
# 초금: 데이터를 "공부용(train)"과 "시험용(test)"으로 나누는 함수.
from sklearn.preprocessing import StandardScaler             # 특성값 정규화 (평균0, 표준편차1)
# 초금: 숫자 키 171cm, 몸무게 65kg처럼 단위가 다른 값을 "같은 눈금"으로 맞춰주는 도구.
from sklearn.ensemble import RandomForestClassifier          # 앙상블 기반 분류 모델
# 초금: "결정 트리 여러 개가 투표하는" 모델. 혼자보다 여러 전문가가 투표하면 더 정확하다는 아이디어.
from sklearn.pipeline import Pipeline                        # 전처리+모델을 하나로 묶는 파이프라인
# 초금: "전처리 → 학습"처럼 여러 단계를 한 줄로 연결한 '조립 라인'. 한 번에 쭉 돌아감.
from sklearn.metrics import accuracy_score                   # 예측 정확도 계산
# 초금: 정확도 = 맞힌 개수 ÷ 전체 문제 수. 1.0=완벽, 0.5=반반 찍기.
import mlflow                                                # 실험 추적 및 모델 관리
# 초금: MLflow의 "메인 창구". mlflow.log_xxx() 함수로 실험 기록을 남깁니다.
# [AI] mlflow 네임스페이스는 "fluent API" — 전역 상태(현재 실험, 현재 run)를
# [AI] 내부적으로 관리하며 mlflow.log_*()만 호출해도 자동으로 현재 run에 기록됨
import mlflow.sklearn                                        # sklearn 모델 전용 MLflow 기능
# 초금: MLflow가 scikit-learn 모델을 "어떻게 저장/불러올지" 아는 전용 코너.
# [AI] mlflow.sklearn은 scikit-learn 전용 flavor 모듈:
# [AI]   - log_model / load_model 시 pickle 직렬화 자동 처리
# [AI]   - 의존 패키지(conda.yaml, requirements.txt, python_env.yaml)를 자동 기록
# [AI]   - 타 flavor 예시: mlflow.pytorch, mlflow.tensorflow, mlflow.xgboost ...
import os                                                    # 환경변수 읽기용
# 초금: OS(운영체제) 관련 기능. 여기선 환경변수 값을 읽는 용도로만 사용.
from mlflow.tracking import MlflowClient                     # MLflow 서버 직접 조작용 클라이언트
# 초금: MLflow를 좀 더 세밀하게 조작할 때 쓰는 "고급 리모컨". 별명(alias) 붙일 때 필요.
# [AI] MlflowClient는 "low-level API" — fluent API와 달리 전역 상태 없이
# [AI] 명시적으로 run_id, model_name 등을 넘겨 조작. Registry의 alias/tag처럼
# [AI] fluent API로 처리할 수 없는 세밀한 작업에 사용.

# ── DagsHub 연결 ──────────────────────────────────────────────
# 초금: DagsHub = "GitHub + MLflow + DVC가 한 곳에 합쳐진" 협업 플랫폼.
# 초금: dagshub.init(mlflow=True) 한 줄이:
# 초금:   1) MLFLOW_TRACKING_URI를 DagsHub 주소로 자동 설정
# 초금:   2) 인증 토큰을 자동 관리 (최초 실행 시 브라우저 OAuth)
# 초금:   3) mlflow.set_tracking_uri까지 내부적으로 호출
# 초금: → 이 블록이 있는 한 아래 set_tracking_uri 호출은 사실상 no-op이 됩니다.
# [AI] dagshub.init은 os.environ에 MLFLOW_TRACKING_URI/USERNAME/PASSWORD를 심고
# [AI] mlflow.set_tracking_uri까지 호출. 최초 실행 시 로컬에 토큰 캐시 저장
# [AI] (~/.config/dagshub/tokens) → 이후 재실행 때는 OAuth 재로그인 불필요.
import dagshub
dagshub.init(repo_owner="wikikoroy", repo_name="mlops", mlflow=True)

# ── MLflow 연결 설정 ──────────────────────────────────────────
# 초금: MLflow는 "어딘가의 기록 장소(= Tracking 서버)"에 실험 결과를 저장합니다.
# 초금: 그 장소의 주소를 먼저 알려줘야 "거기다 기록해!" 라고 시킬 수 있어요.
# [AI] MLflow는 "Tracking Server"라는 중앙 서버에 실험 결과를 기록하는 구조.
# [AI] 구성 요소:
# [AI]   1) Backend Store (SQLite/PostgreSQL): 파라미터·메트릭·메타데이터 저장
# [AI]   2) Artifact Store (로컬/S3/GCS): 모델 바이너리, 플롯, 로그 파일 저장
# [AI]   3) Tracking Server (HTTP): 클라이언트와 스토어를 중개하는 REST API
# 환경변수 MLFLOW_TRACKING_URI가 있으면 사용, 없으면 로컬 서버 주소로 대체
# GitHub Actions에서는 Secrets에 등록된 값이 자동으로 환경변수로 주입됨
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# 초금: os.getenv("키", "기본값") → 컴퓨터에 "키"라는 환경변수가 있으면 그 값을,
# 초금:                            없으면 "기본값"을 돌려준다. 실행 환경마다 다른 서버를 쓰기 쉬움.
# [AI] getenv(key, default): 환경변수 조회 시 없으면 default 반환 (KeyError 방지)
# [AI] 지원되는 URI 스킴 예시:
# [AI]   - http(s)://...       : 원격 Tracking Server
# [AI]   - file:./mlruns       : 로컬 파일 기반(서버 없이 동작, 단 Registry 제한)
# [AI]   - sqlite:///mlflow.db : 로컬 SQLite 직접 사용
# [AI]   - databricks, dagshub 등 클라우드 호스팅 서비스
mlflow.set_tracking_uri(tracking_uri)                        # MLflow가 이 주소의 서버로 연결
# 초금: "앞으로 모든 mlflow 함수는 이 주소에 기록해!" 라고 한 번 선언해 두는 줄.
# [AI] set_tracking_uri는 이후의 모든 mlflow.* 호출이 이 서버를 바라보도록 전역 설정.
# [AI] 코드 내부가 아닌 환경변수로 설정하면 동일 코드를 로컬/CI/프로덕션에서 그대로 재사용 가능.

# Basic Auth가 필요한 MLflow 서버(예: DagsHub)일 경우 인증 정보 설정
# 환경변수에 USERNAME이 있을 때만 실행 → 로컬에서는 건너뜀
# 초금: 회사 공용 MLflow 서버처럼 로그인이 필요한 경우를 위한 분기. 로컬 실습에서는 건너뜁니다.
if "MLFLOW_TRACKING_USERNAME" in os.environ:
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    # [AI] MLflow 클라이언트는 요청 시 이 두 환경변수를 자동으로 읽어 Authorization 헤더에 삽입.
    # [AI] 여기서의 재할당은 사실상 no-op에 가깝지만, 의존 명시 + 누락 시 조기 실패 효과.

experiment_name = "iris_classification"                      # 실험 묶음 이름 (MLflow UI에서 폴더처럼 표시됨)
# 초금: experiment = "학교 과제 이름". 그 안에 여러 번의 시도(run)가 쌓입니다.
# [AI] MLflow 계층 구조:
# [AI]   Experiment (여러 실험의 묶음: "iris_classification")
# [AI]       └── Run (한 번의 학습 시도: n50_d2, n100_d3, ...)
# [AI]              ├── Parameters (n_estimators=50, max_depth=2)
# [AI]              ├── Metrics    (accuracy=0.94)
# [AI]              └── Artifacts  (model/, plots/, ...)
mlflow.set_experiment(experiment_name)                       # 없으면 자동 생성, 있으면 해당 실험에 기록
# 초금: 같은 이름 과제가 있으면 거기에 이어쓰고, 없으면 새 과제 폴더를 만듭니다.
# [AI] 동일 이름의 experiment가 이미 있으면 그 ID를 사용, 없으면 새로 생성.
# [AI] 이후의 start_run()은 이 experiment 아래에 run을 만든다.

# ── 1. 데이터 로드 및 전처리 ──────────────────────────────────
# 초금: 공부할 데이터를 파일에서 읽어와서, 정답과 문제를 분리하고, 공부용/시험용으로 나눕니다.
try:
    df = pd.read_csv("data/iris_data.csv")                   # DVC로 관리되는 데이터 파일 로드
    # 초금: CSV 파일을 엑셀 시트처럼 "df"라는 변수로 읽어옴.
    # [AI] DVC가 추적 중인 실제 데이터는 원격 스토리지에만 있고, 로컬에는
    # [AI] data/iris_data.csv.dvc 메타파일만 있을 수 있음 → 실행 전 `dvc pull` 필요.
    df = df.select_dtypes(include=['number']).dropna()        # 수치형 컬럼만 선택 + 결측치 행 제거
    # 초금: 숫자 컬럼만 남기고, 비어있는(NaN) 행은 버림. 불순물 제거 단계.
    # [AI] select_dtypes(include=['number']): int/float 컬럼만 필터. 범주형은 이 스크립트에서 제외.
    # [AI] dropna(): 결측치가 하나라도 있는 행 전체 제거 (기본 how='any').
    # [AI] iris 데이터는 보통 결측이 없지만 방어적 코딩 관점의 안전장치.
    X = df.drop('target', axis=1)                            # 입력 특성 (꽃받침/꽃잎 길이·너비)
    # 초금: X = 문제지 (꽃잎·꽃받침의 길이·너비 등 특징들)
    # [AI] axis=1 → 컬럼 방향으로 drop (axis=0은 행).
    y = df['target']                                         # 정답 레이블 (0=setosa, 1=versicolor, 2=virginica)
    # 초금: y = 정답지 (이 꽃이 몇 번 종류인지)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 초금: 전체 데이터를 80%(공부용)과 20%(시험용)로 나눔.
    # 초금: random_state=42 → "난수 시드". 같은 숫자를 쓰면 매번 똑같이 나뉘어 재현이 됩니다.
    # [AI] test_size=0.2 → 20% 테스트 / 80% 학습
    # [AI] random_state=42 → 분할 결과 재현성 확보 (같은 seed면 항상 같은 분할)
    # [AI] ※ 이 스크립트에는 stratify가 없음 → 다중 클래스라면 train/test의 클래스 비율이 흔들릴 수 있음
    print("✅ 데이터 로드 및 전처리 완료")
except FileNotFoundError:
    # 초금: 파일이 없을 때 프로그램이 죽지 않고 "친절한 메시지"를 출력하도록 해주는 안전장치.
    print("❌ 데이터를 찾을 수 없음. dvc pull 확인 필요.")
    exit(1)                                                   # 데이터 없으면 실험 자체가 의미 없으므로 종료
    # [AI] exit(1): 비정상 종료 코드 1 반환. CI/CD 파이프라인에서 이 값을 보고 workflow 실패로 판정.

# ── 2. 실험 파라미터 목록 정의 ────────────────────────────────
# 초금: "어떤 공부법(설정)으로 학생을 훈련시킬까?" 를 4가지로 미리 정해둠.
run_results = []                                             # 각 run의 결과를 담을 리스트 (나중에 최고 모델 선택용)
# 초금: 4번의 실험 결과를 이 빈 바구니에 차곡차곡 담습니다.
param_list = [
    {"n_estimators": 50,  "max_depth": 2},                  # 트리 50개, 최대 깊이 2 (단순한 모델)
    # 초금: 트리 = 질문을 계속 나눠가며 답을 찾는 "스무고개 다이어그램".
    # 초금: n_estimators = 이 스무고개를 몇 명이 함께 풀지 (투표 인원).
    # 초금: max_depth    = 질문을 얼마나 깊이 파고들지 (2면 질문 2번, 5면 질문 5번).
    {"n_estimators": 100, "max_depth": 3},                  # 트리 100개, 최대 깊이 3
    {"n_estimators": 200, "max_depth": 5},                  # 트리 200개, 최대 깊이 5
    {"n_estimators": 300, "max_depth": 4},                  # 트리 300개, 최대 깊이 4 (복잡한 모델)
]
# [AI] 수동 grid search에 가까운 형태. 프로덕션에서는 GridSearchCV/Optuna/Ray Tune 등을 함께 쓰며,
# [AI] MLflow의 nested runs 기능으로 부모 run 아래 각 trial을 자식 run으로 기록하는 패턴이 일반적.

# ── 3. 파라미터별 실험 실행 ───────────────────────────────────
# 초금: 위에서 정한 4개 설정을 하나씩 꺼내 "훈련 → 시험 → 성적 기록"을 반복합니다.
for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"  # ex) "n100_d3" → MLflow UI에서 식별용
    # 초금: 이번 실험의 이름표. 나중에 UI에서 어떤 조합인지 한 눈에 알아보기 위한 라벨.
    # [AI] run_name은 사람이 UI에서 구분하기 위한 라벨. 내부적으로는 자동 생성된 run_id(UUID)로 관리됨.
    with mlflow.start_run(run_name=run_name):                # 이 블록 안의 모든 기록이 하나의 run으로 묶임
        # 초금: "지금부터 새 실험 시작!" 선언. with 블록이 끝나면 자동으로 "실험 종료!"가 호출됩니다.
        # 초금: 덕분에 실험 경계가 헷갈릴 일 없고, 중간에 에러가 나도 안전하게 마무리됩니다.
        # [AI] start_run을 with 문으로 쓰면:
        # [AI]   1) 블록 진입 시 새 run 생성 & 현재 run으로 활성화
        # [AI]   2) 블록 종료 시 자동으로 end_run() 호출 → 예외 발생 시에도 run 상태가 FAILED로 마감됨
        # [AI] → 리소스 누수 없이 깔끔하게 실험 경계를 관리

        # 전처리(StandardScaler) + 모델을 하나의 파이프라인으로 구성
        # Pipeline 사용 시 predict()만 해도 자동으로 스케일링 후 예측
        # 초금: 파이프라인 = "빨래 → 헹굼 → 건조"처럼 공정을 묶은 것.
        # 초금: 여기선 "스케일링 → 랜덤포레스트"를 한 덩어리로 묶었습니다.
        pipe = Pipeline([
            ("scaler", StandardScaler()),                    # 특성값을 평균0, 표준편차1로 정규화
            # 초금: 1단계 "스케일러" — 숫자들을 같은 눈금으로 맞춰줌.
            ("clf", RandomForestClassifier(**params, random_state=42))  # 랜덤포레스트 (random_state로 재현성 보장)
            # 초금: 2단계 "분류기(clf = classifier)" — 실제 예측을 담당하는 모델.
            # 초금: **params는 위 param_list의 dict를 "key=value 형태로 풀어서 넣기" 트릭.
            # [AI] **params: dict 언패킹 — {"n_estimators": 50, "max_depth": 2}가
            # [AI] n_estimators=50, max_depth=2 키워드 인자로 전달됨.
        ])
        pipe.fit(X_train, y_train)                           # 학습: 스케일러 fit + 모델 학습 한 번에 처리
        # 초금: fit = "공부시키기". X_train(문제)과 y_train(정답)을 보여주며 규칙을 학습.
        # [AI] Pipeline.fit 내부 흐름:
        # [AI]   1) scaler.fit_transform(X_train) → 평균·표준편차 계산 후 스케일 적용
        # [AI]   2) clf.fit(scaled_X_train, y_train) → 랜덤포레스트 학습
        # [AI] 테스트 시에는 fit_transform이 아닌 transform만 호출되어 data leakage 방지.

        acc = accuracy_score(y_test, pipe.predict(X_test))   # 테스트셋으로 정확도 평가
        # 초금: predict = "시험 보기" → 예측 답안 생성.
        # 초금: accuracy_score(정답, 예측) = 몇 % 맞혔는지 계산.
        # [AI] pipe.predict(X_test) 내부에서 scaler.transform → clf.predict가 자동 연결됨.

        mlflow.log_params(params)                            # n_estimators, max_depth를 MLflow에 기록
        # 초금: "이번 실험에서 썼던 설정값들을 노트에 적어둬" 라는 뜻.
        # [AI] log_params(dict): 여러 파라미터를 한 번에 기록. 단건은 log_param(key, value).
        # [AI] 파라미터는 한 run 내에서 덮어쓰기 불가 (불변) — 실험 재현성 보장 원칙.
        mlflow.log_metric("accuracy", acc)                   # 정확도를 MLflow에 기록 (나중에 비교 가능)
        # 초금: "이번 실험의 성적(정확도)도 노트에 적어둬".
        # [AI] log_metric은 step 파라미터를 받아 시계열 기록 가능 (예: epoch별 loss).
        # [AI] 여기서는 단일 값만 기록하므로 step 생략 → 기본 step=0.

        # 모델을 MLflow artifact로 저장 → name="model"이 predict.py의 URI 조합과 일치해야 함
        model_info = mlflow.sklearn.log_model(pipe, name="model")
        # 초금: 학습이 끝난 모델을 "실험 첨부파일"처럼 저장. 나중에 꺼내서 그대로 재사용 가능.
        # [AI] log_model이 하는 일:
        # [AI]   1) pipe 객체를 pickle로 직렬화하여 artifact 스토어의 "model/" 폴더에 저장
        # [AI]   2) MLmodel 메타파일 생성 (flavor, 시그니처, python 버전 등 기록)
        # [AI]   3) requirements.txt / conda.yaml 자동 생성 (현재 env 기준)
        # [AI] 반환된 model_info.model_uri 형식: "runs:/<run_id>/model"
        # [AI] → 이 URI로 나중에 mlflow.sklearn.load_model(uri)로 불러올 수 있음.

        # 이 run의 결과를 리스트에 저장 (for문 끝난 후 최고 모델 선택에 사용)
        # 초금: MLflow에 기록하는 것과 별개로, 파이썬 코드 안에서도 결과를 모아둡니다.
        # 초금: 나중에 "누가 최고였지?" 판단을 코드로 빠르게 하기 위함.
        run_results.append({
            "run_name": run_name,
            "accuracy": acc,
            "model_uri": model_info.model_uri                # ex) runs:/abc123/model
        })
        print(f"  {run_name}: {acc:.4f} | uri: {model_info.model_uri}")

# ── 4. 최고 모델 선택 ─────────────────────────────────────────
# accuracy 기준으로 가장 높은 결과를 가진 딕셔너리 반환
# 초금: 4명의 학생(실험) 중 시험 점수가 가장 높은 1등을 찾는 단계.
best = max(run_results, key=lambda x: x["accuracy"])
# 초금: max()에 key를 주면 "정확도로 비교해서 최댓값 찾아" 라는 뜻이 됩니다.
# [AI] max(iterable, key=func): 각 원소에 func을 적용한 결과가 가장 큰 원소를 반환.
# [AI] key=lambda x: x["accuracy"] → 각 dict의 accuracy 값으로 비교.
# [AI] 동률이 있으면 "먼저 나온" 원소가 선택됨 (max의 안정적 동작 규칙).
print(f"🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")

# ── 5. Model Registry에 등록 ──────────────────────────────────
# run URI → Registry로 복사 (버전 자동 부여됨)
# 초금: "실험실에서 뽑은 최고 선수를 '선수 등록소'에 정식 등록" 하는 단계.
# 초금: 이때 자동으로 버전 번호(v1, v2, v3...)가 붙어서 나중에 "그때 그 v3 선수 불러!" 라고 부를 수 있습니다.
# [AI] MLflow Model Registry 개념:
# [AI]   - Experiment/Run: "실험 기록"용 — 수많은 trial이 쌓이는 곳
# [AI]   - Registry: "배포 관리"용 — 프로덕션 후보 모델만 선별해 버전/alias로 관리
# [AI]   - 하나의 모델 이름(name) 아래 Version 1, 2, 3...이 쌓이고,
# [AI]     alias(production, staging)로 특정 버전을 가리킨다 → 코드 수정 없이 배포 버전 교체 가능.
registered = mlflow.register_model(
    model_uri=best["model_uri"],                             # 최고 모델의 run URI
    # 초금: "어떤 모델을 등록할지" — 최고 선수의 위치(주소)를 알려줍니다.
    name="iris_classifier"                                   # Registry에서 사용할 모델 이름
    # 초금: 등록소에서 쓸 "선수 이름". 이름이 같으면 새 버전으로 추가되고, 처음이면 새로 생성됩니다.
    # [AI] name이 Registry에 없으면 자동 생성, 있으면 새 Version이 추가됨.
)
print(f"✅ 등록 완료! Version: {registered.version}")
# 초금: 등록 후 받는 영수증에 "몇 번째 버전인지" 찍혀 나옵니다.
# [AI] registered.version: 방금 생성된 버전 번호 (첫 등록이면 1, 이후 2, 3, ... 자동 증가).

# ── 6. Production alias 설정 ──────────────────────────────────
# 초금: "이번에 등록한 v1을 '현역 선수(production)'로 임명한다!" 는 단계.
# 초금: 나중에 예측할 때 "production 붙은 선수 불러!" 라고 간단히 호출할 수 있게 됩니다.
client = MlflowClient()                                      # Registry 조작을 위한 클라이언트
# 초금: alias(별명) 붙이기는 기본 리모컨(mlflow.xxx)으로는 안 되고, 고급 리모컨(MlflowClient)이 필요합니다.
# [AI] MlflowClient는 set_tracking_uri()에 설정된 서버를 기본으로 사용.
# [AI] 인자로 tracking_uri, registry_uri를 명시할 수도 있음(서로 다른 서버 사용 시).
client.set_registered_model_alias(
    name="iris_classifier",                                  # 대상 모델 이름
    # 초금: 어느 선수(모델 이름)에게 배지를 달아줄지.
    alias="production",                                      # 붙일 별명 (predict.py에서 @production으로 참조)
    # 초금: 배지 이름. "production"은 "현재 운영 중인 공식 선수"라는 뜻으로 관행적으로 씁니다.
    version=registered.version                               # 방금 등록된 버전에 alias 연결
    # 초금: 몇 번 버전에게 배지를 달지. 여기선 방금 등록한 버전.
)
# 초금: 이후 나중에 v2, v3을 또 등록하고, 그 중 하나에 production 배지를 "옮겨" 달면
# 초금: 자동으로 이전 배지는 떨어지고 새 버전이 공식 선수가 됩니다 → 코드 수정 없이 배포 교체!
# [AI] alias 시스템 (MLflow 2.x 이후 권장 방식, 이전의 "stage"를 대체):
# [AI]   - 하나의 alias는 정확히 하나의 버전을 가리킴
# [AI]   - 같은 alias를 다른 버전에 재할당하면 자동으로 이전 연결이 끊기고 이동(atomic swap)
# [AI]   - 클라이언트는 "models:/iris_classifier@production" 형식으로 조회
# [AI]     → 배포 코드는 alias만 바라보고, 실제 어떤 버전이 가리켜지는지는 운영자가 제어
print(f"🚀 production alias → Version {registered.version}")
