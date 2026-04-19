# =============================================================================
# 초금: 이 파일이 뭘 하는 프로그램인지 — 초보자를 위한 한 장 요약
# =============================================================================
# 초금: "MLflow에 저장해 둔 Iris 꽃 분류 모델을 인터넷(HTTP)으로 불러 쓸 수 있게
# 초금:  작은 웹 서버를 띄우는 프로그램"입니다. (= "모델 배포용 API 서버")
#
# 초금: 일상 비유로 설명하면:
# 초금:   • 학습(train)은 "요리 레시피 만들기"
# 초금:   • MLflow 등록(registry)은 "레시피를 책으로 출간"
# 초금:   • app.py(FastAPI)는 "손님이 주문할 수 있는 식당을 오픈"하는 단계
# 초금:     → 손님이 "4가지 꽃잎·꽃받침 수치"를 주문서로 제출하면
# 초금:     → 주방(model)이 "이 꽃은 무슨 종인가?"를 즉석에서 답해줌
#
# 초금: 제공하는 엔드포인트(메뉴) 2가지:
# 초금:   ① GET  /          → "서버 살아있어?" 확인용 (헬스 체크)
# 초금:   ② POST /predict   → 실제 예측 주문 (입력: 수치 4개, 출력: 예측 + 확률)
#
# 초금: 핵심 구성요소:
# 초금:   • FastAPI    = 파이썬 웹 API 프레임워크 (Flask보다 빠르고 자동 문서화됨)
# 초금:   • Pydantic   = 입력 JSON을 자동 검증/변환 (타입 안전)
# 초금:   • MLflow     = Registry에서 @production alias 모델 로드
# 초금:   • numpy      = 모델 입력 형태(2차원 배열)로 변환
#
# 초금: 실행 방법:
# 초금:   uvicorn app:app --reload --port 8000
# 초금:   → 브라우저에서 http://localhost:8000/docs 로 Swagger UI 자동 생성됨
#
# 초금: 테스트 예시 (터미널에서):
# 초금:   curl -X POST http://localhost:8000/predict \
# 초금:        -H "Content-Type: application/json" \
# 초금:        -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# =============================================================================

import os
# 초금: 환경변수(OS에 저장된 설정값) 읽기용. MLflow 서버 주소를 환경변수로 받기 위함.
import mlflow.sklearn
# 초금: MLflow에서 scikit-learn 모델을 로드하는 모듈 (load_model 함수 제공).
# [AI] mlflow.sklearn.load_model은 내부적으로 MLmodel 메타파일을 읽어
# [AI] flavor 정보를 확인하고 pickle을 역직렬화해 원본 객체(Pipeline)를 복원한다.
import numpy as np
# 초금: 숫자 배열을 다루는 라이브러리. scikit-learn은 2D 배열(행렬) 입력을 요구해서 필요.
from fastapi import FastAPI
# 초금: FastAPI = "웹 API를 10분 만에 만들 수 있는" 파이썬 프레임워크.
# [AI] FastAPI는 Starlette(ASGI) + Pydantic 기반. async 지원, 자동 OpenAPI/Swagger 생성,
# [AI] 타입 힌트를 통한 런타임 검증이 핵심 특징.
from pydantic import BaseModel
# 초금: pydantic = "JSON 입력 자동 검증기". 잘못된 타입이 들어오면 422 에러 자동 반환.
# [AI] BaseModel 상속 시 __init__/.dict()/.json() 등이 자동 생성되고,
# [AI] 필드 타입 힌트를 기반으로 validation·coercion 수행 (str → float 등 자동 변환).

# 1. MLflow 설정 및 모델 로드
# 초금: 서버가 "어느 MLflow 창고에서 모델을 꺼내올지" 결정하는 단계.
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5080")
# 초금: 환경변수 MLFLOW_TRACKING_URI가 있으면 그 값, 없으면 기본값(포트 5080) 사용.
# [AI] 코드에 URI를 하드코딩하지 않고 환경변수로 주입받는 "12-Factor App" 원칙.
# [AI] 동일 이미지/코드를 dev/staging/prod 서버에서 재사용 가능.
mlflow.set_tracking_uri(tracking_uri)
# 초금: "앞으로 mlflow 함수는 이 주소로 연결해!" 라고 전역 설정하는 줄.

app = FastAPI(title="Iris Classifier API", version="1.0")
# 초금: FastAPI 앱 객체 생성. title/version은 /docs 자동 문서에 표시됩니다.
# [AI] 이 app 변수가 ASGI 엔트리포인트. uvicorn app:app 으로 실행 시
# [AI] "app.py 파일의 app 객체를 사용"하라는 의미.

# 서버 시작 시 모델을 미리 로드합니다.
# 초금: 서버가 처음 켜질 때 딱 한 번만 모델을 메모리에 로드해 둡니다.
# 초금: → 요청이 들어올 때마다 매번 로드하면 느려지므로, 미리 준비해두는 패턴.
# [AI] 전역 스코프에서 로드 = 모듈 import 시점에 1회 실행 = 워커 프로세스당 1회.
# [AI] 워커를 여러 개(--workers 4) 띄우면 프로세스마다 독립된 모델 객체를 갖는다.
# [AI] 대안 패턴: FastAPI의 lifespan 이벤트 핸들러 사용 (더 명시적, 리소스 정리 용이).
try:
    model = mlflow.sklearn.load_model("models:/iris_classifier@production")
    # 초금: "models:/모델명@alias" 형식 → 버전 번호 몰라도 alias로 현재 운영 모델 참조.
    # 초금: @production alias가 가리키는 최신 버전을 자동으로 가져옵니다.
    # [AI] 배포 교체 시: Registry에서 alias를 다른 version으로 이동시키고 서버만 재시작하면 끝
    # [AI] → 코드 변경·빌드·배포 파이프라인 없이도 모델 교체 가능.
    print("✅ Production 모델 로드 완료!")
except Exception as e:
    # 초금: 모델 로드 실패(서버 꺼짐, alias 없음 등) 시 앱은 그래도 시작되지만
    # 초금: /predict 호출 시 NameError가 터지게 됨.
    print(f"❌ 모델 로드 실패: {e}")
    # [AI] 현 구현은 fail-open 방식 — 서버는 뜨지만 predict는 500 리스크.
    # [AI] 프로덕션 권장: raise로 앱 시작 자체를 실패시키거나, health_check에서
    # [AI] 모델 상태를 반영해 K8s liveness/readiness probe가 실패하도록 만든다.

# 2. 데이터 구조 정의
# 초금: 손님이 제출할 주문서(JSON)의 양식을 미리 정의합니다.
# 초금: 이 양식에서 벗어난 요청이 오면 FastAPI가 자동으로 422 에러로 거부.
class IrisInput(BaseModel):
    sepal_length: float    # 초금: 꽃받침 길이 (cm)
    sepal_width: float     # 초금: 꽃받침 너비 (cm)
    petal_length: float    # 초금: 꽃잎 길이 (cm)
    petal_width: float     # 초금: 꽃잎 너비 (cm)
    # [AI] 필드 타입이 float로 선언돼 있어 "5" (int) → 5.0 (float) 자동 변환,
    # [AI] "abc" 같은 비숫자 문자열은 ValidationError로 거부됨.
    # [AI] 범위 검증이 필요하면 Field(gt=0, le=10) 형태로 제약 가능.

iris_classes = ["setosa", "versicolor", "virginica"]
# 초금: 모델이 반환하는 숫자(0,1,2)를 사람이 읽을 수 있는 이름으로 매핑.
# 초금: 0=setosa, 1=versicolor, 2=virginica (sklearn iris 데이터셋의 표준 순서)

# 3. 엔드포인트 설정
# 초금: endpoint = "API의 개별 메뉴". 여기서는 2개(/와 /predict)를 만듭니다.

@app.get("/")
# 초금: @app.get("/") = "GET 방식으로 / 주소에 요청이 오면 아래 함수를 실행"
def health_check():
    # 초금: 헬스 체크 = "서버 살아있어?" 확인용. 보통 로드밸런서·모니터링이 주기적으로 호출.
    return {"status": "ok", "model": "iris_classifier@production"}
    # 초금: dict를 반환하면 FastAPI가 자동으로 JSON 직렬화해서 응답합니다.
    # [AI] 정식 구현이라면 model 객체 로드 여부까지 반영해야 의미있는 헬스체크.

@app.post("/predict")
# 초금: @app.post("/predict") = "POST 방식으로 /predict에 주문서가 오면 아래 함수를 실행"
# 초금: GET은 "조회", POST는 "데이터 제출" 용도라는 HTTP 관례를 따름.
# 초금: ※ 브라우저 주소창에 /predict 입력하면 GET이라 405 Method Not Allowed가 나옵니다.
# 초금: ※ 테스트는 /docs(Swagger UI) 또는 curl/Postman으로 하세요.
def predict(data: IrisInput):
    # 초금: 함수 인자에 `data: IrisInput` 이라고 적기만 하면:
    # 초금:   1) FastAPI가 요청 본문(JSON)을 자동으로 IrisInput 객체로 변환
    # 초금:   2) 타입이 안 맞으면 자동으로 422 에러 반환
    # 초금:   3) /docs에서 자동으로 입력 예시 생성
    # [AI] 이 패턴이 FastAPI의 핵심 매력 — "타입 힌트가 곧 스키마 + 검증 + 문서".

    # 입력 데이터를 numpy 배열로 변환
    # 초금: scikit-learn은 "샘플 N개가 담긴 2D 배열" 형태만 받습니다.
    # 초금: 한 건만 예측할 때도 [[값, 값, 값, 값]] 처럼 이중 대괄호 필수!
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    # [AI] shape = (1, 4). sklearn의 predict/predict_proba는 2D(n_samples, n_features)를 기대.
    # [AI] 1D 배열(shape=(4,))로 넣으면 DeprecationWarning/ValueError가 발생.

    # 모델 예측
    # 초금: 한 건만 예측하지만 결과도 배열로 나오므로 [0]으로 첫 번째 요소만 꺼냅니다.
    prediction = model.predict(features)[0]
    # 초금: predict()       → 클래스 번호만 반환 (예: 0 또는 1 또는 2)
    probability = model.predict_proba(features)[0].tolist()
    # 초금: predict_proba() → 각 클래스의 확률 반환 (예: [0.98, 0.01, 0.01])
    # 초금: .tolist()는 numpy 배열을 파이썬 리스트로 변환 → JSON 직렬화 가능하게.
    # [AI] numpy scalar/array는 기본 json.dumps가 처리 못 해서 tolist()/int() 변환 필요.

    # 결과 반환 (이 return문이 반드시 있어야 null이 안 나옵니다!)
    # 초금: return이 빠지면 함수가 None을 반환 → 응답 body가 "null"로 찍힙니다.
    return {
        "prediction": int(prediction),              # 초금: numpy.int64 → 파이썬 int
        "class_name": iris_classes[prediction],     # 초금: 숫자 → 사람이 읽는 이름
        "probability": probability                  # 초금: 각 클래스 확률 리스트
    }
    # [AI] 응답 스키마도 Pydantic BaseModel로 정의해 response_model=... 으로 지정하면
    # [AI] OpenAPI 문서에 명확한 응답 구조가 나타나고, 실수로 필드가 빠지는 것을 방지 가능.
