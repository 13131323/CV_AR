CV_AR: 실시간 장면 이해 기반 환경 적응형 AR 아바타 행동 생성 연구
본 프로젝트는 파이썬(백엔드/AI 비전 처리 및 LLM 연동)과 유니티(프론트엔드/AR 환경)로 구성되어 있습니다. 프로젝트를 클론한 뒤 아래의 전체 세팅 및 실행 가이드를 꼼꼼히 따라 환경을 구성해 주시기 바랍니다.

🛠️ 1. 사전 셋업 (최초 1회 필수 진행)
1-1. 프로젝트 클론 및 Git LFS (대용량 파일)
이 프로젝트는 대용량 AI 모델 가중치 파일(.pt)과 유니티 에셋을 공유하기 위해 Git LFS를 사용합니다. 만약 sam_b.pt 파일 용량이 1KB 미만으로 보인다면 Git LFS가 정상 작동하지 않은 것입니다. 아래 명령어로 LFS를 설치하고 다시 받아주세요.

git lfs install
git lfs pull
1-2. 파이썬 (백엔드/AI) 환경 구성
파이썬 환경은 가상환경(Virtual Environment)을 통해 동일한 패키지 버전을 공유합니다.

# 가상환경 생성 (파이썬 3.9 이상 권장)
python -m venv .venv

# 가상환경 활성화 (Mac/Linux)
source .venv/bin/activate
# Windows의 경우: .venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
1-3. API 키 설정 (.env)
프로젝트 루트에 .env.example 파일을 복사하여 .env 파일을 생성하고 본인의 API 키를 기입하세요.

GEMINI_API_KEY="본인의_제미나이_키"
GEMINI_MODEL="gemini-2.5-flash"
OPENAI_API_KEY="본인의_오픈AI_키"
OPENAI_MODEL="gpt-4o-mini"
HF_TOKEN="선택사항_허깅페이스_토큰"
1-4. 유니티 (프론트엔드/AR) 환경 구성
Unity Hub에서 Unity 6000.3.17f1 버전을 설치합니다. (버전이 다르면 씬이 깨질 수 있습니다.)
Unity Hub에서 프로젝트 내부의 unity/CV_AR 폴더를 엽니다.
초기 로딩 시 Library 폴더가 자동 생성되느라 시간이 다소 걸릴 수 있습니다.
주의: 유니티 프로젝트 실행을 위해서는 최상단에 있는 UnityScripts 폴더가 반드시 포함되어 있어야 합니다.
🚨 2. 하드웨어 세팅 및 캘리브레이션 (매우 중요)
2-1. 웹캠 연결 및 권한
실시간 비전 파이프라인 구동을 위해 PC에 웹캠이 연결되어 있어야 합니다. Mac 사용자의 경우 터미널에서 카메라 접근 권한을 허용해 주셔야 합니다.

2-2. 카메라 캘리브레이션 (필수)
현재 코드는 1280x720 해상도에 맞춘 임시 카메라 행렬 값(f_x=900)을 사용하고 있습니다. 본인의 카메라 환경에서 체커보드 캘리브레이션을 수행하지 않으면 AR 공간상에서 아바타와 실제 사물 간에 수십 cm의 좌표 오차가 발생합니다.

vision/stream.py 의 camera_matrix
vision/spatial/transformer.py 의 f_x, f_y 위 두 파일의 수치를 캘리브레이션 결과 값으로 교체한 뒤 실행해 주세요.
🚀 3. 프로젝트 실행 방법
[모드 A] 유니티 메인 서버 실행 (실제 AR 연동)
서버 구동 (터미널) 가상환경을 활성화한 후, 반드시 프로젝트 루트 폴더에서 모듈 방식(-m)으로 아래 명령어를 실행하세요.

source .venv/bin/activate
python -m llm.server_websocket
(주의: python llm/server_websocket.py로 실행 시 패키지 인식 오류가 발생합니다.)

터미널에서 모델 로딩이 완료되고 OpenCV 웹캠 창이 정상적으로 뜨면, 유니티 에디터로 돌아가 메인 씬의 Play(▶) 버튼을 눌러 통신을 시작합니다.

[모드 B] 파이썬 단독 실시간 데모 (디버그용)
유니티를 켜지 않고 파이썬 비전 파이프라인과 LLM 추론 결과를 터미널 로그로만 확인하고 싶을 때 사용합니다.

source .venv/bin/activate
python -m llm.realtime_demo
