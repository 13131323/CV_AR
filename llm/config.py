"""
OpenAI API 설정

환경변수 OPENAI_API_KEY가 필요합니다.
예) 프로젝트 루트의 .env 파일에 OPENAI_API_KEY="your-api-key-here" 작성
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
        '프로젝트 루트에 .env 파일을 만들고 OPENAI_API_KEY="..." 로 설정하세요.'
    )