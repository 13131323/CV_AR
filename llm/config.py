"""
Gemini API 설정

환경변수 GEMINI_API_KEY가 필요합니다.
예) 프로젝트 루트의 .env 파일에 GEMINI_API_KEY="your-api-key-here" 작성
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수가 설정되지 않았습니다. "
        '프로젝트 루트에 .env 파일을 만들고 GEMINI_API_KEY="..." 로 설정하세요.'
    )