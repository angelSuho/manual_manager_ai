# Description: 텍스트를 음성으로 변환하는 기능을 제공하는 모듈
from gtts import gTTS
import io
import base64

def generate_tts(text: str, lang='ko'):
    try:
        tts = gTTS(text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return base64.b64encode(audio_fp.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"TTS 오류: {e}")
        return None
