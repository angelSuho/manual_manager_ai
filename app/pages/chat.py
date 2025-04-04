import base64, re, markdown
import torch

from config.config import IMAGE_PATTERN, client, st
from config import streamlit_config
from services.ai_service import ask_lang_graph_agent, summarize_text
from services.tts_service import generate_tts
from streamlit_current_location import current_position

streamlit_config.apply_streamlit_settings()
streamlit_config.apply_custom_css()

def convert_links(text):
    url_pattern = re.compile(r'(https?://[^\s]+)')
    return url_pattern.sub(r'<a href="\1" target="_blank">\1</a>', text)


position = current_position()
# get_geolocation 함수를 호출하여 위치 정보를 받아옵니다.

if position is not None:
    # 세션에 위치 저장 (키 이름은 "user_location" 등으로 자유롭게 지정)
    st.session_state["user_location"] = position

# URL 쿼리 파라미터 추출 (st.query_params 사용)
params = st.query_params

if "car" in params:
    st.session_state.car_type = params["car"]

if "car_type" not in st.session_state:
    st.warning("차량을 먼저 선택해주세요.")
    st.stop()

torch.classes.__path__ = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = None

# -----------------------------
# 사이드바 디자인용 CSS 삽입
# -----------------------------
st.markdown("""
<style>
/* 사이드바 전체 스타일 */
[data-testid="stSidebar"] {
    background-color: #F9F9F9 !important;
    padding: 1rem;
}

/* 로고 + 타이틀 영역 */
.sidebar-header {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
}

.sidebar-logo {
    width: 40px;
    margin-right: 0.5rem;
}

.sidebar-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0;
    padding: 0;
}

/* 현재 차량 카드 스타일 */
.car-info-card {
    background-color: #ECECEC;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 1rem;
    text-align: center;
    border: 1px solid #DDD;
}
.car-info-card h3 {
    margin-top: 0;
    margin-bottom: 5px;
    color: #333;
}
.car-info-card .car-name {
    font-size: 1.1rem;
    font-weight: bold;
    color: #555;
}

/* 입력 섹션 스타일 */
.input-section {
    margin-top: 1rem;
    padding: 15px;
    background-color: #FFF;
    border: 1px solid #EEE;
    border-radius: 8px;
}
.input-section h4 {
    margin-top: 0;
    margin-bottom: 1rem;
    color: #333;
}

/* 메인 헤더 스타일 (상단 고정 영역) */
.fixed-header {
    background-color: #FFF;
    padding: 10px 0;
    border-bottom: 1px solid #DDD;
    margin-bottom: 1rem;
}

/* 메시지 스타일 */
.user-message {
    background-color: #E6F4FF;
    color: #000;
    padding: 10px;
    border-radius: 8px;
    margin: 0.5rem 0;
    max-width: 80%;
    word-wrap: break-word;
}

.bot-message {
    background-color: #F2F2F2;
    padding: 10px;
    border-radius: 8px;
    margin: 0.5rem 0;
    max-width: 80%;
    word-wrap: break-word;
}

.custom-image {
    max-width: 400px;
    border-radius: 5px;
    margin-left: auto;
}
            
.sidebar-header .sidebar-logo {
    width: 25px;   /* 원하는 너비로 조정 */
    height: auto;
    margin-right: 0.5rem;
}

/* FAQ 섹션용 스타일 */
.faq-container {
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 사이드바 구성
# -----------------------------
with st.sidebar:
    # 로고와 타이틀
    st.markdown("""
    <div class="sidebar-header">
        <img src="https://kcc-llm.s3.ap-northeast-2.amazonaws.com/logo.png" class="sidebar-logo"/>
        <h1 class="sidebar-title">KCC User Assistant</h1>
    </div>
    """, unsafe_allow_html=True)

    # 현재 차량 정보 (카드 스타일)
    st.markdown(f"""
    <div class="car-info-card">
        <h3>현재 차량</h3>
        <div class="car-name">Mercedes-Benz {st.session_state.car_type.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    # 음성/이미지 업로드 영역 (카드 느낌)
    st.markdown("""
        <h4>음성 입력 및 이미지 첨부</h4>
    """, unsafe_allow_html=True)

    st.session_state.audio_file = st.audio_input(
        "마이크 아이콘을 눌러 음성으로 입력하세요.",
        label_visibility="collapsed"
    )

    st.session_state.uploaded_image = st.file_uploader(
        "이미지를 업로드하여 질문해보세요.",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

# -----------------------------
# 상단 고정 헤더
# -----------------------------
st.markdown('<div class="fixed-header"><h3 style="margin:0; padding-bottom: 7px;">KCC User Assistant</h3></div>', unsafe_allow_html=True)


def display_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
            # 이미지가 있을 경우
            if msg.get("image"):
                st.markdown(
                    f"""
                    <div>
                        <img src='data:image/png;base64,{msg['image']}' class='custom-image' style='float:right; margin-top:8px; width:300px;'/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        elif msg["role"] == "assistant":
            content_html = f"<div><div class='bot-message'>{msg['content']}</div>"
            if msg.get("image") and msg["image"].startswith("http"):
                content_html += f"<br><img src='{msg['image']}' width='500' style='margin-top:8px;'/>"
            content_html += "</div>"
            st.markdown(content_html, unsafe_allow_html=True)
            
            # 오디오 플레이어
            if "tts" in msg and msg["tts"]:
                tts_audio_b64 = msg["tts"]
                audio_html = f"""
                <audio controls style="width:500px;">
                    <source src="data:audio/mp3;base64,{tts_audio_b64}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

def handle_user_input(user_prompt=None, audio_file=None):
    if audio_file:
        transcript_result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko"
        )
        user_prompt = transcript_result.text
    
    data_url = None
    if st.session_state.uploaded_image:
        image_bytes = st.session_state.uploaded_image.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded_image}"
    if user_prompt:
        encoded_image = None
        if st.session_state.uploaded_image:
            encoded_image = base64.b64encode(st.session_state.uploaded_image.getvalue()).decode()

        st.session_state.messages.append({
            "role": "user", 
            "content": user_prompt,
            "image": encoded_image
        })

        st.markdown(f"<div class='user-message'>{user_prompt}</div>", unsafe_allow_html=True)
        if st.session_state.get("uploaded_image"):
            st.markdown(
                f"<div style='display:flex;justify-content:flex-end;'>"
                f"<img src='data:image/png;base64,{base64.b64encode(st.session_state.uploaded_image.getvalue()).decode()}' width='300'/></div><br>",
                unsafe_allow_html=True
            )

        recent_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[-10:]
            if msg["content"].strip()
        ]

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            with st.status("답변을 생성중입니다...", expanded=True) as status:
                try:
                    # 이전 히스토리를 함께 전달하여 문맥 유지
                    resp = ask_lang_graph_agent(user_prompt, data_url, recent_history)["messages"]
                    # print(resp)
                    assistant_response = resp[-1].content if isinstance(resp, list) else resp
                    if data_url:
                        assistant_response += f"\n\n 점검을 위해 서비스 센터에 방문 예정이라면\n 👉'서비스 센터' 라고 입력해주세요!"
                except Exception as e:
                    st.error(f"AI 응답 생성 중 오류가 발생했습니다: {e}")
                    return
                status.update(label="답변 생성 완료!", state="complete", expanded=False)
        try:
            related_image = None
            match = re.search(IMAGE_PATTERN, assistant_response, flags=re.IGNORECASE)
            if match:
                related_image = match.group(0)
                assistant_response = re.sub(IMAGE_PATTERN, '', assistant_response, count=1, flags=re.IGNORECASE)
            # 마크다운 처리
            assistant_response = markdown.markdown(assistant_response, extensions=['extra', 'sane_lists', 'nl2br'])

            st.markdown(f"<div class='bot-message' style='width:700px;'>{assistant_response}</div>", unsafe_allow_html=True)
            if related_image:
                st.image(related_image, width=500)
            
            # TTS 생성
            with st.spinner("TTS 생성 중입니다..."):
                if assistant_response.strip():
                    summary_text = summarize_text(assistant_response)
                    tts_audio_b64 = generate_tts(summary_text, lang="ko")
                else:
                    tts_audio_b64 = None

                if tts_audio_b64:
                    audio_html = f"""
                    <audio controls style="width:500px;">
                        <source src="data:audio/mp3;base64,{tts_audio_b64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "tts": tts_audio_b64,
                "image": related_image
            })

        except Exception as e:
            st.error(f"최종 응답 생성 중 오류가 발생했습니다: {e}")


# -----------------------------
# FAQ 버튼 영역 (자주 묻는 질문)
# -----------------------------
# FAQ 영역을 감싸는 박스 + 제목 추가
st.markdown("""
<div class="faq-container">
    <h4>자주 묻는 질문 (FAQ)</h4>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("경고등 종류가 뭐가 있나요?", key="faq1", use_container_width=True):
        st.session_state.user_prompt = "경고등 종류가 뭐가 있나요?"
    if st.button("근처에 서비스 센터는 어디 있나요?", key="faq3", use_container_width=True):
        st.session_state.user_prompt = "근처 서비스 센터"

with col2:
    if st.button("타이어 공기압 경고등이 켜졌을 때 조치는?", key="faq2", use_container_width=True):
        st.session_state.user_prompt = "타이어 공기압 경고등이 켜졌을 때 조치는?"
    if st.button("파워 스티어링 시스템에 문제가 발생하면 어떻게 대처해야 하나요?​", key="faq4", use_container_width=True):
        st.session_state.user_prompt = "파워 스티어링 시스템에 문제가 발생하면 어떻게 대처해야 하나요?​"

# FAQ 영역 닫기
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 대화 표시 및 채팅 입력
# -----------------------------
display_messages()

user_prompt = st.chat_input("KCC에게 질문해주세요.", key="main_chat_input")

if not user_prompt:
    user_prompt = st.session_state.user_prompt

if st.session_state.audio_file:
    handle_user_input(audio_file=st.session_state.audio_file)
    st.session_state.audio_file = None
    st.session_state.user_prompt = None
elif user_prompt:
    handle_user_input(user_prompt)
    st.session_state.user_prompt = None
