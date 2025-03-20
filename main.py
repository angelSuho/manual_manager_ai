import streamlit as st
import os, base64, re
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
import openai

from ai_agent import ask_lang_graph_agent, car_type, IMAGE_PATTERN
from tts_utils import generate_tts

# 초기화
load_dotenv()

local_path = "./"
input_path = local_path + "/input"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit 페이지 구성
st.set_page_config(
    page_title="KCC Auto Manager", 
    page_icon=Image.open(local_path + "/images/favicon.ico"),
    layout="wide"
)

# CSS 스타일 적용
st.markdown("""
<style>
.sidebar-room {
    padding: 10px 15px;
    border-radius: 5px;
    margin-bottom: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.sidebar-room:hover {
    background-color: #ECECEC;
}
.sidebar-room.selected {
    background-color: #E3E3E3;
}

.chat-container {
    padding: 15px;
}
.user-message {
    text-align: right;
    padding: 10px;
    background-color: #007bff;
    color: white;
    border-radius: 15px 15px 0px 15px;
    margin-bottom: 10px;
    max-width: 70%;
    display: inline-block;
}
.bot-message {
    text-align: left;
    padding: 10px;
    background-color: #f1f1f1;
    color: black;
    border-radius: 15px 15px 15px 0px;
    margin-bottom: 10px;
    max-width: 70%;
    display: inline-block;
}
[data-testid="stMarkdownContainer"]:has(.user-message) {
    display: flex !important;
    justify-content: flex-end !important;
}
[data-testid="stMarkdownContainer"]:has(.bot-message) {
    display: flex !important;
    justify-content: flex-start !important;
}
.block-container {
    padding-top: 7rem !important;
    margin-top: -3rem !important;
}
[data-testid="stSidebar"] {
    padding-top: 2rem !important;
    margin-top: -3rem !important;
}
@media (max-width: 768px) {
    .bot-message, .user-message {
        max-width: 95% !important;
    }
}
</style>
""", unsafe_allow_html=True)

if "chat_rooms" not in st.session_state:
        st.session_state.chat_rooms = {
            "채팅방 1": [
                {"role": "user", "content": "안녕하세요!"},
                {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
            ],
            "채팅방 2": [
                {"role": "user", "content": "오늘 날씨 어때?"},
                {"role": "assistant", "content": "오늘은 맑고 따뜻합니다."},
            ],
        }
        st.session_state.current_room = "채팅방 1"

# 사이드바
with st.sidebar:
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(local_path + "/images/logo.png", width=40)
    with col2:
        st.markdown(
            "<h1 style='margin: 0; padding: 0; display: flex; align-items: center; height: 100%;'>KCC Auto Manager</h1>",
            unsafe_allow_html=True
        )
    st.subheader("현재 차량: Mercedes-Benz " + car_type)
    st.subheader("채팅방 목록")
    for room in st.session_state.chat_rooms.keys():
        if room == st.session_state.current_room:
            room_class = "sidebar-room selected"
        else:
            room_class = "sidebar-room"

        if st.markdown(f"<div class='{room_class}' onclick=''>{room}</div>", unsafe_allow_html=True):
            st.session_state.current_room = room

st.markdown(
    f"""
    <div style='display: flex; align-items: center;'>
        <h3 style='margin: 0; padding: 0;'>KCC Auto Manager</h3>
        <span style='font-size: 18px; font-weight: 600; margin-left: 8px;'>{st.session_state.current_room}</span>
    </div>
    """,
    unsafe_allow_html=True
)
# 메시지 출력 (HTML 기반 메시지)
def display_messages():
    for msg in st.session_state.messages:
        role_class = "user-message" if msg["role"] == "user" else "bot-message"
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
        
        if msg["role"] == "assistant":
            if msg.get("tts"):
                audio_bytes = base64.b64decode(msg["tts"])
                st.audio(audio_bytes, format="audio/mp3")
            if msg.get("image"):
                st.image(msg["image"], width=500)

display_messages()
# 사용자 입력 처리
def handle_user_input():
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    user_prompt = st.chat_input("메시지를 입력하세요")

    with st.expander("음성 입력 및 이미지 첨부 열기"):
        audio_file = st.audio_input("마이크 아이콘을 눌러 음성으로 입력하세요.")
        uploaded_image = st.file_uploader("이미지를 업로드하여 질문해보세요.", type=["png", "jpg", "jpeg", "gif"])

    if audio_file is not None:
        st.session_state.audio_file = audio_file
    if uploaded_image is not None:
        st.session_state.uploaded_image = uploaded_image

    combined_prompt = ""
    image_path = None

    if st.session_state.audio_file:
        with st.spinner("음성 인식 중..."):
            transcript_result = client.audio.transcriptions.create(
                model="whisper-1",
                file=st.session_state.audio_file,
                language="ko"
            )
        user_prompt = transcript_result.text

    if st.session_state.uploaded_image:
        image_path = os.path.join(input_path, st.session_state.uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(st.session_state.uploaded_image.getbuffer())
        combined_prompt += f"{image_path}\n\n"

    if user_prompt:
        combined_prompt += user_prompt
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        if st.session_state.uploaded_image:
            st.markdown(
                f"<div style='display:flex;justify-content:flex-end;'>"
                f"<img src='data:image/png;base64,{base64.b64encode(st.session_state.uploaded_image.getvalue()).decode()}' width='300'/></div>",
                unsafe_allow_html=True)
        st.write("")
        st.markdown(f"<div class='user-message'>{user_prompt}</div>", unsafe_allow_html=True)

        with st.spinner("답변을 생성 중입니다..."):
            try:
                assistant_response = ask_lang_graph_agent(combined_prompt)["messages"][-1].content
                related_image = None
                match = re.search(IMAGE_PATTERN, assistant_response, flags=re.IGNORECASE)
                if match:
                    related_image = match.group(0)
                    assistant_response = re.sub(IMAGE_PATTERN, '', assistant_response, count=1, flags=re.IGNORECASE)

                tts_audio_b64 = generate_tts(assistant_response, lang="ko") if assistant_response.strip() else None
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_response,
                    "tts": tts_audio_b64,
                    "image": related_image if related_image else None
                }
                st.session_state.messages.append(assistant_message)

                if related_image:
                    st.image(related_image, width=500)
                st.markdown(f"<div class='bot-message'>{assistant_response}</div>", unsafe_allow_html=True)

                if tts_audio_b64:
                    audio_bytes = base64.b64decode(tts_audio_b64)
                    st.audio(audio_bytes, format="audio/mp3")

                st.session_state.audio_file = None
                st.session_state.uploaded_image = None

            except Exception as e:
                st.error(f"AI 응답 생성 중 오류가 발생했습니다: {e}")

handle_user_input()
