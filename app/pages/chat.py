import os, base64, re, time
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config.config import IMAGE_PATTERN, client
from config import streamlit_config
from services.ai_service import ask_lang_graph_agent, st, summarize_text
from services.tts_service import generate_tts

streamlit_config.apply_streamlit_settings()
streamlit_config.apply_custom_css()

if "car_type" not in st.session_state:
    st.warning("차량을 먼저 선택해주세요.")
    st.stop()

torch.classes.__path__ = []
if st.session_state.get("clear_chat"):
    st.session_state.messages = []
    st.session_state.clear_chat = False
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = None

# 사이드바
with st.sidebar:
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("./data/images/logo.png", width=40)
    with col2:
        st.markdown(
            "<h1 style='margin: 0; padding: 0; display: flex; align-items: center; height: 100%;'>KCC Auto Manager</h1>",
            unsafe_allow_html=True
        )
    st.subheader("현재 차량: Mercedes-Benz " + st.session_state.car_type)
    
    st.subheader("음성 입력 및 이미지 첨부")
    st.session_state.audio_file = st.audio_input("마이크 아이콘을 눌러 음성으로 입력하세요.", label_visibility="collapsed")
    st.session_state.uploaded_image = st.file_uploader("이미지를 업로드하여 질문해보세요.", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    # st.markdown("### 바로가기")
    # st.markdown("[메르세데스-벤츠 공식 홈페이지](https://www.mercedes-benz.co.kr/)")
    # st.markdown("[메르세데스-벤츠 공식 사용 메뉴얼](https://www.mercedes-benz.co.kr/passengercars/services/manuals.html)")

st.markdown('<div class="fixed-header"><h3 style="margin:0; padding-bottom: 7px;">KCC Auto Manager</h3></div>', unsafe_allow_html=True)

# ------------------------------
# CLIP 모델 로드 (정책 체크용)
# ------------------------------
@st.cache_resource
def load_clip_model():
    clip_model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(clip_model_name)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    return model, processor

# 흐림(블러) 여부 판정 함수 (OpenCV 사용)
def is_blurry(pil_image, blur_threshold=550):
    """이미지가 특정 임계값(blur_threshold) 이하로 선명도가 낮으면 True 반환"""
    # PIL 이미지를 OpenCV용 NumPy 배열로 변환 (RGB -> BGR)
    cv_image = np.array(pil_image)[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    st.write("라플라시안 분산 값:", laplacian_var)
    return laplacian_var < blur_threshold

# CLIP 모델 및 디바이스 설정
clip_model, clip_processor = load_clip_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# 정책 목록 및 CLIP 임계값
policies = [
    "혐오/차별 이미지",
    "성인 이미지"
]
clip_threshold = 0.7

# 메시지 출력 (HTML 기반 메시지)
def display_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # 텍스트 부분
            st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
            
            # 이미지가 있을 경우 따로 출력 (텍스트 박스와 분리)
            if msg.get("image"):
                st.markdown(
                    f"""
                    <div>
                        <img src='data:image/png;base64,{msg['image']}' class='custom-image' style='float:right; margin-top:8px; width:300px'/>
                    </div>
                    """, unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            content_html = f"<div><div class='bot-message'>{msg['content']}</div>"
            if msg.get("image") and msg["image"].startswith("http"):
                    content_html += f"<br><img src='{msg['image']}' width='500' style='margin-top:8px;'/>"
            content_html += "</div>"
            st.markdown(content_html, unsafe_allow_html=True)
            st.audio(base64.b64decode(msg["tts"]), format="audio/mp3")

# ==================================
# 사용자 입력 처리
# ==================================
def handle_user_input(user_prompt=None, audio_file=None):
    # 이미지가 업로드된 경우 흐림 및 정책 위반 여부 확인
    if st.session_state.uploaded_image is not None:
        image = Image.open(st.session_state.uploaded_image).convert("RGB")
        # 1) 흐림(블러) 체크
        if is_blurry(image, blur_threshold=550):
            st.warning("이 이미지는 흐리게 감지되었습니다. 다시 선명한 이미지를 업로드해 주세요.")
            st.session_state.uploaded_image = None
            return
        # 2) CLIP을 이용한 정책 위반 체크
        clip_inputs = clip_processor(
            text=policies,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        with torch.no_grad():
            policy_outputs = clip_model(**clip_inputs)
        logits = policy_outputs.logits_per_image  # (batch, num_policies)
        probs = logits.softmax(dim=1)[0].tolist()
        violation_list = []
        for i, policy in enumerate(policies):
            score = probs[i]
            st.write(f"- **{policy}** 유사도 확률: {score:.4f}")
            if score > clip_threshold:
                violation_list.append(policy)
        if violation_list:
            st.error("다음 정책을 위반할 가능성이 감지되었습니다:")
            for v in violation_list:
                st.write(f"- {v}")
            st.warning("위반 의심이 있으니, 이미지를 다시 확인하거나 다른 이미지를 업로드해주세요.")
            st.session_state.uploaded_image = None
            return

    combined_prompt = ""

    if audio_file:
        transcript_result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko"
        )
        user_prompt = transcript_result.text

    if st.session_state.uploaded_image:
        image_path = os.path.join("./data/", st.session_state.uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(st.session_state.uploaded_image.getbuffer())
        combined_prompt = f"{image_path}\n\n{user_prompt}"

    if user_prompt:
        encoded_image = None
        if st.session_state.uploaded_image:
            encoded_image = base64.b64encode(st.session_state.uploaded_image.getvalue()).decode()

        st.session_state.messages.append({
            "role": "user", 
            "content": user_prompt,
            "image": encoded_image
            }
        )
        combined_prompt += user_prompt

        # 이미지가 있다면 표시
        st.markdown(f"<div class='user-message'>{user_prompt}</div>", unsafe_allow_html=True)
        if st.session_state.get("uploaded_image"):
            st.markdown(
                f"<div style='display:flex;justify-content:flex-end;'>"
                f"<img src='data:image/png;base64,{base64.b64encode(st.session_state.uploaded_image.getvalue()).decode()}' width='300'/></div>",
                unsafe_allow_html=True)        

        with st.spinner("답변을 생성 중입니다..."):
            try:
                # AI 응답 생성
                assistant_response = ask_lang_graph_agent(combined_prompt)["messages"][-1].content
                related_image = None
                match = re.search(IMAGE_PATTERN, assistant_response, flags=re.IGNORECASE)
                if match:
                    related_image = match.group(0)
                    assistant_response = re.sub(IMAGE_PATTERN, '', assistant_response, count=1, flags=re.IGNORECASE)
                
                # 스트리밍 형식으로 한 글자씩 출력
                streaming_placeholder = st.empty()
                displayed_text = ""
                for char in assistant_response:
                    displayed_text += char
                    streaming_placeholder.markdown(f"<div class='bot-message'>{displayed_text}</div>", unsafe_allow_html=True)
                    time.sleep(0.005)

                if related_image:
                    st.image(related_image, width=500)
                
                with st.spinner("TTS 생성 중입니다..."):
                    # 수정 코드:
                    if assistant_response.strip():
                        summary_text = summarize_text(assistant_response)
                        tts_audio_b64 = generate_tts(summary_text, lang="ko")
                    else:
                        tts_audio_b64 = None
                    if tts_audio_b64:
                        audio_bytes = base64.b64decode(tts_audio_b64)
                        st.audio(audio_bytes, format="audio/mp3")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "tts": tts_audio_b64,
                    "image": related_image
                })

            except Exception as e:
                st.error(f"AI 응답 생성 중 오류가 발생했습니다: {e}")

# FAQ 버튼 영역: 2열로 배치 (원하는 레이아웃에 맞게 조정 가능)
st.write("")
col1, col2 = st.columns(2)
with col1:
    if st.button("계기판에 경고등이 켜졌을 때 어떻게 해야 하나요?", key="faq1", use_container_width=True):
        st.session_state.user_prompt = "계기판에 경고등이 켜졌을 때 어떻게 해야 하나요?"
    if st.button("근처에 서비스 센터는 어디 있나요?", key="faq3", use_container_width=True):
        st.session_state.user_prompt = "근처 서비스 센터"
with col2:
    if st.button("내비게이션, 오디오, 스마트 기능 등은 어떻게 설정하나요?", key="faq2", use_container_width=True):
        st.session_state.user_prompt = "내비게이션, 오디오, 스마트 기능 등은 어떻게 설정하나요?"
    if st.button("블루투스, 스마트폰 연동 등 연결 문제가 발생하면 어떻게 해결하나요?", key="faq4", use_container_width=True):
        st.session_state.user_prompt = "블루투스, 스마트폰 연동 등 연결 문제가 발생하면 어떻게 해결하나요?"

display_messages()
# 항상 chat_input은 실행되어야 함!
user_prompt = st.chat_input("KCC에게 질문해주세요.", key="main_chat_input")

# FAQ 버튼 눌렀을 경우에도 user_input에 값을 주입
if not user_prompt:
    user_prompt = st.session_state.user_prompt
if st.session_state.audio_file:
    handle_user_input(audio_file=st.session_state.audio_file)
    st.session_state.audio_file = None
    st.session_state.user_prompt = None
elif user_prompt:
    handle_user_input(user_prompt)
    st.session_state.user_prompt = None