import streamlit as st
import openai
from PIL import Image
import os
import base64
from dotenv import load_dotenv

# 환경 변수 로드 (API 키를 .env 파일에서 가져옴)
load_dotenv()

st.title("GPT‑4o 이미지 질의응답 예제")
openai.api_key = os.getenv("OPENAI_API_KEY")
# 사용자가 이미지를 업로드
uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
user_question = st.text_input("이미지에 대해 질문을 입력하세요:")

if st.button("질문하기"):
    if uploaded_image is None:
        st.error("먼저 이미지를 업로드하세요.")
    elif not user_question:
        st.error("질문을 입력하세요.")
    else:
        # 업로드된 이미지 표시 (이미지 객체를 생성하면 파일 포인터가 이동됨)
        image = Image.open(uploaded_image)
        
        # 파일 포인터를 처음으로 돌려놓음
        uploaded_image.seek(0)
        
        # 이미지 데이터를 바이트로 읽고 base64로 인코딩
        image_bytes = uploaded_image.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded_image}"
        
        # GPT-4o API 호출 (예시 - 실제 사용법은 공식 문서 참고)
        try:
            mapping_prompt = """
The following is a separate '/' separator that provides two pieces of information, which are derived from the image-based answers and descriptions of general vehicle-related features and Mercedes-Benz vehicle features.

The left part of the '/' indicates the general vehicle-related functions,
The right section describes the vehicle-related features of the Mercedes.

Important: Each item in the data list provided should be mapped to the same internal function description, regardless of the fact that it means the same content, even if the exact string does not match if similar representations exist.
This rule applies to all items in the data list.

This effectively combines and analyzes the data extracted from the image-based answers with the vehicle feature mapping information provided in text to produce more accurate and consistent answers.

Below is a partial list of available data:
차량 충돌 방지 경고등 / 액티브 브레이크 어시스트
차선 이탈 경고등 / 액티브 차선 이탈 방지 어시스트
차선 변경 경고등 / 액티브 차선 변경 어시스트
사각지대 경고등 / 사각지대 어시스트 및 액티브 사각지대 어시스트
후방 교차 충돌 경고등 / 후방 교차 충돌 경고 시스템
어댑티브 크루즈 컨트롤 / 액티브 디스턴스 어시스트
자동 주차 보조 / 주차 어시스트 시스템
경사로 주행 보조 / 경사로 주행 보조 시스템
어댑티브 스티어링 / 액티브 스티어링 어시스트
자동 긴급 제동 / 액티브 비상 제동 어시스트
교통 표지판 인식 / 교통 표지판 어시스트
에어 서스펜션 / 에어매틱 / AMG 라이드 컨트롤+
회생제동 시스템 / 에너지 회수 시스템
운전자 모니터링 경고 / 운전자 모니터링 시스템
전방 충돌 경고 / 충돌 경고 시스템
후방 카메라 시스템 / 후방 주차 카메라 시스템

**Final answer format:**:
In the first line of the final answer, the core functional name of the Mercedes
Write a detailed descripFtion of the feature from the second line
The entire answer should be written in Korean, but the exceptions that begin with 'NO:' will remain the same
    """

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": 
                                f"You are a highly experienced professional specializing in Mercedes Benz eqs car manuals. "
                                    "Your role is to write your answers coherently so that users can easily understand the features. "
                                    "When a user provides an image along with a question, you must analyze the image. "
                                    "If a non-vehicle-related, blurred, or inappropriate image is entered, please respond with a message that starts with 'NO:' "
                                    "and explain why the image is unacceptable. Otherwise, answer normally. "
                                    f"{mapping_prompt}"
                                    f"\n\nUser question: {user_question}\n\n"
                                    "And please translate the entire answer into Korean, except for responses starting with 'NO:'."
                                    "Please ensure that each section is clearly separated by new lines and that the final answer is presented in an organized and professional manner in Korean."
                             },
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ]
            )
            answer = response.choices[0].message.content
            st.write("답변:", answer)
        except Exception as e:
            st.error(f"API 호출 중 오류 발생: {e}")