import streamlit as st
from config import streamlit_config

# Streamlit 설정 및 CSS 적용
streamlit_config.apply_streamlit_settings()
streamlit_config.apply_custom_css()

car_types = [
    {"name": "EQS", "image": "./data/images/eqs.png"},
    {"name": "EQE", "image": "./data/images/eqe.png"},
    {"name": "C-Class", "image": "./data/images/c-class.png"},
    {"name": "E-Class", "image": "./data/images/e-class.png"},
    {"name": "A-Class", "image": "./data/images/a-class.png"},
    {"name": "S-Class", "image": "./data/images/s-class.png"},
    {"name": "AMG GT", "image": "./data/images/amg-gt.png"},
    {"name": "GLA", "image": "./data/images/gla.png"},
    {"name": "GLC", "image": "./data/images/glc.png"},
]

# 헤더 섹션
st.markdown("""
    <div class="fixed-header">
        <h3 style="margin:0; padding-bottom: 7px;">KCC Auto Manager</h3>
    </div>
    """, unsafe_allow_html=True)

# 소개 섹션
st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1>환영합니다!</h1>
        <p>KCC Auto Manager 챗봇은 차량 매뉴얼에 대한 정보를 제공합니다.</p>
        <p>아래에서 차량 모델을 선택하여 채팅을 시작하세요.</p>
    </div>
    """, unsafe_allow_html=True)

# 차량 선택 섹션
car_types = [
    "EQS 450+", "EQE 350+", "CLA 250", "C 300", "E 350",
    "S 580", "AMG GT 4-Door", "GLA 250", "GLC 300",
    "GLE 450", "G 63 AMG", "300 SL Gullwing"
]

selected_car = st.selectbox("차량 모델을 선택하세요:", car_types)

if st.button("채팅 시작"):
    st.session_state.car_type = selected_car
    st.experimental_set_query_params(car_type=selected_car)
    st.experimental_rerun()
