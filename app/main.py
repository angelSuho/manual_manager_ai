from config.config import st, car_types
from config import streamlit_config
import torch

streamlit_config.apply_streamlit_settings()

st.markdown("<h2 style='text-align:center;'>🚗 Where would you like to go?</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Explore your travel opportunities with us!</p>", unsafe_allow_html=True)

torch.classes.__path__ = []
for i in range(0, len(car_types), 3):
    cols = st.columns([1, 0.2, 1, 0.2, 1])

    for idx, col in enumerate([cols[0], cols[2], cols[4]]):
        if i + idx < len(car_types):
            car = car_types[i + idx]
            with col:
                st.image(car['image'], use_container_width=False, width=220)
                if st.button(car['name'], key=car["name"], use_container_width=True):
                    st.session_state.car_type = car["name"]
                    st.session_state.clear_chat = True  # flag 설정
                    st.switch_page("pages/chat.py")

