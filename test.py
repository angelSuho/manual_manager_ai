import streamlit as st

st.set_page_config(page_title="ChatGPT 스타일 채팅", layout="wide")

# CSS 스타일링 (ChatGPT 스타일 유사)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #202123;
    color: white;
    padding-top: 10px;
}

.sidebar-room {
    padding: 10px 15px;
    border-radius: 5px;
    margin-bottom: 5px;
    cursor: pointer;
    color: #fff;
    transition: background-color 0.3s ease;
}

.sidebar-room:hover {
    background-color: #333541;
}

.sidebar-room.selected {
    background-color: #40414f;
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
    display: inline-block;
}
            
[data-testid="stMarkdownContainer"]:has(.user-message) {
    display: flex !important;
    justify-content: flex-end !important;
}

/* bot-message는 기본 왼쪽 정렬 */
[data-testid="stMarkdownContainer"]:has(.bot-message) {
    display: flex !important;
    justify-content: flex-start !important;
}

.bot-message {
    text-align: left;
    padding: 10px;
    background-color: #f1f1f1;
    color: black;
    border-radius: 15px 15px 15px 0px;
    margin-bottom: 10px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
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

# 사이드바 (채팅방 목록)
with st.sidebar:
    st.markdown("## 채팅방 목록")

    for room in st.session_state.chat_rooms.keys():
        if room == st.session_state.current_room:
            room_class = "sidebar-room selected"
        else:
            room_class = "sidebar-room"

        if st.markdown(f"<div class='{room_class}' onclick=''>{room}</div>", unsafe_allow_html=True):
            st.session_state.current_room = room

# 메인 (선택된 채팅방 메시지 표시)
st.markdown(f"### {st.session_state.current_room}")
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.chat_rooms[st.session_state.current_room]:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# 입력창
user_input = st.chat_input("메시지를 입력하세요...")
if user_input:
    st.session_state.chat_rooms[st.session_state.current_room].append({"role": "user", "content": user_input})
    st.session_state.chat_rooms[st.session_state.current_room].append({"role": "assistant", "content": "답변 준비중입니다..."})
    st.experimental_rerun()
