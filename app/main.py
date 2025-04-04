from config.config import st, car_types
from config import streamlit_config
import torch

torch.classes.__path__ = []
# 페이지 전체 레이아웃(가로폭) 설정
streamlit_config.apply_streamlit_settings()

# ===== CSS 스타일 커스터마이징 =====
st.markdown("""
<style>
/* 전체 폰트와 기본 색 설정 */
body {
    margin: 0;
    padding: 0;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #444;
    background-color: #f5f7fa;
}

/* ===== 배경 웨이브 영역 (헤더) ===== */
.wave-container {
    position: relative;
    width: 100%;
    height: 150px;
    background: linear-gradient(135deg, #8AA6E2, #A58BC8);
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;  /* 세로 중앙정렬 */
    flex-direction: column;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-radius: 16px;
}

.wave-container h1 {
    color: #fff;
    font-size: 36px;
    font-weight: 700;
    margin: 0;
}

.wave-container p {
    color: #fffce6;
    font-size: 16px;
    margin: 0;
}

.wave {
    position: absolute;
    bottom: 0;
    width: 100%;
    height: 80px;
    background: url('https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Wave.svg/1920px-Wave.svg.png');
    background-size: cover;
    background-repeat: no-repeat;
    transform: translateY(40px);
    opacity: 0.8;
}

/* ===== 서비스 소개 섹션 ===== */
.services-section {
    margin: 40px auto;
    max-width: 1200px;
    text-align: center;  /* 섹션 제목만 중앙 정렬 */
}
.services-section h2 {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 25px;
    color: #444;
}

/* 4개 아이템을 가로로 배치 */
.services-wrapper {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 30px;
    justify-items: center;
    align-items: start;
}

/* ===== 메모지 느낌의 박스 스타일 ===== */
.service-box {
    background: #fffde7; 
    border: 1px solid #f0e8b0;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.08);
    border-radius: 6px;
    padding: 20px;
    text-align: left;
    position: relative;
}

/* 상단 중앙에 핀을 꽂은 것 같은 장식 */
.service-box::before {
    content: "";
    position: absolute;
    top: 8px;
    left: 50%;
    width: 13px;
    height: 12px;
    background: #808080;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 0 0 2px #fffde7;
}

/* 아이콘 크기와 색상 */
.service-icon {
    font-size: 28px;  
    margin-bottom: 8px;
    color: #777;      
}

/* 제목 */
.service-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 6px;
    color: #444;
}

/* 설명 */
.service-desc {
    font-size: 14px;
    color: #777;
    line-height: 1.5;
}

/* ===== 차량 선택 안내 ===== */
.choose-flex {
    position: relative;
    display: block;
    background: linear-gradient(90deg, #8DA6D2, #B2A5E0);
    color: #fff;
    font-size: 20px;
    font-weight: 600;
    padding: 15px 40px;
    border-radius: 30px;
    margin: 30px auto 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    width: fit-content;
    text-align: center;
}
            
/* ===== 차량 카드 컨테이너 ===== */
.cards-container {
    display: flex;
    gap: 30px;
    padding: 20px;
    align-items: center;
    justify-content: center;
}

/* 카드 그리드 배치 (3개씩 한 줄) */
.card-row {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-bottom: 40px;
}
.card-link {
    text-decoration: none;
    color: inherit;
    margin: 5px 10px 5px;
}

/* 차량 카드 */
.card {
    width: 280px;
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    padding: 20px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.card:hover {
    transform: translateY(-10px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
}
.card img {
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 12px;
    margin-bottom: 15px;
}
.card-title {
    font-size: 18px;
    font-weight: bold;
    margin-top: 10px;
    color: #444;
}

/* ===== 푸터 ===== */
.footer {
    margin-top: 40px;
    padding: 20px;
    font-size: 14px;
    color: #777;
    text-align: center;
    border-top: 1px solid #ddd;
    position: relative;
}

/* 푸터의 하트 애니메이션 (예시) */
.footer-heart {
    color: #e25555;
    animation: heartbeat 1.5s infinite;
}

@keyframes heartbeat {
    0% { transform: scale(1); }
    25% { transform: scale(1.2); }
    40% { transform: scale(1); }
    60% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# ===== 상단 웨이브 헤더 =====
st.markdown("""
<div class="wave-container">
    <h1>🚘 User Assistant</h1>
    <p>차량 매뉴얼 검색 · 경고등 인식 · 서비스 센터 탐색 · AI 상담</p>
    <div class="wave"></div>
</div>
""", unsafe_allow_html=True)

# ===== 서비스 소개 섹션 (메모지 스타일) =====
st.markdown("""
<div class="services-section">
    <h2>제공하는 서비스</h2>
    <div class="services-wrapper">
        <div class="service-box">
            <div class="service-icon">🔍</div>
            <div class="service-title">매뉴얼 검색</div>
            <div class="service-desc">원하는 정보를 빠르게 찾아내고, 정리된 요약을 확인하세요.</div>
        </div>
        <div class="service-box">
            <div class="service-icon">🧠</div>
            <div class="service-title">이미지 경고등 분석</div>
            <div class="service-desc">사진 한 장이면 어떤 경고등인지 인식하고, 대처법을 안내합니다.</div>
        </div>
        <div class="service-box">
            <div class="service-icon">📍</div>
            <div class="service-title">서비스 센터 찾기</div>
            <div class="service-desc">현재 위치 기반으로 가장 가까운 센터 정보를 제공합니다.</div>
        </div>
        <div class="service-box">
            <div class="service-icon">💬</div>
            <div class="service-title">AI 상담</div>
            <div class="service-desc">간단한 질문부터 전문 상담까지, AI가 여러분을 지원합니다.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== 차량 선택 안내 =====
st.markdown("""
<div class="choose-flex">
    🚗 보유하신 차량을 선택해주세요 🚗
</div>
""", unsafe_allow_html=True)

# ===== 차량 카드 UI (카드 전체 클릭 시 이동) =====
st.markdown('<div class="cards-container">', unsafe_allow_html=True)

cards_per_row = 3
for i in range(0, len(car_types), cards_per_row):
    cols = st.columns(cards_per_row)
    for j in range(cards_per_row):
        if i + j < len(car_types):
            car = car_types[i + j]
            with cols[j]:
                st.markdown(f"""
                <a class="card-link" href="/chat?car={car['name']}" target="_self">
                    <div class="card">
                        <img src="{car['image']}" alt="{car['name']}">
                        <div class="card-title">{car['name']}</div>
                    </div>
                </a>
                """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ===== 푸터 =====
st.markdown("""
<div class="footer">
    Mercedes-Benz • KCC 오토 공식 파트너 • Designed with <span class="footer-heart">❤️</span> by Team 채원이와 아이
</div>
""", unsafe_allow_html=True)
