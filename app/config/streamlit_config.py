import streamlit as st
from PIL import Image
from PIL import Image
from config.config import IMAGE_PATH

def apply_streamlit_settings():
    st.set_page_config(
        page_title="KCC User Assistant",
        page_icon=Image.open(IMAGE_PATH + "/favicon.ico"),
        layout="wide"
    )

def apply_custom_css():
    st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
        top: 0;
        width: 100%;
        background-color: #fff;
        z-index: 9999;
        padding-top: 10px;
        margin-top: 45px;
        color: black;
    }
    .content {
        margin-top: 10px; 
    }
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
    .custom-image {
        width: 50%;
        height: auto;
        display: block;
        margin-top: 5px;
    }
    @media (max-width: 768px) {
        .custom-image {
            width: 95%;
        }
    }
    </style>
    """, unsafe_allow_html=True)
