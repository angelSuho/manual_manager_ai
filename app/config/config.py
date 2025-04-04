from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import openai
import os

load_dotenv()

INPUT_PATH = f"./data/uploads"
IMAGE_PATH = f"./data/images"
IMAGE_PATTERN = r'((?:[A-Za-z]:[\\/]|https?://).+?\.(?:png|jpg|jpeg|gif))'

car_types = [
    {"name": "EQS", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/eqs.png"},
    {"name": "S-Class", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/s-class.png"},
    {"name": "E-Class", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/e-class.png"},
    {"name": "GLA", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/gla.png"},
    {"name": "GLC", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/glc.png"},
    {"name": "EQE", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/eqe.png"},
    {"name": "C-Class", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/c-class.png"},
    {"name": "A-Class", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/a-class.png"},
    {"name": "AMG GT", "image": "https://kcc-llm.s3.ap-northeast-2.amazonaws.com/amg-gt.png"},
]

# 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")