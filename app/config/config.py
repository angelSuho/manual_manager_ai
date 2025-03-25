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
    {"name": "EQS", "image": "./data/images/eqs.png"},
    {"name": "S-Class", "image": "./data/images/s-class.png"},
    {"name": "E-Class", "image": "./data/images/e-class.png"},
    {"name": "GLA", "image": "./data/images/gla.png"},
    {"name": "GLC", "image": "./data/images/glc.png"},
    {"name": "EQE", "image": "./data/images/eqe.png"},
    {"name": "C-Class", "image": "./data/images/c-class.png"},
    {"name": "A-Class", "image": "./data/images/a-class.png"},
    {"name": "AMG GT", "image": "./data/images/amg-gt.png"},
]

# 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")