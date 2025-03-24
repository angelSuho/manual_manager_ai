from dotenv import load_dotenv
from openai import OpenAI
import openai
import os

load_dotenv()

INPUT_PATH = f"./data/uploads"
IMAGE_PATH = f"./data/images"
CAR_TYPE = "EQS 450+"
IMAGE_PATTERN = r'((?:[A-Za-z]:[\\/]|https?://).+?\.(?:png|jpg|jpeg|gif))'

# 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")