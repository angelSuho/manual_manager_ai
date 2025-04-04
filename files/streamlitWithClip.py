import os
import cv2
import torch
import clip
from PIL import Image
import streamlit as st
import numpy as np
import pickle

save_path = "./files/"

def imread_unicode(file_path):
    stream = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img

def resize_image(image, max_dim=800):
    height, width = image.shape[:2]
    if max(height, width) > max_dim:
        scaling_factor = max_dim / float(max(height, width))
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

# CLIP 모델과 전처리 로딩 (예: RN50 사용하면 더 빠를 수 있음)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

st.title("Image Similarity Finder")
st.write("타겟 이미지와 아이콘 이미지 간의 유사도를 계산합니다.")

# 타겟 이미지 업로드 (없으면 기본 이미지 사용)
uploaded_target = st.file_uploader("타겟 이미지를 업로드하세요 (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
if uploaded_target is not None:
    target_image = Image.open(uploaded_target).convert("RGB")
else:
    st.warning("타겟 이미지가 업로드되지 않았습니다. 기본 타겟 이미지를 사용합니다.")
    target_image_path = "page165_icon4.jpeg"  # 기본 타겟 이미지 경로
    target_image = Image.open(target_image_path).convert("RGB")

# 썸네일 표시
st.image(target_image, caption="타겟 이미지", width=300)


# 아이콘 이미지 폴더와 임베딩 저장 파일 경로
icon_folder = "./icons/계기판_디스플레이"
embeddings_path = save_path + "icon_embeddings.pkl"

def compute_and_save_embeddings():
    embeddings = {}
    for filename in os.listdir(icon_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(icon_folder, filename)
        img_cv = imread_unicode(image_path)
        if img_cv is None:
            continue
        img_cv = resize_image(img_cv, max_dim=500)
        # 이미지 전처리
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(input_tensor)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embeddings[filename] = embedding.cpu()  # CPU로 이동하여 저장
    # 저장 (pickle 사용)
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    print("임베딩 저장 완료:", embeddings_path)
    
def load_embeddings():
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings
    else:
        return None

# 만약 저장된 임베딩이 없다면 계산하고 저장
icon_embeddings = load_embeddings()
if icon_embeddings is None:
    compute_and_save_embeddings()
    icon_embeddings = load_embeddings()

# 타겟 이미지 처리
target_input = preprocess(target_image).unsqueeze(0).to(device)
with torch.no_grad():
    target_embedding = model.encode_image(target_input)
target_embedding /= target_embedding.norm(dim=-1, keepdim=True)

# 저장된 임베딩과 타겟 이미지 간의 유사도 계산
results = []
for filename, embedding in icon_embeddings.items():
    # 각 임베딩은 [1, D] 형태이므로 내적을 통해 코사인 유사도를 계산
    similarity = (target_embedding.cpu() @ embedding.T).item()
    results.append((filename, similarity))

# 유사도 내림차순 정렬
results.sort(key=lambda x: x[1], reverse=True)
print("최대 유사도 이미지:", results[0])
st.image(os.path.join(icon_folder, results[0][0]), caption="최대 유사도 이미지", width=300)
file_path = results[0][0]
file_name = os.path.basename(file_path)
st.write(file_path)
name_without_ext, _ = os.path.splitext(file_name)
st.write(name_without_ext)
