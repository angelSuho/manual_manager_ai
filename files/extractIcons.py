import os
import fitz
import json
import re
import torch
import clip
from PIL import Image
import numpy as np

ICON_OUTPUT_DIR = "./icons"
SIMILARITY_THRESHOLD = 0.98

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def is_icon_candidate(char: str, font_name: str, font_size: float) -> bool:
    if any(keyword in font_name.lower() for keyword in ["symb", "icon", "dingbat", "glyph"]):
        return True
    cp = ord(char)
    if 0xE000 <= cp <= 0xF8FF:
        return True
    if char.isalnum() or char.isspace() or char in ".,!?()[]{}:;\"'–-_=+…":
        return False
    if cp < 128:
        return False
    if font_size > 10:
        return True
    return False

def get_clip_embedding(pix) -> torch.Tensor:
    image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    image_pil = Image.fromarray(image_array[..., :3])
    image_input = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding

def is_duplicate(embedding, embeddings_dict) -> bool:
    for stored_emb in embeddings_dict.values():
        similarity = (embedding @ stored_emb.T).item()
        if similarity >= SIMILARITY_THRESHOLD:
            return True
    return False

def extract_icons(pdf_path: str, icon_mapping: dict, embeddings_dict: dict) -> None:
    doc = fitz.open(pdf_path)
    zoom = 3.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index, page in enumerate(doc):
        text_dict = page.get_text("rawdict")

        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_name = span["font"]
                        font_size = span["size"]

                        for char in span["chars"]:
                            c = char["c"]
                            bbox = fitz.Rect(char["bbox"])

                            if is_icon_candidate(c, font_name, font_size):
                                clip_rect = bbox
                                pix = page.get_pixmap(matrix=matrix, clip=clip_rect)

                                embedding = get_clip_embedding(pix)
                                if not is_duplicate(embedding, embeddings_dict):
                                    safe_char = sanitize_filename(c)
                                    safe_font = sanitize_filename(font_name)
                                    filename = f"icon_{safe_char}_{safe_font}_{page_index}_{int(bbox.x0)}_{int(bbox.y0)}.png"
                                    out_path = os.path.join(ICON_OUTPUT_DIR, filename)
                                    pix.save(out_path)

                                    key = f"{c}||{font_name}||{page_index}||{int(bbox.x0)}-{int(bbox.y0)}"
                                    icon_mapping[key] = {
                                        "glyph": c,
                                        "font": font_name,
                                        "image": filename,
                                    }
                                    embeddings_dict[key] = embedding

def main():
    pdf_files = [
        "./pdfs/mercedes-eqs-sedan-manual_1.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_2.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_3.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_4.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_5.pdf",
    ]

    os.makedirs(ICON_OUTPUT_DIR, exist_ok=True)
    icon_mapping = {}
    embeddings_dict = {}

    for pdf in pdf_files:
        extract_icons(pdf, icon_mapping, embeddings_dict)
        print(f"{pdf} 처리 완료.")

    with open("icon_mapping.json", "w", encoding="utf-8") as f:
        json.dump(icon_mapping, f, ensure_ascii=False, indent=4)

    print("아이콘 매핑이 완료되었습니다.")
    print("icon_mapping.json 및 ./icons 폴더를 확인하세요.")

if __name__ == "__main__":
    main()
