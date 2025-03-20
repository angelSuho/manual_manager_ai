import fitz
import os
import json
from dotenv import load_dotenv

load_dotenv()

pdf_directory = './pdfs'
img_dir = 'extracted_images'
os.makedirs(img_dir, exist_ok=True)

def get_title_level(font_size):
    if font_size >= 30:
        return "grand_title"
    elif font_size >= 18:
        return "sub_title"
    else:
        return "content"

def analyze_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    document_structure = []
    current_grand_title = None
    current_sub_title = None

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if 'lines' in block:
                # 블록 내 모든 스팬의 텍스트를 합치고, 그중 가장 큰 폰트 크기를 찾습니다.
                block_text_list = []
                max_font_size = 0

                for line in block['lines']:
                    for span in line['spans']:
                        # 텍스트를 합칩니다.
                        span_text = span['text'].strip()
                        block_text_list.append(span_text)
                        
                        # 가장 큰 폰트 크기를 추적합니다.
                        if span['size'] > max_font_size:
                            max_font_size = span['size']

                # 하나의 문자열로 합친 뒤 앞뒤 공백 제거
                full_block_text = ' '.join(block_text_list).strip()
                if not full_block_text:
                    continue  # 빈 문자열이면 스킵

                # 최대 폰트 크기를 기준으로 제목/소제목/본문 결정
                title_level = get_title_level(max_font_size)

                if title_level == "grand_title":
                    current_grand_title = {"title": full_block_text, "sub_titles": []}
                    document_structure.append(current_grand_title)
                    current_sub_title = None

                elif title_level == "sub_title":
                    current_sub_title = {"title": full_block_text, "contents": [], "images": []}
                    if current_grand_title:
                        current_grand_title["sub_titles"].append(current_sub_title)

                else:  # content
                    if current_sub_title:
                        current_sub_title["contents"].append(full_block_text)

                # 이미지 추출 (중복 제거)
                seen_xrefs = set()
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    if xref in seen_xrefs:
                        # 이미 처리한 이미지라면 넘어감
                        continue
                    seen_xrefs.add(xref)

                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_ext = base_image["ext"]
                    img_name = f"{os.path.basename(pdf_path)}_page{page.number}_{img_index}.{img_ext}"

                    img_filepath = os.path.join(img_dir, img_name)
                    with open(img_filepath, 'wb') as f:
                        f.write(img_bytes)

                    if current_sub_title:
                        current_sub_title["images"].append(img_filepath)

    return document_structure


final_result = []
for pdf_filename in os.listdir(pdf_directory):
    if pdf_filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, pdf_filename)
        pdf_structure = analyze_pdf(pdf_path)
        final_result.append({"pdf_file": pdf_filename, "structure": pdf_structure})

with open('structured_output.json', 'w', encoding='utf-8') as json_file:
    json.dump(final_result, json_file, ensure_ascii=False, indent=4)

print("모든 PDF 파일이 분석되어 JSON으로 저장되었습니다.")
