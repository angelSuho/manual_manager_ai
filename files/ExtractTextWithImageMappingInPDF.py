import fitz
import os
import json
import re

pdf_directory = './files/pdfs/e_class'
img_dir = '../extracted_images'
exclude_texts = [
    "경고등 및 표시등 가능한 원인/결과 및 M 해결 방법",
    "디스플레이 메시지 가능한 원인/결과 및 M 해결 방법"
]

os.makedirs(img_dir, exist_ok=True)

def get_title_level(font_size):
    if font_size >= 30:
        return "grand_title"
    elif font_size >= 18:
        return "sub_title"
    else:
        return "content"

def analyze_pdf_normal(pdf_path, img_dir="../extracted_images"):
    """
    기존 로직: 폰트 크기에 따라 grand_title/sub_title/content 분류
    """
    doc = fitz.open(pdf_path)
    document_structure = []
    current_grand_title = None
    last_sub_title = None

    undesired_pattern = re.compile(r'^[A-Z]\d{3}\s\d{4}\s\d{2}$')
    skip_substrings = ["사용 설명서"]

    for page_index, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if 'lines' not in block:
                continue

            block_text_list = []
            max_font_size = 0
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if text:
                        block_text_list.append(text)
                    if span['size'] > max_font_size:
                        max_font_size = span['size']

            full_block_text = ' '.join(block_text_list).strip()
            if not full_block_text:
                continue

            # 페이지 번호(예: 현재 페이지번호 또는 1-indexed 번호) 제외 처리
            if full_block_text.isdigit():
                continue

            if undesired_pattern.match(full_block_text):
                continue
            if any(sub in full_block_text for sub in skip_substrings):
                continue
            for text_to_remove in exclude_texts:
                if text_to_remove in full_block_text:
                    full_block_text = full_block_text.replace(text_to_remove, "")
            full_block_text = full_block_text.strip()
            if not full_block_text:
                continue

            title_level = get_title_level(max_font_size)

            if title_level == "grand_title":
                current_grand_title = {
                    "title": full_block_text,
                    "sub_titles": []
                }
                document_structure.append(current_grand_title)
                last_sub_title = None
            elif title_level == "sub_title":
                if current_grand_title is None:
                    current_grand_title = {"title": "Untitled Grand Title", "sub_titles": []}
                    document_structure.append(current_grand_title)
                sub_title_dict = {"title": full_block_text, "contents": []}
                current_grand_title["sub_titles"].append(sub_title_dict)
                last_sub_title = sub_title_dict
            else:
                if last_sub_title:
                    last_sub_title["contents"].append(full_block_text)
                else:
                    if current_grand_title is None:
                        current_grand_title = {"title": "Untitled Grand Title", "sub_titles": []}
                        document_structure.append(current_grand_title)
                    sub_title_dict = {"title": "본문", "contents": [full_block_text]}
                    current_grand_title["sub_titles"].append(sub_title_dict)
                    last_sub_title = sub_title_dict

        # 이미지 추출
        seen_xrefs = set()
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_name = f"{os.path.basename(pdf_path)}_page{page_index}_{img_index}.{img_ext}"

            img_filepath = os.path.abspath(os.path.join(img_dir, img_name)).replace("\\", "/")
            with open(img_filepath, 'wb') as f:
                f.write(img_bytes)

            if last_sub_title and (img_filepath not in last_sub_title["contents"]):
                last_sub_title["contents"].append(img_filepath)

    return document_structure

def analyze_pdf_special(pdf_path, img_dir="extracted_images"):
    """
    특수 로직: 기존 대제목/소제목 분류 없이 간단한 리스트 형태로 저장.
    """
    doc = fitz.open(pdf_path)
    document_structure = []
    special_entries = []

    undesired_pattern = re.compile(r'^[A-Z]\d{3}\s\d{4}\s\d{2}$')
    skip_substrings = ["사용 설명서"]

    for page_index, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if 'lines' not in block:
                continue
            block_text_list = []
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if text:
                        block_text_list.append(text)
            full_block_text = ' '.join(block_text_list).strip()
            if not full_block_text:
                continue

            # 페이지 번호 제거
            if full_block_text.isdigit():
                continue
            if undesired_pattern.match(full_block_text):
                continue
            if any(sub in full_block_text for sub in skip_substrings):
                continue
            for text_to_remove in exclude_texts:
                if text_to_remove in full_block_text:
                    full_block_text = full_block_text.replace(text_to_remove, "")
            full_block_text = full_block_text.strip()
            if not full_block_text:
                continue

            special_entries.append(full_block_text)

        # 이미지 추출 (선택적)
        seen_xrefs = set()
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_name = f"{os.path.basename(pdf_path)}_page{page_index}_{img_index}.{img_ext}"

            img_filepath = os.path.abspath(os.path.join(img_dir, img_name)).replace("\\", "/")
            with open(img_filepath, 'wb') as f:
                f.write(img_bytes)

            special_entries.append(f"[IMAGE] {img_filepath}")

    document_structure.append({
        "pdf_mode": "special_for_5pdf",
        "contents": special_entries
    })
    return document_structure

def main():
    final_result = []

    for pdf_filename in os.listdir(pdf_directory):
        if not pdf_filename.lower().endswith('.pdf'):
            continue

        pdf_path = os.path.join(pdf_directory, pdf_filename)

        # # '5.pdf'인 경우 특수 로직 적용
        # routing_pdfs = [
        #     'mercedes-eqs-sedan-manual_5.pdf'
        #     ]
        # if pdf_filename in routing_pdfs:
        #     pdf_structure = analyze_pdf_special(pdf_path)
        # else:
        pdf_structure = analyze_pdf_normal(pdf_path)

        final_result.append({"pdf_file": pdf_filename, "structure": pdf_structure})

    result_pdf_name = f'{pdf_filename.replace(".pdf", "")}.json'
    print(f"JSON 파일로 저장 중: {result_pdf_name}")
    with open(result_pdf_name, 'w', encoding='utf-8') as json_file:
        json.dump(final_result, json_file, ensure_ascii=False, indent=4)

    print("모든 PDF 파일이 분석되어 JSON으로 저장되었습니다.")

if __name__ == "__main__":
    main()
