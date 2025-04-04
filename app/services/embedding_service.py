# Description: Pinecone에 데이터를 인덱싱
import os
from langchain.docstore.document import Document
from pinecone import Pinecone
from dotenv import load_dotenv
import hashlib  # 파일 해시 계산을 위한 모듈
import json  # JSON 파일 처리를 위한 json 모듈 임포트

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config.config import st

load_dotenv()

jsons = [
    "./data/jsons/mercedes-e-class-sedan-manual.json",
    "./data/jsons/mercedes-eqs-sedan-manual.json",
    "./data/jsons/mercedes-gla-suv-manual.json",
    "./data/jsons/mercedes-s-class-sedan-manual.json",
]

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# ==========================================
# Pinecone 인덱스 초기화 및 데이터 인덱싱 함수
# ==========================================
def get_file_hash(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()
def load_stored_hash(hash_file):
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
def save_hash(hash_file, hash_value):
    with open(hash_file, "w", encoding="utf-8") as f:
        f.write(hash_value)
def index_data() -> str:
    """
    usage.json 파일의 내용을 읽어와 각 섹션의 하위 항목(서브타이틀) 단위로 문서를 생성하고,
    각 문서에는 아래의 정보가 포함됩니다.
      - 전체 섹션 제목
      - 해당 서브타이틀 제목
      - 서브타이틀의 전체 내용 (텍스트로 결합)
      - PDF 파일 이름
      - 서브타이틀에서 추출한 이미지 URL 목록 (contents와 images 모두 확인)
    이전에 저장된 해시와 비교하여 변경이 있을 때만 인덱싱을 진행합니다.
    """
    car_type = st.session_state.get("car_type", "EQS")
    index_name = car_type.lower()
    st.session_state.car_type = index_name

    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
        )

    embedding = OpenAIEmbeddings(model="text-embedding-3-large") # text-embedding-ada-002, text-embedding-3-large
    database = PineconeVectorStore(index_name=index_name, embedding=embedding)

    target_json = None
    for json_file in jsons:
        if index_name in json_file:
            target_json = json_file
            break

    hash_file = f"{target_json}.hash"
    current_hash = get_file_hash(target_json)
    stored_hash = load_stored_hash(hash_file)
    
    if stored_hash != current_hash:
        json_file = target_json
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
            documents = []
            # usage.json은 리스트 형태의 항목들을 포함
            for item in data:
                structure = item.get("structure", [])
                # 각 섹션에 대해
                for section in structure:
                    section_title = section.get("title", "")
                    sub_titles = section.get("sub_titles", [])
                    content = section.get("content", [])
                    images = section.get("images", [])

                    # Case 1: sub_titles가 있는 경우 (기존 처리)
                    if sub_titles:
                        for sub in sub_titles:
                            sub_title = sub.get("title", "")
                            contents = sub.get("contents", [])
                            sub_images = sub.get("images", [])

                            # 이미지 추출
                            image_paths = [c for c in contents if isinstance(c, str) and c.lower().endswith(('.jpeg', '.jpg', '.png', '.gif'))]
                            if isinstance(sub_images, list):
                                image_paths.extend(sub_images)
                            elif isinstance(sub_images, str):
                                image_paths.append(sub_images)

                            # 문서 생성
                            content_text = f"{section_title}\n{sub_title}\n" + "\n".join(contents)
                            metadata = {
                                "section_title": section_title,
                                "sub_title": sub_title,
                                "image_paths": json.dumps(image_paths)
                            }
                            documents.append(Document(page_content=content_text, metadata=metadata))

                    # Case 2: sub_titles가 없고 content가 바로 있는 경우
                    elif content:
                        # 이미지 추출
                        image_paths = [c for c in content if isinstance(c, str) and c.lower().endswith(('.jpeg', '.jpg', '.png', '.gif'))]
                        if isinstance(images, list):
                            image_paths.extend(images)
                        elif isinstance(images, str):
                            image_paths.append(images)

                        content_text = f"{section_title}\n" + "\n".join(content)
                        metadata = {
                            "section_title": section_title,
                            "sub_title": "",  # 서브타이틀 없음
                            "image_paths": json.dumps(image_paths)
                        }
                        documents.append(Document(page_content=content_text, metadata=metadata))
            # 생성된 문서를 데이터베이스(Pinecone)에 추가
            database.add_documents(documents)
        # 새 해시값 저장 (이후 변경 여부 판단)
        save_hash(hash_file, current_hash)
    return index_name

index_data()
