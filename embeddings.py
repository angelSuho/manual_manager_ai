# Description: Pinecone에 데이터를 인덱싱
import os
from langchain.docstore.document import Document
from pinecone import Pinecone
from dotenv import load_dotenv
import hashlib  # 파일 해시 계산을 위한 모듈
import json  # JSON 파일 처리를 위한 json 모듈 임포트

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

file_path = "./files/"
index_name = "kcc"

embedding = OpenAIEmbeddings(model="text-embedding-3-large") #text-embedding-ada-002, text-embedding-3-large
database = PineconeVectorStore(index_name=index_name, embedding=embedding)

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
def index_data():
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
    hash_file = file_path + "test.json.hash"
    current_hash = get_file_hash(file_path + "test.json")
    stored_hash = load_stored_hash(hash_file)
    
    if stored_hash != current_hash:
        json_file = file_path + "test.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
            documents = []
            # usage.json은 리스트 형태의 항목들을 포함
            for item in data:
                pdf_file = item.get("pdf_file", "")
                structure = item.get("structure", [])
                # 각 섹션에 대해
                for section in structure:
                    section_title = section.get("title", "")
                    sub_titles = section.get("sub_titles", [])
                    # 각 서브타이틀 단위로 Document 생성
                    for sub in sub_titles:
                        sub_title = sub.get("title", "")
                        contents = sub.get("contents", [])
                        # 서브타이틀의 내용 전체를 하나의 텍스트로 결합
                        content_text = f"{section_title}\n{sub_title}\n" + "\n".join(contents)
                        
                        # 이미지 URL 추출: contents에서 이미지 파일 경로와 sub의 images 항목 모두 확인
                        image_paths = []
                        # contents 내의 이미지 URL (파일 확장자로 판별)
                        for content in contents:
                            if content.lower().endswith(('.jpeg', '.jpg', '.png', '.gif')):
                                image_paths.append(content)
                        # sub 항목 내에 images 키가 있으면 추가 (문자열 또는 리스트 처리)
                        images = sub.get("images", [])
                        if images:
                            if isinstance(images, list):
                                image_paths.extend(images)
                            elif isinstance(images, str):
                                image_paths.append(images)
                        
                        # 메타데이터에 PDF 파일명, 섹션 제목, 서브타이틀, 이미지 목록을 저장
                        metadata = {
                            "pdf_file": pdf_file,
                            "section_title": section_title,
                            "sub_title": sub_title,
                            "image_paths": json.dumps(image_paths)
                        }
                        # Document 객체 생성 후 리스트에 추가
                        documents.append(Document(page_content=content_text, metadata=metadata))
            # 생성된 문서를 데이터베이스(Pinecone)에 추가
            database.add_documents(documents)
        # 새 해시값 저장 (이후 변경 여부 판단)
        save_hash(hash_file, current_hash)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
# 인덱스 선정
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
    )
index = pc.Index(index_name)

index_data()
