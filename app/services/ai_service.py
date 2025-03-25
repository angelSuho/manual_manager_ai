# Description: AI agent를 구성하는 모듈
import os, requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from typing import List

## image similarity finder
import cv2
import torch
import clip
from PIL import Image
import numpy as np
import pickle
import re

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from typing import List, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

from config.config import IMAGE_PATTERN, st
from services.embedding_service import index_data

# ==========================================
# 설정 및 초기화
# ==========================================
load_dotenv()

model = "gpt-4o"
llm = ChatOpenAI(model=model)
embedding = OpenAIEmbeddings(model="text-embedding-3-large") # text-embedding-ada-002, text-embedding-3-large

class AgentState(TypedDict, total=False):
    messages: List[HumanMessage]
    image: Optional[str]
    index: str

# --- 요약 함수 추가 ---
def summarize_text(text):
    # 입력 텍스트를 Document 객체로 감싸기
    docs = [Document(page_content=text)]
    
    # map 단계 프롬프트 (각 문서를 요약할 때)
    map_prompt_template = (
        "다음 내용을 한국어로 간결하게 요약해 주세요:\n\n{text}\n\n요약:"
    )
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    
    # combine 단계 프롬프트 (여러 요약을 종합할 때)
    combine_prompt_template = (
        "다음은 여러 문서의 요약입니다. 이를 종합하여 한국어로 한 문장으로 요약해 주세요:\n\n{text}\n\n최종 요약:"
    )
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
    
    # 요약 체인 생성 (map_reduce 체인)
    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce", 
        map_prompt=map_prompt, 
        combine_prompt=combine_prompt
    )
    summary = chain.run(docs)
    return summary

# ==========================================
# 이미지 관련 설정 및 함수 초기화
# ==========================================
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

icon_folder = "./data/images/icons/계기판_디스플레이"
embeddings_path = "./data/icon_embeddings.pkl"


# ==========================================
# KakaoMap 검색을 수행하는 Tool 추가
# ==========================================
@tool
def kakao_map_search_tool(query: str) -> str:
    """
    사용자의 위치(예: lat/lon)와 검색어를 받아,
    카카오맵에서 가까운 장소를 찾아주는 예시 함수.
    query 예시: 'lat=37.499,lon=127.0264,KCC오토 서비스센터'
    """
    kakao_api_key = os.getenv("KAKAO_REST_API_KEY")
    if not kakao_api_key:
        return "카카오 API 키가 설정되지 않았습니다."

    parts = query.split(",")
    if len(parts) < 3:
        return "쿼리 형식이 잘못되었습니다. 예: 'lat=37.499,lon=127.0264,KCC오토 서비스센터'"

    lat_part = parts[0].replace("lat=","").strip()
    lon_part = parts[1].replace("lon=","").strip()
    keyword = parts[2].strip()

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = { "Authorization": f"KakaoAK {kakao_api_key}" }
    params = {
        "query": keyword,
        "x": lon_part,  # 카카오맵은 x=경도, y=위도
        "y": lat_part,
        "radius": 20000  # 예: 반경 20km
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if "documents" not in data or not data["documents"]:
            return "검색 결과가 없습니다."

        docs = data["documents"][:3]
        result_lines = []
        for d in docs:
            place_name = d["place_name"]
            address = d["road_address_name"] or d["address_name"]
            distance = d.get("distance", "?")  # meter 단위
            phone = d["phone"]
            place_url = d.get("place_url", "")  # 카카오맵 상세 페이지 링크

            # Markdown 링크 형태로 넣어줍니다.
            # "[카카오맵 링크](https://...)" 형식
            line = (
                f"- {place_name} (거리: {distance}m)  \n"  # ← 뒤에 스페이스 2칸 + 개행
                f"  주소: {address}  \n"
                f"  전화: {phone}  \n"
            )
            if place_url:
                line += f"  [카카오맵 링크]({place_url})  \n"


            result_lines.append(line)

        final_str = "\n".join(result_lines)
        return f"[가까운 KCC오토 서비스센터 검색 결과]\n{final_str}"

    except Exception as e:
        return f"카카오맵 검색 중 오류: {e}"

# ==========================================
# 3) "가까운 서비스 센터" 노드
# ==========================================
def service_center_search_node(state: AgentState) -> Command:
    # 1) ...KakaoMap 툴 호출...
    raw_result = kakao_map_search_tool("lat=37.499,lon=127.0264,KCC오토 서비스센터")

    # 2) raw_result를 파싱하거나, 여기서는 그냥 문자열 붙이기
    final_answer = """\

    
### 가까운 KCC오토 서비스 센터
다음은 가까운 메르세데스 벤츠 KCC오토 서비스 센터 목록입니다. 차량 점검 및 수리를 위한 서비스 제공이 가능합니다.

""" + raw_result + """


위 서비스 센터에서는 정기 점검, 자동차 수리 및 기타 다양한 서비스를 제공합니다. 
필요에 따라 미리 전화로 예약하시기를 권장합니다.

[출처: 카카오맵]
"""
    # 3) 메시지에 저장
    state["service_center_result"] = final_answer
    state.setdefault("messages", []).append(
        HumanMessage(content=final_answer, name="service_center_search")
    )
    return Command(update={'messages': state["messages"]}, goto="evaluate")

# ==========================================
# lang_graph 정의
# ==========================================
# 여기 **위쪽**에 있던 `retrieve_or_image_node`는 제거하거나 주석 처리하고,
# **이 아래쪽**의 함수만 사용!
def routing_node(state: AgentState) -> Command[Literal["retrieve_search", "image_search", "service_center_search"]]:
    try:
        user_query = state["messages"][0]["content"]
        state["index"] = None
        state["index"] = index_data()
        
        # 0) "가까운 서비스 센터" 키워드 감지
        if ("가까운 서비스 센터" in user_query) or ("근처 서비스 센터" in user_query):
            return Command(update={'messages': state['messages']}, goto="service_center_search")

        # 1) 이미지 체크
        match = re.search(IMAGE_PATTERN, user_query, flags=re.IGNORECASE)
        if match:
            image_url = match.group(0)
            updated_query = re.sub(IMAGE_PATTERN, '', user_query, count=1, flags=re.IGNORECASE).strip()
            state["messages"][0]["content"] = updated_query
            state["image"] = image_url
            return Command(update={'messages': state['messages'], 'image': state['image'], 'index': state['index']}, goto="image_search")

        # 2) 그 외는 text search
        return Command(update={'messages': state['messages'], 'index': state['index']}, goto="retrieve_search")
    except Exception as e:
        st.error(f"라우팅 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")


def image_search_node(state: AgentState) -> Command:
    try:
        target_image_path = state["image"]
        target_image = Image.open(target_image_path).convert("RGB")

        # 저장된 아이콘 이미지 임베딩 로딩 또는 계산 함수
        def compute_and_save_embeddings():
            embeddings = {}
            for filename in os.listdir(icon_folder):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                image_path = os.path.join(icon_folder, filename)
                # imread_unicode와 resize_image는 사용자의 이미지 전처리 함수라고 가정합니다.
                img_cv = imread_unicode(image_path)
                if img_cv is None:
                    continue
                img_cv = resize_image(img_cv, max_dim=500)
                # 이미지 전처리 (PIL 이미지로 변환 후 CLIP 전처리)
                img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model.encode_image(input_tensor)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embeddings[filename] = embedding.cpu()  # CPU로 이동하여 저장
            # 임베딩 저장 (pickle 사용)
            with open(embeddings_path, "wb") as f:
                pickle.dump(embeddings, f)
            
        def load_embeddings():
            if os.path.exists(embeddings_path):
                with open(embeddings_path, "rb") as f:
                    embeddings = pickle.load(f)
                return embeddings
            else:
                return None

        # 저장된 임베딩이 없다면 계산하고 저장
        icon_embeddings = load_embeddings()
        if icon_embeddings is None:
            compute_and_save_embeddings()
            icon_embeddings = load_embeddings()

        # 타겟 이미지 전처리 및 임베딩 추출, 정규화
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
        state["image"] = results[0][0]

        # state에 추가된 정보를 포함하여 업데이트된 state를 반환
        return Command(update=state)
    except Exception as e:
        st.error(f"이미지 검색 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")

@tool
def vector_retrieve_tool(query: str, index: str) -> List[Document]:
    """Retrieve documents based on the given query."""
    database = PineconeVectorStore(index_name=index, embedding=embedding)
    return database.as_retriever(search_kwargs={"k": 3}).invoke(query)

def dynamic_state_modifier(agent_input: AgentState) -> str:
    image_val = agent_input.get("image")
    image_line = f"The Topic of the provided target image is {image_val}. " if image_val and image_val != "no_image" else ""
    car_type = st.session_state.get("car_type", "EQS")
    index = agent_input.get("index", "eqs")
    return (
        f"""
        You are a highly experienced professional specializing in Mercedes Benz {car_type} car manuals. 
        {image_line}
        The name of the 'index' is {index}, which examines the information in this index.
        Based solely on the provided information, please deliver a well-researched and fact-based answer—free from personal opinions.
        Translate the answer into Korean and ensure that each section is clearly formatted on a separate line, as described below.\
        
        ## {{제목}}
        - Provide a concise yet precise title that encapsulates the key feature or function.
        (Example: "Driver Display Charge Status Window Function")

        ### {{상세 설명}}
        - Offer a comprehensive explanation of the feature, including its operational mechanism, benefits, and technical specifications.
        - Structure your explanation with clarity, using paragraphs or bullet points where appropriate.
        - If the provided content contains numbered items (e.g., "1 ...", "2 ...", etc.), please preserve and format these as numbered lists in your answer.
        - If applicable, include any relevant references or examples to support your answer.
        - Additionally, if there is an image directly related to the provided information, remember image link.

        Please ensure your final answer is entirely in Korean and adheres strictly to the structure provided.
        """
    )

def retrieve_search_node(state: AgentState) -> Command:
    try:
        st.session_state.car_type = state["index"]
        retrieve_search_agent = create_react_agent(
            llm,
            tools=[vector_retrieve_tool],
            state_modifier=dynamic_state_modifier(state)  # 함수로 교체
        )
        result = retrieve_search_agent.invoke(state)
        # 내부 retrieval 결과를 상태에 저장
        state["retrieve_result"] = result['messages'][-1].content
        
        # 기존 messages 리스트에 retrieval 결과를 추가
        state.setdefault("messages", []).append(
            HumanMessage(content=state["retrieve_result"], name="retrieve_search")
        )
        # 전체 messages 리스트를 업데이트
        return Command(update={'messages': state["messages"]})
    except Exception as e:
        st.error(f"retrieve 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")

def evaluate_node(state: AgentState) -> Command[Literal['web_search', END]]:
    try:
        retrieve_result = state["messages"][-1].content
        
        if retrieve_result == "":
            st.write("검색 결과가 없습니다. 웹 검색을 시도합니다.")
            return Command(goto='web_search')
        
        car_type = st.session_state.get("car_type", "EQS")
        # 2) 평가 프롬프트 (명확히 'yes'/'no'만 출력하도록 지시)
        eval_prompt = PromptTemplate.from_template(

            f"""You are an expert on Mercedes Benz {car_type} car manuals.
        Please evaluate the following retrieved results for completeness and relevance.
        If the retrieved content is less than 500 characters, or if it does not comprehensively address the user's query with detailed technical information, respond with exactly "yes". Otherwise, respond with exactly "no".
        You must respond with one word only, either "yes" or "no", with no additional commentary.
        Retrieve Results:
        {{result}}
        """
        
        )
        eval_chain = eval_prompt | llm
        evaluation = eval_chain.invoke({"result": state.get("retrieve_result")})
        if "yes" in evaluation.content.lower():
            st.write("답변이 충분하지 않습니다. 웹 검색을 시도합니다.")
            return Command(goto='web_search')
        else:
            return Command(goto=END)
    except Exception as e:
        st.error(f"evaluate 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")

def web_search_node(state: AgentState) -> Command:
    try:
        tavily_search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )

        car_type = st.session_state.get("car_type", "EQS")
        web_search_agent = create_react_agent(
            llm, 
            tools=[tavily_search_tool],
            state_modifier = (f"""
                You are a highly experienced professional specializing in Mercedes Benz {car_type} car manuals.
                Based solely on the provided website information, please generate a comprehensive, fact-based, and detailed response.
                Your answer must be fully translated into Korean, free from personal opinions, and organized using the structure below.
                **Do not include any image references or display any images in your answer, even if they appear in the source information.**\
                
                ## {{제목}}
                - Provide a concise title that encapsulates the key feature or function.
                (Example: 'Driver Display Charge Status Window Function')

                ### {{상세 설명}}
                - Deliver a thorough explanation of the feature, including technical details, benefits, and practical usage instructions.
                - If applicable, include any relevant references or source information from the website.
                - IMPORTANT: At the very end of your answer, on a new line, include the actual source URL extracted from the provided website information.
                Format it exactly as:
                [출처: 실제_출처_URL]
                (Replace 실제_출처_URL with the actual URL found in the source; do not use a placeholder.)
                
                Please ensure that each section is clearly separated by new lines and that the final answer is presented in an organized and professional manner in Korean.
                """
            )
        )

        result = web_search_agent.invoke(state)
        state["web_result"] = result['messages'][-1].content
        state.setdefault("messages", []).append(
            HumanMessage(content=state["web_result"], name="web_search")
        )
        return Command(update={'messages': state["messages"]})
    except Exception as e:
        st.error(f"웹 검색 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")

# 그래프 구성
graph_builder = StateGraph(AgentState)
graph_builder.add_node("retrieve_search", retrieve_search_node)
graph_builder.add_node("evaluate", evaluate_node)
graph_builder.add_node("web_search", web_search_node)
graph_builder.add_node("image_search", image_search_node)
graph_builder.add_node("routing", routing_node)
graph_builder.add_node("service_center_search", service_center_search_node)

# 엣지 연결
graph_builder.add_edge(START, "routing")
graph_builder.add_edge("service_center_search", "evaluate")
graph_builder.add_edge("image_search", "retrieve_search")
graph_builder.add_edge("retrieve_search", "evaluate")
graph_builder.add_edge("web_search", END)

graph = graph_builder.compile()

# ==========================================
# Agent 실행 함수
# ==========================================
def ask_lang_graph_agent(query):
    return graph.invoke({"messages": [{"role": "user", "content": query}]})
