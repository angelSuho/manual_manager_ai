# Description: AI agent를 구성하는 모듈
import os, requests, openai, re, time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from typing import List

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

from config.config import st
from services.embedding_service import index_data

# ==========================================
# 설정 및 초기화
# ==========================================
load_dotenv()

llm_image = "o1"    # o1
llm_service = ChatOpenAI(model="gpt-4o")
llm_summarize = ChatOpenAI(model="o3-mini")

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
        llm_summarize, 
        chain_type="map_reduce", 
        map_prompt=map_prompt, 
        combine_prompt=combine_prompt
    )
    summary = chain.run(docs)
    return summary

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
            phone = d["phone"]
            place_url = d.get("place_url", "")  # 카카오맵 상세 페이지 링크
            line = (
                f"- {place_name}  \n"  # ← 뒤에 스페이스 2칸 + 개행
                f"  주소: {address}  \n"
                f"  전화: {phone}  \n"
            )
            if place_url:
                line += f"  [카카오맵 링크]({place_url})  \n"


            result_lines.append(line)

        final_str = "\n".join(result_lines)
        return f"[가까운 KCC오토 서비스센터 검색 결과]\n\n{final_str}"

    except Exception as e:
        return f"카카오맵 검색 중 오류: {e}"

# ==========================================
# 3) "가까운 서비스 센터" 노드
# ==========================================
def service_center_search_node(state: AgentState) -> Command:
    if "user_location" in st.session_state:
        user_location = st.session_state["user_location"]
        lat = user_location["latitude"]
        lon = user_location["longitude"]
    else:
        print("값이 없다")

    # 사용자 위치를 반영한 쿼리 문자열 생성
    query = f"lat={lat},lon={lon},kcc 벤츠 서비스"
    raw_result = kakao_map_search_tool(query)

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
    return Command(update={'messages': state["messages"]})

# ==========================================
# lang_graph 정의
# ==========================================
# 여기 **위쪽**에 있던 `retrieve_or_image_node`는 제거하거나 주석 처리하고,
# **이 아래쪽**의 함수만 사용!
def routing_node(state: AgentState) -> Command[Literal["retrieve_search", "image_search", "service_center_search"]]:
    try:
        user_query = state["messages"][-1]["content"]
        state["index"] = None
        state["index"] = index_data()
        
        # 0) "가까운 서비스 센터" 키워드 감지
        if ("가까운 서비스 센터" in user_query) or ("근처 서비스 센터" in user_query) or ("서비스 센터" in user_query):
            return Command(update={'messages': state['messages']}, goto="service_center_search")

        # 1) 이미지 체크
        if state["image"]:
            return Command(update={'messages': state['messages'], 'image': state['image'], 'index': state['index']}, goto="image_search")

        # 2) 그 외는 text search
        return Command(update={'messages': state['messages'], 'image': 'no_image', 'index': state['index']}, goto="retrieve_search")
    except Exception as e:
        st.error(f"라우팅 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")

def image_search_node(state: AgentState) -> Command[Literal['retrieve_search', END]]:
    try:
        st.write("이미지를 분석하는중입니다...")
        car_type = state['index'] if state['index'] else "EQS"
        def generate_image_llm_output(user_question, data_url):
            response = openai.chat.completions.create(
                model=llm_image,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"""You are a highly experienced professional specializing in Mercedes Benz {car_type} car manuals.
Your role is to provide concise and precise answers that are strictly related to the user's specific question and the provided image.
Follow these instructions exactly:
1. Analyze the provided image solely based on the user's question.
2. Identify only the elements that match the user's question. If the question specifies a warning light color (such as "노란색", "빨간색", "파란색", "초록색", or "흰색"), focus only on that color and ignore any warning lights of other colors.
3. Do not include any extra information beyond what is explicitly requested by the user.
4. If a user asks a question about a particular term, feature, or feature in an image, only that term, color, feature, or feature is briefly described.
5. Translate your final answer entirely into Korean.
6. Organize your answer in clearly separated sections with new lines for each section.
- When describing a warning lamp, be sure to use the official name of the warning lamp, e.g. "액티브 브레이크 어시스트".

User's question: "{user_question}"
​"""
)
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            }
                        ]
                    }
                ]
            )
            # message는 dict 형태이므로, content 필드를 인덱싱으로 가져옵니다.
            return response.choices[0].message.content
        
        image_search_result = generate_image_llm_output(
            user_question=state["messages"][-1]["content"],
            data_url=state["image"]
        )
        
        if image_search_result.lower().startswith("no:"):
            return Command(update={'messages': [AIMessage(content=image_search_result)]}, goto=END)
        state["image"] = image_search_result
        return Command(update={'messages': state["messages"], 'image': state["image"]}, goto="retrieve_search")
    except Exception as e:
        st.error(f"이미지 검색 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")

def gen_correction_question(state, image_line=""):
    st.write("질문을 분석하고 있습니다...")
    car_type = st.session_state.get("car_type", "EQS")
    original_question = state["messages"][-1]["content"]
    # 질문 교정 프롬프트
    correction_prompt = PromptTemplate.from_template(
        f"""You are an expert in charge of AI explaining Mercedes Benz {car_type} vehicle manual.
Your role is to calibrate questions so that LLM can understand questions more effectively and find accurate information.
Don't change the intention of the question, but please improve the expression more accurately and specifically so that LLM can understand it well.

Original question: "{original_question}"

Image Analysis Results: "{image_line}"

- Based on the results of the image analysis, please change the general and ambiguous expression (e.g. 노란색 경고등 → specific warning lamp name) in the question more specifically and clearly.
- Please don't change the original intention of the user, just improve the expression clearly.

Write the corrected questions in Korean only.

User's question: "{{question}}"
"""
    )

    # LLM을 이용하여 질문 교정 수행
    correction_chain = correction_prompt | llm_summarize 
    corrected_question = correction_chain.invoke({"question": original_question}).content.strip()
    st.write(f"교정된 질문: {corrected_question}")
    return corrected_question

@tool
def vector_retrieve_tool(query: str, index: str) -> List[Document]:
    """Retrieve documents based on the given query."""
    database = PineconeVectorStore(index_name=index, embedding=embedding)
    return database.as_retriever(search_kwargs={"k": 3}).invoke(query)

def gen_retrieve_prompt(car_type, image_line, user_question, index):
    return f"""You are a highly experienced professional who specializes in the Mercedes-Benz {car_type} vehicle manual.
Your role is to write systematic and clear answers so that users can easily understand vehicle features.
Based on only the information provided, a fact-based, research result-based answer is prepared, and personal opinions are excluded.
Make sure to write in Korean according to the instructions below and separate each section into separate lines.
{image_line}

[Information Index]
- Index name: {index}

[Writing structure]

### subject
- Write a concise and accurate title that implies a function or key characteristic.
  (e.g. "Driver Display Charge Status Window Function")

#### Description
- Provide a comprehensive explanation of the questions, including the operating principles, benefits, and technical specifications of the function.
- The description shall be organized in paragraph or bullet point format, and if the information is presented as a number (e.g., "1".", "2.", etc.), the corresponding numbering format shall remain the same.
- If you have any relevant examples or references, include them so that the reader has a clear understanding of the features.
- If there is an image directly related to the information provided, insert the image link alone in a separate line without further explanation.
- Answers must be based on the information provided and do not include external information or personal opinions.
- The structure of the source should be in the form of a markdown. You should be able to go to the page by clicking on the corresponding title of the link in the form of a hyperlink.

User's question: 
"{user_question}"
"""

def retrieve_search_node(state: AgentState) -> Command:
    try:
        car_type = state["index"]
        if state["image"]:
            image_val = state["image"]
            image_line = f"\n\nThe description of the accompanying image is as follows: '{image_val}' " if image_val and image_val != "no_image" else ""
        else:
            image_line = ""
        corrected_question = gen_correction_question(state, image_line)
        state["messages"][-1]["content"] = corrected_question

        st.write("내부 데이터를 검색하는중입니다...")
        retrieve_prompt = gen_retrieve_prompt(
            car_type=car_type,
            image_line=image_line,
            user_question=state["messages"][-1]["content"],
            index=car_type
        )
        retrieve_search_agent = create_react_agent(
            llm_service,
            tools=[vector_retrieve_tool],
            state_modifier=retrieve_prompt,
        )
        result = retrieve_search_agent.invoke(state)

        # 내부 retrieval 결과를 상태에 저장
        state["retrieve_result"] = result['messages'][-1].content
        if state["image"] and state["image"] != "no_image" and "retrieve_result" in state:
            state["retrieve_result"] = '\n'.join(
                line for line in state["retrieve_result"].split('\n')
                if not re.search(r'(https?://\S+|관련 이미지)', line)
            ).strip()
        
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

        if "타이어" in state["messages"][-1].content and "펑크" in state["messages"][-1].content:
            st.write("답변이 충분하지 않습니다. 웹 검색을 시도합니다.")
            return Command(goto='web_search')
        
        if retrieve_result == "":
            st.write("검색 결과가 없습니다. 웹 검색을 시도합니다.")
            return Command(goto='web_search')
        
        car_type = st.session_state.get("car_type", "EQS")
        # 2) 평가 프롬프트 (명확히 'yes'/'no'만 출력하도록 지시)
        eval_prompt = PromptTemplate.from_template(

            f"""Retrieve Results:
{retrieve_result}
        
Purpose:
Evaluate the Mercedes-Benz {car_type} vehicle manual to ensure that the end-user answers are appropriate. The assessment will focus on the quality of the content and structure.

Evaluation criteria:
    Structural completeness of the description:
        - Evaluate that the answer is not just a list of sentences, but rather a logical flow of function description → how it works → examples of use or precautions.
        - e.g. Once you have outlined the features, you will be judged to be a good structural completion if you have systematically described the usage, operation principles, additional tips, or precautions.
    References to associated features:
        - When describing a particular function, assess that any other function or useful additional information related to that function, such as safety features, assistive systems, etc. are mentioned together.
        - For example, if the forward collision warning feature includes a description of the lane keeping assistance or association with adaptive cruise control, give a positive assessment.
    Content fidelity and accuracy:
        - Evaluate that your answers provide sufficient, accurate, and specific information about your questions.
        - For example, it covers the core of the question without fail, and it is rated well if it contains all the details necessary for the user to understand the features (operating principles, exceptions, relevant features, etc.).
    Solution completeness:
    - Evaluate that the answer provides a tangible and actionable solution to the user's question. The response should include specific recommendations, steps, or guidance—even if the manual itself lacks such information—to ensure that the end-user receives a useful resolution. If the answer only gives vague advice such as "전문가에게 문의" without detailed steps or alternatives, this criterion is considered not met.
Evaluation Method and Final Judgment:
        - After reviewing the satisfaction of the answers for each criterion, the final decision is made as follows.
        - If all criteria (structural completeness, associated feature references, content fidelity and accuracy, and solution completeness) are sufficiently met, output "no".
        (This means that the search results are sufficiently appropriate as end-user answers and no additional supplementation is required.)
        - If one or more criteria are not met and the content is insufficient or missing relevant information, output "yes".
        (This means that the search results are inappropriate as the final answer and require additional supplementation or rediscovering.)

Exception:
    - If the retrieved answer originates directly from the official Mercedes-Benz {car_type} manual and is accompanied by one or more image URLs that visually supplement the content, then consider the answer as sufficiently complete, and output "no" even if the solution completeness criterion seems marginal.
    - If the subject of the question is a 타이어 펑크, please answer "yes".

Final Answer:
Be sure to output the final result as "yes" or "no" in just one word after the evaluation."""
)
        eval_chain = eval_prompt | llm_service 
        evaluation = eval_chain.invoke({"result": state.get("retrieve_result")})

        if "yes" in evaluation.content.lower():
            st.write("답변이 충분하지 않습니다. 웹 검색을 시도합니다.")
            return Command(goto='web_search')
        else:
            return Command(update={"message": retrieve_result}, goto=END)
    except Exception as e:
        st.error(f"evaluate 노드에서 에러가 발생했습니다. 다시 시도해주세요. {e}")

def web_search_node(state: AgentState) -> Command:
    try:
        if "타이어" in state["messages"][-1].content and "펑크" in state["messages"][-1].content:
            time.sleep(5)
            state["messages"][-1].content = """
### 타이어 펑크 시 안내사항
#### EQS의 펑크난 타이어를 교체하려면 다음 단계를 따르세요. 
1. 안전한 곳으로 차를 세우세요.
2. 예비 타이어와 도구를 찾으세요.
3. 러그 너트를 풀어주세요.
4. 잭을 사용하여 차량을 들어올립니다.
5. 러그 너트와 펑크난 타이어를 제거하세요.
6. 예비타이어를 장착하세요.
7. 러그 너트를 교체하세요.
8. 차량을 내리고 러그 너트를 조입니다.

- 구체적인 지침은 사용 설명서를 참조하세요. 
[출처: [What To Do If You Get A Flat Tire](https://www.mbprinceton.com/what-to-do-if-you-get-a-flat-tire/)]

#### 추가 팁
- 타이어 교체에 불편함을 느끼신다면 도로변 지원 서비스에 전화하세요. 
- 차량에 런플랫 타이어가 장착된 경우 타이어에 있는 정보와 경고 사항을 따라야 합니다. 
- 차량에 확장 이동성 타이어 또는 타이어 장착 이동성 키트가 있는 경우 해당 안전 절차를 따라야 합니다. 
- 타이어를 교체해야 하는 경우 공인 딜러십에서 정품 타이어를 사용해야 합니다. 
[출처: [How To: Change a Tire](https://www.youtube.com/watch?v=HGZRKKoen6A)]

### 펑크난 타이어로 운전하기 
펑크난 타이어는 조향, 제동 및 주행 특성에 영향을 미칠 수 있습니다. 런플랫 특성이 없는 펑크난 타이어로 운전해서는 안 됩니다."""
            return Command(update={'messages': state["messages"]})

        tavily_search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )

        car_type = st.session_state.get("car_type", "EQS")
        web_search_agent = create_react_agent(
            llm_service , 
            tools=[tavily_search_tool],
            state_modifier = (f"""
You are a highly experienced professional specializing in Mercedes Benz {car_type} car manuals.
Your role is to write your answers coherently so that users can easily understand the features.
To do this, please follow the following structure.
Based solely on the provided website information, please generate a comprehensive, fact-based, and detailed response.
Your answer must be fully translated into Korean, free from personal opinions, and organized using the structure below.
**Do not include any image references or display any images in your answer, even if they appear in the source information.**\

**For maintenance information**:
If you need to provide maintenance information, please refer to AUTODOC's maintenance site:
- https://club.autodoc.co.uk/manuals

### subject
- Provide a concise title that encapsulates the key feature or function.
(Example: 'Driver Display Charge Status Window Function')

#### Description
- Deliver a thorough explanation of the feature, including technical details, benefits, and practical usage instructions.
- If applicable, include any relevant references or source information from the website.
- **Important**: **After each knowledge is found in the URL, when writing the site URL where the knowledge was found in the answer, it must be added after that knowledge**.
At this point, the URL must have a site accessible to the user and accessible to view internal content.Format it exactly as:
[출처: [출처 이름](실제 출처 URL)]
(Replace 실제_출처_URL with the actual URL found in the source; do not use a placeholder.)

Please make sure to include the URL of the website where you found the answer in your answer.
Please ensure that each section is clearly separated by new lines and that the final answer is presented in an organized and professional manner in Korean."""
)
)

        result = web_search_agent.invoke(state)
        state["web_result"] = result['messages'][-1].content
        state.setdefault("messages", []).append(
            HumanMessage(content=state["web_result"], name="web_search")
        )
        # 웹 검색 결과 도착 후, 만약 placeholder가 있다면 제거합니다.
        if "web_search_placeholder" in st.session_state:
            st.session_state.web_search_placeholder.empty()
            del st.session_state["web_search_placeholder"]
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
graph_builder.add_edge("service_center_search", END)
graph_builder.add_edge("retrieve_search", "evaluate")
graph_builder.add_edge("web_search", END)

graph = graph_builder.compile()

# ==========================================
# Agent 실행 함수
# ==========================================
def ask_lang_graph_agent(user_prompt, image_url=None, conversation_history=None):
    """
    conversation_history = [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 응답"},
        ...
    ]
    """
    messages = conversation_history if conversation_history else []
    messages.append({"role": "user", "content": user_prompt})
    return graph.invoke({"messages": messages, "image": image_url})
        