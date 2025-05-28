import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_message_histories import ChatMessageHistory

# ========== 0. 配置环境变量（API Key、追踪等） ==========
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "ed-langchain"
os.environ["DEEPSEEK_API_KEY"] = "sk-b99a159e8e3046ea9482d13d53e6e23a"

# ========== 1. 初始化大语言模型（LLM） ==========
llm = ChatDeepSeek(model="deepseek-chat")

# ========== 2. 加载网页文档（抓取知识源） ==========
# 这里只抓取 Angular Signals 指南页面的主要内容
web_loader = WebBaseLoader(
    web_paths=['https://angular.dev/guide/signals'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('docs-app-main-content'))
    )
)
raw_documents = web_loader.load()

# ========== 3. 文本切块（分割大文档为小片段） ==========
# 为什么要切块？因为大模型输入长度有限，分块后检索和理解更高效
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(raw_documents)

# ========== 4. 文档向量化并存入向量数据库 ==========
# 用 HuggingFace 的 embedding 模型将文本转为向量，方便后续语义检索
embedding_model = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')
vector_db = Chroma.from_documents(documents=chunked_documents, embedding=embedding_model)

# ========== 5. 创建检索器（Retriever） ==========
retriever = vector_db.as_retriever()

# ========== 6. 构建主问答 Prompt 模板 ==========
# 这个模板告诉 LLM 如何用检索到的上下文回答问题
main_qa_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
{context}
"""
main_qa_prompt = ChatPromptTemplate.from_messages([
    ('system', main_qa_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# ========== 7. 创建文档拼接链（stuff chain） ==========
# 负责把检索到的文档片段和问题拼接后交给 LLM 生成答案
stuff_chain = create_stuff_documents_chain(llm=llm, prompt=main_qa_prompt)

# ========== 8. 构建历史感知检索 Prompt ==========
# 这个模板让 LLM 能理解多轮对话中的上下文引用
history_aware_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history,
formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as it."""
history_aware_prompt = ChatPromptTemplate.from_messages([
    ('system', history_aware_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# ========== 9. 创建历史感知检索链 ==========
# 让检索器能理解多轮对话中的上下文引用
history_aware_retriever_chain = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=history_aware_prompt
)

# ========== 10. 持久化对话历史（每个 session 独立保存） ==========
session_history_store = {}
def get_session_history(session_id: str):
    if session_id not in session_history_store:
        session_history_store[session_id] = ChatMessageHistory()
    return session_history_store[session_id]

# ========== 11. 串联检索链和生成链，形成完整 RAG 问答流程 ==========
# 先用历史感知检索链找相关文档，再用 stuff_chain 生成答案
rag_chain = create_retrieval_chain(history_aware_retriever_chain, stuff_chain)

# ========== 12. 支持多轮对话的 RAG Chain ==========
# RunnableWithMessageHistory 能自动管理对话历史，实现多轮问答
multi_turn_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# ========== 13. 多轮问答示例 ==========
# 第一次提问
response1 = multi_turn_rag_chain.invoke(
    {'input': 'What is computed Signal?'},
    config={'configurable': {'session_id': 'zs12345'}}
)
print('answer to 1st question:')
print(response1['answer'])

# 第二次提问（上下文会自动带入）
response2 = multi_turn_rag_chain.invoke(
    {'input': 'What are common way of creating it?'},
    config={'configurable': {'session_id': 'zs12345'}}
)
print('answer to 2nd question:')
print(response2['answer'])