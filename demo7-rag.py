import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import format_document, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_message_histories import ChatMessageHistory

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"]="ed-langchain"
os.environ["DEEPSEEK_API_KEY"]="sk-b99a159e8e3046ea9482d13d53e6e23a"

llm = ChatDeepSeek(model ="deepseek-chat")

#1 Load Document, could be from DB, word, wiki or documents
loader = WebBaseLoader(
    web_paths=['https://angular.dev/guide/signals'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('docs-app-main-content'))
    )
)

docs = loader.load()

# print(len(docs))
# print(docs)

#2 Split text response b/c big document not good for LLM to parse
"""
需要切割文本（split text），主要原因有：

1. **LLM 输入长度有限**  
   大语言模型（如 GPT、DeepSeek）每次能处理的文本有“最大 token 限制”（比如 4k、8k、16k tokens），太长会被截断或报错。

2. **提升检索效果**  
   RAG 检索时，把大文档切成小块，可以更精准地找到与问题相关的片段，提高答案相关性。

3. **减少噪音**  
   小块文本更聚焦，减少无关内容干扰模型判断。

4. **提升效率**  
   小块文本 embedding 更快，检索和召回速度也更高。

**总结：**  
切割文本是为了让 LLM 能高效、准确地处理和理解大文档内容，是 RAG 等应用的标准做法。
"""

"""
chunk_overlap 是为了尽量保证每一个 chunk的 语法完整性
"""
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = splitter.split_documents(docs)

# for s in splits:
#     print(f'{s}**\n')

#3 store documents into vector store
embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')

vector_store = Chroma.from_documents(documents=splits,embedding=embeddings)

#4 create retriever
retriever = vector_store.as_retriever()

#5 create a question template
system_prompt = """You are an assistant for question-answering tasks.
Ue the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Use the there sentences maximum and keep the answer concise.\n
{context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ('system',system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

#6 create chain, retriever first then combine with LLM to answer question
chain1 = create_stuff_documents_chain(llm=llm,prompt=prompt)

# chain2 = create_retrieval_chain(retriever=retriever, combine_docs_chain=chain1)

# resp = chain2.invoke({
#     'input': 'How to understand dependency of computed signal is dynamic?'
# })

# print(resp)

# create a prompt template for child chain
contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history,
formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as it."""

retriever_history_temp = ChatPromptTemplate.from_messages([
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# create child chain
history_chain = create_history_aware_retriever(llm,retriever, retriever_history_temp)

# persist chat history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# create parent chain
chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

resp1 = result_chain.invoke(
    {'input': 'What is computed Signal?'},
    config={'configurable': {'session_id': 'zs12345'}}
)

print('answer to 1st q')
print(resp1['answer'])

resp2 = result_chain.invoke(
    {'input': 'What are common way of creating it?'},
    config={'configurable': {'session_id': 'zs12345'}}
)

print('answer to 2nd q')
print(resp2['answer'])