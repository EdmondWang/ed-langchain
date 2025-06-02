
import os
import ssl
from typing import Optional, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic.v1 import BaseModel, Field

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "ed-langchain"

llm = ChatDeepSeek(model="deepseek-chat")

persist_dir = 'demo10_chroma_data_dir'

embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')

vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# result = vector_store.similarity_search_with_score('How to build a research assistant?')
# print(result[0][0])

system = '''
你是一个将用户问题转换为数据库查询的专家。你可以访问关于如何汽车类的的应用程序的软件库的教程视频数据库。
如果有你不熟悉的缩略词或单词，不要试图改变它们。
'''

prompt = ChatPromptTemplate.from_messages([
    ('system', system),
    ('human', '{question}')
])


# 内容的相似性和发布年份
class Search(BaseModel):
    query: str = Field(default=None, description='Similarity search query applied to video transcripts')
    publish_year: Optional[int] = Field(default=None, description='Publish Year of video')


chain = {'question': RunnablePassthrough()} | prompt | llm.with_structured_output(Search)

# resp = chain.invoke('How to build a research assistant?')
# print(resp)

# resp2 = chain.invoke('videos on RAG published in 2023')
# print(resp2)

def retrieve(search: Search)-> List[Document]:
    query = search.query
    publish_year = search.publish_year

    _filter = None

    if publish_year:
        # create retrieve condition
        _filter = {
            'publish_year': {
                '$eq': publish_year
            },
        }

    return vector_store.similarity_search(query=query, filter=_filter)

new_chain = chain | retrieve

result = new_chain.invoke('Model Y 2025 款有哪些新亮点')

print(len(result))
# print(result)
print([
    (doc.metadata.get('title', ''), doc.metadata.get('publish_year', ''))
    for doc in result
])