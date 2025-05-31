import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_PROJECT']='ed-langchain'

llm = ChatDeepSeek(model ='deepseek-chat')

# prepare testing data
documents = [
    Document(page_content='狗是伟大的伴侣，以其忠诚和友好而闻名', metadata={'source': '宠物文档'}),
    Document(page_content='猫是独立的动物，通常喜欢自己的空间', metadata={'source': '宠物文档'}),
    Document(page_content='鹦鹉能够模仿人类的语言，非常有趣', metadata={'source': '宠物文档'}),
    Document(page_content='仓鼠体型小巧，是儿童喜欢的宠物之一', metadata={'source': '宠物文档'}),
    Document(page_content='金鱼色彩斑斓，适合观赏', metadata={'source': '宠物文档'}),
    Document(page_content='兔子性格温顺，喜欢吃胡萝卜', metadata={'source': '宠物文档'}),
    Document(page_content='乌龟寿命长，象征着长寿', metadata={'source': '宠物文档'}),
    Document(page_content='蜥蜴喜欢晒太阳，是冷血动物', metadata={'source': '宠物文档'}),
    Document(page_content='刺猬有很多刺，但其实很温顺', metadata={'source': '宠物文档'}),
    Document(page_content='蛇有各种颜色和花纹，有些人喜欢养蛇作为宠物', metadata={'source': '宠物文档'}),
    Document(page_content='松鼠活泼好动，喜欢吃坚果', metadata={'source': '宠物文档'}),
    Document(page_content='鹦鹉喜欢群居，能发出多种叫声', metadata={'source': '宠物文档'}),
    Document(page_content='狗能帮助人类完成许多任务，如导盲和搜救', metadata={'source': '宠物文档'}),
    Document(page_content='猫咪喜欢晒太阳，经常在窗台上打盹', metadata={'source': '宠物文档'}),
    Document(page_content='金鱼需要定期换水，保持水质清洁', metadata={'source': '宠物文档'}),
    Document(page_content='兔子喜欢在草地上奔跑', metadata={'source': '宠物文档'}),
    Document(page_content='乌龟冬天会冬眠', metadata={'source': '宠物文档'}),
    Document(page_content='仓鼠喜欢储存食物在嘴巴里', metadata={'source': '宠物文档'}),
    Document(page_content='刺猬遇到危险时会蜷缩成球', metadata={'source': '宠物文档'}),
    Document(page_content='蛇通过吐舌头感知气味', metadata={'source': '宠物文档'}),
    Document(page_content='松鼠会在树上筑巢', metadata={'source': '宠物文档'}),
    Document(page_content='鹦鹉需要丰富的玩具来保持活力', metadata={'source': '宠物文档'}),
    Document(page_content='狗喜欢和主人互动', metadata={'source': '宠物文档'}),
    Document(page_content='猫有很强的夜视能力', metadata={'source': '宠物文档'}),
    Document(page_content='金鱼喜欢群游', metadata={'source': '宠物文档'}),
    Document(page_content='兔子的牙齿会不断生长', metadata={'source': '宠物文档'}),
    Document(page_content='乌龟喜欢晒背', metadata={'source': '宠物文档'}),
    Document(page_content='仓鼠夜间活动频繁', metadata={'source': '宠物文档'}),
    Document(page_content='刺猬喜欢独处', metadata={'source': '宠物文档'}),
    Document(page_content='蛇需要恒温环境', metadata={'source': '宠物文档'}),
    Document(page_content='松鼠冬天会储存食物', metadata={'source': '宠物文档'}),
    Document(page_content='鹦鹉羽毛色彩鲜艳', metadata={'source': '宠物文档'}),
    Document(page_content='狗能感知主人的情绪', metadata={'source': '宠物文档'}),
    Document(page_content='猫喜欢用爪子抓东西', metadata={'source': '宠物文档'}),
    Document(page_content='金鱼对水温变化敏感', metadata={'source': '宠物文档'}),
    Document(page_content='兔子喜欢挖洞', metadata={'source': '宠物文档'}),
    Document(page_content='乌龟行动缓慢但很有耐心', metadata={'source': '宠物文档'}),
    Document(page_content='仓鼠喜欢跑轮子', metadata={'source': '宠物文档'}),
    Document(page_content='刺猬喜欢吃昆虫', metadata={'source': '宠物文档'}),
    Document(page_content='蛇有脱皮的习性', metadata={'source': '宠物文档'}),
    Document(page_content='松鼠尾巴蓬松，可以保暖', metadata={'source': '宠物文档'}),
    Document(page_content='鹦鹉喜欢洗澡', metadata={'source': '宠物文档'}),
    Document(page_content='狗需要定期遛弯', metadata={'source': '宠物文档'}),
    Document(page_content='猫喜欢高处', metadata={'source': '宠物文档'}),
    Document(page_content='金鱼寿命可达十年以上', metadata={'source': '宠物文档'}),
    Document(page_content='兔子喜欢群居', metadata={'source': '宠物文档'}),
    Document(page_content='乌龟喜欢安静的环境', metadata={'source': '宠物文档'}),
    Document(page_content='仓鼠喜欢啃咬东西', metadata={'source': '宠物文档'}),
    Document(page_content='刺猬有很强的嗅觉', metadata={'source': '宠物文档'}),
    Document(page_content='蛇有灵活的身体', metadata={'source': '宠物文档'}),
]

# model = SentenceTransformer('moka-ai/m3e-base')

# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')

# Create instance of vector space
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='./persist_db')

# Similarity higher then score get lower
# print('Similarity higher then score get lower')
# print(vector_store.similarity_search_with_score('咖啡猫'))

retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# print('The result of retriever')
# print(retriever.batch(['刺猬', '乌龟', '鹦鹉喜欢学人说话']))

# prompt template
message = '''
使用提供的上下文仅回答这个问题。
{question}
上下文：
{context}    
'''

prompt_template = ChatPromptTemplate.from_messages([
    ('human', message)
])

chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_template | llm

resp = chain.invoke('请问谁是人类的好伙伴？')
print('** answer to "请问谁是人类的好伙伴"')
print(resp)

resp = chain.invoke('请介绍狗')
print('** answer to "请介绍狗"')
print(resp)
