import os
from operator import itemgetter

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_message_histories import ChatMessageHistory
from sqlalchemy import create_engine

# ========== 0. 配置环境变量（API Key、追踪等） ==========
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "ed-langchain"

llm = ChatDeepSeek(model="deepseek-chat")

# LangChian use sqlalchemy to connect DB
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'study_langchain'
USERNAME = 'root'
PASSWORD = 'Dinuan_1209'
MYSQl_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQl_URI)

# test if db connected
# print(db.get_usable_table_names())
# print(db.run('SELECT * FROM study_langchain.employee;'))

# Generate SQL query according to question
test_chain = create_sql_query_chain(llm=llm,db=db)
# resp = test_chain.invoke({'question': '请问员工表中有多少条记录？'})
# print(resp)

answer_prompt = ChatPromptTemplate.from_template("""
你是一个SQL专家。
给定以下用户问题，SQL语句和SQL执行后的结果，回答用户问题。
注意，在递交 生成的 SQL query 给 mysql 查询前，去除多余的 SQLQuery 文本。

Human Question: {question}
Generated SQL query: {query}
SQL Result: {result}

回答:
""")

# create a tool to execute sql query
execute_sql_tool = QuerySQLDatabaseTool(db=db)

# generate sql query -> execute sql query -> pass prompt template
chain = (RunnablePassthrough
         .assign(query=test_chain)
         .assign(result=itemgetter('query') | execute_sql_tool)
         | answer_prompt
         | llm
         | StrOutputParser())

resp = chain.invoke({'question': '请问员工表中有多少条记录？'})

print(resp)


