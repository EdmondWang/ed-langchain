import os
from operator import itemgetter

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.prebuilt import chat_agent_executor
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

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

# use agent to complete whole db integration
system_prompt = """
您是一个被设计用来与SQL数据可交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户制定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询并重试。
不要对数据库做任何DML语句（插入，更新，删除等）

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""

system_message = SystemMessage(content=system_prompt)

agent_executor = chat_agent_executor.create_tool_calling_executor(model=llm, tools=tools, prompt=system_message)


resp = agent_executor.invoke({
    'messages': [
        HumanMessage(content='请问员工表中有多少条记录？'),
    ]
})

result = resp['messages']
print(result)
print(len(result))
print(result[len(result) - 1])