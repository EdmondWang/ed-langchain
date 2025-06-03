import os
import ssl
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from pydantic.v1 import BaseModel, Field

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "ed-langchain"

llm = ChatDeepSeek(model="deepseek-chat")


class Person(BaseModel):
    name: Optional[str] = Field(default=None, description='Person name')
    hair_color: Optional[str] = Field(default=None, description='Person hair color')
    height_in_meters: Optional[str] = Field(default=None, description='Person height')

class ManyPerson(BaseModel):
    people: List[Person]


prompt = ChatPromptTemplate.from_messages([(
    'system',
    '你是一个专业的提取算法。只从未结构化的文本中提取信息。如果你不确定要提取的属性的值, 该属性值返回None'),
    ('human', '{text}')
])

chain = {'text': RunnablePassthrough()} | prompt | llm.with_structured_output(schema=ManyPerson)

text = '马路上走来一个女生，名字不知道，长长的黑头发披在肩上，大概1米7左右;走在她旁边的是她男朋友，叫刘海，比她高10厘米'
resp = chain.invoke(text)

print(resp)