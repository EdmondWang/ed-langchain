import os

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langserve import add_routes

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"]="ed-langchain"

llm = ChatDeepSeek(model ="deepseek-chat")

parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，对{domain}载人工具领域非常了解。"),
    ("user", "{text}")
])

# create chain
chain = prompt_template | llm | parser

app = FastAPI(title="My Lang Chain Service",version="v1.0.0",description="For a better life")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
