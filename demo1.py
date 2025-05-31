import os

from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

# LANGSMITH_TRACING=true
# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# LANGSMITH_API_KEY="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"
# LANGSMITH_PROJECT="pr-sparkling-cappelletti-63"
# OPENAI_API_KEY="<your-openai-api-key>"

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"]="ed-langchain"

llm = ChatDeepSeek(model ="deepseek-chat")

messages = [
    ("system", "你是一个智能助手，对载人工具领域非常了解。"),
    ("human", "你好，请罗列人类文明至今发明的载具")
]

result = llm.invoke(messages)

# print(result)

parser = StrOutputParser()
parsed_str = parser.invoke(result)
# print(parsed_str)

# create chain
chain = llm | parser

print(chain.invoke(messages))