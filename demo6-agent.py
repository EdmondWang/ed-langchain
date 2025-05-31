import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import chat_agent_executor

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"]="ed-langchain"

llm = ChatDeepSeek(model ="deepseek-chat")

# LangChain built in search engine called Tavily
search = TavilySearchResults(max_results=2)
# Bind Search tool into LLM
tools = [search]
# model_with_tools = llm.bind_tools(tools)

# when no agent
# resp = llm.invoke([('human', '上海明天天气怎么样')])
# print(resp)

# LLM itself able to decide whether call tool to complete answer
# resp = model_with_tools.invoke([('human', '你好，我是艾德蒙')])
# print(f'Model_Result_Content: {resp.content}')
# resp = model_with_tools.invoke([('human', '上海明天天气怎么样')])
# print(f'Tools_Result_Content: {resp.tool_calls}')

# Create Agent
agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools)

resp = agent_executor.invoke({'messages': [HumanMessage(content='你好，我是艾德蒙')]})
print(resp['messages'])

resp = agent_executor.invoke({'messages': [HumanMessage(content='上海明天天气怎么样')]})
print(resp['messages'][2].content)