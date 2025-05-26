import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_deepseek import ChatDeepSeek

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['LANGSMITH_PROJECT']='ed-langchain'
os.environ['DEEPSEEK_API_KEY']='sk-b99a159e8e3046ea9482d13d53e6e23a'

# chatbot sample
llm = ChatDeepSeek(model ='deepseek-chat')

prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个非常乐于助人的智能助手，对{domain}载人工具领域非常了解。'),
    MessagesPlaceholder(variable_name='my_msg')
])

# create chain
chain = prompt_template | llm

# keyed by session id, valued by session chat history
store = {}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg' # the key of sending msg each time
)

config: RunnableConfig = {
    'configurable': {
        'session_id': 'zs123'
    }
}

# first round
resp = do_message.invoke({
    'my_msg': [HumanMessage(content='你好啊！我的名字叫艾德蒙')], # ('human', '你好啊！我的名字叫艾德蒙')
    'domain': '海洋',
}, config=config)

print('### respond to #1')
print(resp.content)

# second round
resp = do_message.invoke({
    'my_msg': [HumanMessage(content='请问我叫什么名字？')], # [('human', '请问我叫什么名字？')],
    'domain': '海洋'
}, config=config)

print('### respond to #2')
print(resp.content)

# third round, return data in stream
print('### respond to #3')
for resp in do_message.stream({
    'my_msg': [HumanMessage(content='给我讲一个该领域相关的行业笑话')], # [('human', '请问我叫什么名字？')],
    'domain': '海洋'
}, config=config):
# one token each time resp get looped
    print(resp.content, end='-')

