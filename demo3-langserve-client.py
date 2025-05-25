import os
from langserve import RemoteRunnable

# os.environ["HTTP_PROXY"] = "http://localhost:7890"
# os.environ["HTTPS_PROXY"] = "http://localhost:7890"
# need close vpn to run

if __name__ == "__main__":
    client = RemoteRunnable("http://localhost:8000/chain")
    print(client.invoke({
        "domain": "前端",
        "text": "介绍浏览器原理"
    }))