import os
import ssl
import datetime

from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "ed-langchain"

llm = ChatDeepSeek(model="deepseek-chat")

persist_dir = 'demo10_chroma_data_dir'

urls = [
    # 'https://www.youtube.com/watch?v=DjuXACWYkkU',
    'https://www.youtube.com/watch?v=pgvEnDkNUfg',
    'https://www.youtube.com/watch?v=z4eT6Lxgc6M',
    'https://www.youtube.com/watch?v=HHElDBDlOKM'
]

docs = []

for url in urls:
    # 获取元数据
    meta_loader = YoutubeLoaderDL.from_youtube_url(youtube_url=url, add_video_info=True)
    meta_docs = meta_loader.load()
    # 获取字幕
    video_id = meta_loader.extract_video_id(youtube_url=url)
    # transcript_docs = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=False).load()
    transcriptStrList = []
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch('pgvEnDkNUfg')
    for snippet in fetched_transcript:
        transcriptStrList.append(snippet.text)

    # 合并元数据和字幕
    if meta_docs:
        # 只取第一个（通常每个loader只返回一个Document）
        meta = meta_docs[0].metadata
        transcript = transcriptStrList

        # 提取年份并新增到 metadata
        publish_date = meta.get("publish_date")
        if publish_date:
            try:
                dt = datetime.datetime.fromisoformat(publish_date)
                meta["publish_year"] = dt.year
            except Exception:
                meta["publish_year"] = None
        else:
            meta["publish_year"] = None

        # 合成一个新的Document
        from langchain_core.documents import Document
        doc = Document(page_content="\n".join(transcriptStrList), metadata=meta)
        docs.append(doc)

print(len(docs))
for doc in docs:
    print('page content metadata: \n', doc.metadata)
    print(f'page content length: \n {len(doc.page_content)}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
split_doc = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')

vector_store = Chroma.from_documents(documents=split_doc, embedding=embeddings, persist_directory=persist_dir)


