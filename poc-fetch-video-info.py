import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


#  AI LLM related video
# url = "https://www.youtube.com/watch?v=DjuXACWYkkU"

# New Model Y in 2025
url = "https://www.youtube.com/watch?v=pgvEnDkNUfg"

ydl_opts = {}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    print('** title \n', info['title'])
    print('** description \n', info['description'])
    print('** upload_date \n', info['upload_date'])
    print('** duration \n', info['duration'])

print('** transcript \n')
transcriptStrList = []
ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch('pgvEnDkNUfg')
for snippet in fetched_transcript:
    transcriptStrList.append(snippet.text)
print(transcriptStrList)