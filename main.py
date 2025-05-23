from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
from pytube import YouTube
import os
import uuid

app = FastAPI()

# Whisperモデルをロード（一番小さい"tiny"モデルを使います。これなら無料プランでも動きやすい）
print("Loading Whisper model 'tiny'...")
whisper_model = whisper.load_model("tiny")
print("Whisper model loaded.")

class VideoRequest(BaseModel):
    youtube_url: str

@app.post("/transcribe")
async def transcribe_video(request: VideoRequest):
    temp_filename = f"temp_audio_{uuid.uuid4()}.mp4"
    
    try:
        print(f"Downloading audio from: {request.youtube_url}")
        yt = YouTube(request.youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            raise HTTPException(status_code=400, detail="Audio stream not found.")
        audio_stream.download(filename=temp_filename)
        print(f"Audio downloaded to: {temp_filename}")

        print("Transcribing audio...")
        result = whisper_model.transcribe(temp_filename, fp16=False)
        transcript_text = result["text"]
        print("Transcription complete.")

        return {"transcript": transcript_text}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"Temporary file {temp_filename} removed.")