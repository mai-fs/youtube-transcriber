from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
from pytube import YouTube
import os
import uuid
import time # time.sleep() を使うためにインポート
from urllib.error import HTTPError # YouTubeダウンロード時のHTTPエラーをキャッチするためにインポート

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
    max_retries_download = 3 # YouTubeダウンロードの最大リトライ回数
    initial_retry_delay = 10  # 最初のリトライ待機時間 (秒)
    
    try:
        # --- YouTube音声ダウンロード処理（リトライ付き） ---
        download_successful = False
        current_retry_delay = initial_retry_delay
        for attempt in range(max_retries_download):
            try:
                print(f"Attempt {attempt + 1}/{max_retries_download} to download audio from: {request.youtube_url}")
                yt = YouTube(request.youtube_url)
                audio_stream = yt.streams.filter(only_audio=True).first()
                if not audio_stream:
                    # この場合はリトライしても無駄なので、即座にエラーとする
                    print(f"Audio stream not found for {request.youtube_url}")
                    raise HTTPException(status_code=400, detail="Audio stream not found.")
                
                audio_stream.download(filename=temp_filename)
                print(f"Audio downloaded successfully to: {temp_filename}")
                download_successful = True
                break  # ダウンロード成功したらループを抜ける
            
            except HTTPError as e:
                if e.code == 429: # Too Many Requests
                    if attempt < max_retries_download - 1:
                        print(f"HTTP 429 Error from YouTube. Retrying in {current_retry_delay} seconds...")
                        time.sleep(current_retry_delay)
                        current_retry_delay *= 2 # 次のリトライまでの時間を倍にする (エクスポネンシャルバックオフ)
                        continue
                    else:
                        print("Max retries reached for YouTube download due to HTTP 429 error.")
                        raise HTTPException(status_code=503, detail=f"Failed to download audio from YouTube after multiple retries (429 Error): {e}")
                else:
                    # 429以外のHTTPErrorはリトライせずにエラーとする
                    print(f"HTTPError (not 429) during YouTube download: {e}")
                    raise HTTPException(status_code=500, detail=f"HTTPError during YouTube download: {e}")
            
            except Exception as e:
                # pytubeのその他の予期せぬエラー (ネットワークエラーなども含む可能性)
                # これもリトライ対象とするか検討の余地ありだが、一旦リトライ対象外とする
                print(f"An unexpected error occurred during YouTube download attempt {attempt + 1}: {e}")
                if attempt < max_retries_download - 1:
                    print(f"Unexpected error, but retrying in {current_retry_delay} seconds...")
                    time.sleep(current_retry_delay)
                    current_retry_delay *=2
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to download audio due to unexpected error after retries: {str(e)}")

        if not download_successful:
            # このケースには通常到達しないはずだが、念のため
            raise HTTPException(status_code=500, detail="Audio download failed after all retries without specific error.")
        # --- YouTube音声ダウンロード処理ここまで ---

        print("Transcribing audio...")
        result = whisper_model.transcribe(temp_filename, fp16=False) # fp16=False はCPUでの実行を想定
        transcript_text = result["text"]
        print("Transcription complete.")

        return {"transcript": transcript_text}

    except HTTPException as e:
        # 既にHTTPExceptionとして発生したエラーはそのまま再raiseする
        # (ダウンロード失敗時の400, 503, 500エラーなど)
        print(f"HTTPException caught: {e.detail}")
        raise e # ここで再raiseしないと、クライアントに正しいエラーが返らない
    
    except Exception as e:
        # その他の予期せぬエラー (Whisperの処理中など)
        print(f"An unexpected error occurred in transcribe_video: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"Temporary file {temp_filename} removed.")