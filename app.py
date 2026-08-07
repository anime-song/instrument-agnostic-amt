import os
# CUDA アロケータを expandable segments にして、断片化による OOM を防ぐ（torch より先に設定）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import uuid
import sys
import torch
import shutil
import traceback
import librosa
import numpy as np
import asyncio
import yt_dlp as youtube_dl
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
from typing import Optional

OUTPUT_ROOT = "U:"  # @param {type:"string"}
WINDOW_BATCH_SIZE = 4  # @param {type:"integer"}
MAX_MIDI_MELODIC_INSTRUMENTS = 15  # @param {type:"integer"}
SKIP_DRUM_STEMS = True  # @param {type:"boolean"}
CLEANUP_SEPARATED_STEMS = True  # @param {type:"boolean"}
MERGE_ONSET_MS = 50.0  # @param {type:"number"}



def remove_extension(file: str):
    return ".".join(os.path.basename(file).split('.')[:-1])

# ---- 抽象层：Transcriber 接口 ----
class BaseTranscriber:
    """抽象的音频转录器接口"""
    async def transcribe(self, input_path: str, output_folder: str):
        """将音频转录为 MIDI 文件"""
        raise NotImplementedError("transcribe() must be implemented")

# ---- 具体实现：TranskunTranscriber ----
from infer_stem import run_stem_separated_transcription

class TranskunTranscriber(BaseTranscriber):

    def _transcribe_sync(self, input_path: str, output_folder: str) -> str:
        stem_pipeline_result = run_stem_separated_transcription(
            input_path,
            checkpoint_path=None,
            output_root=OUTPUT_ROOT,
            window_batch_size=WINDOW_BATCH_SIZE,
            max_midi_melodic_instruments=MAX_MIDI_MELODIC_INSTRUMENTS,
            cleanup_separated_stems=CLEANUP_SEPARATED_STEMS,
            merge_onset_ms=MERGE_ONSET_MS,
            predict_beat_chord=True,
            transcribe_lyrics=True,
            low_vram_mode=True
        )
        merged_midi_path = Path(stem_pipeline_result["merged_midi_path"])

        # Move the merged MIDI file to the output folder
        new_midi_path = Path(output_folder) / merged_midi_path.name
        shutil.move(merged_midi_path, new_midi_path)
        return str(new_midi_path)

    async def transcribe(self, input_path: str, output_folder: str) -> str:
        """异步包装同步推理"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, input_path, output_folder)


# ============ 以下为主应用逻辑 ============

# 初始化当前使用的 transcriber，可替换成别的实现
transcriber: BaseTranscriber = TranskunTranscriber()
event_queue = asyncio.Queue() # 广播队列，用于推送状态更新

async def broadcast_status():
    """向所有SSE客户端广播最新文件列表"""
    await event_queue.put(json.dumps(UPLOADED_FILES, ensure_ascii=False))

# 线程池
executor = ThreadPoolExecutor(max_workers=4)

# 清理目录
shutil.rmtree('uploads', ignore_errors=True)
shutil.rmtree('outputs', ignore_errors=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

app = FastAPI(title="Transkun FastAPI Service")
templates = Jinja2Templates(directory="templates")

UPLOADED_FILES = []
running = False

# ---------- 数据模型 ----------
class Message(BaseModel):
    message: str

class UploadedFile(BaseModel):
    id: str
    file: str
    status: str

# ---------- 页面 ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

# ---------- 上传 ----------
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> Message:
    for file in files:
        name = file.filename
        dest = os.path.join("uploads", name)
        async with aiofiles.open(dest, 'wb') as f:
            await f.write(await file.read())
        UPLOADED_FILES.append({
            "id": str(uuid.uuid4()),
            "file": name,
            "status": "pending"
        })
    await broadcast_status()
    return Message(message="Files uploaded")

# ---------- 文件列表 ----------
@app.get("/events")
async def sse_events():
    """Server-Sent Events 推送文件状态"""
    async def event_stream():
        # 先推送一次初始状态
        yield f"data: {json.dumps(UPLOADED_FILES, ensure_ascii=False)}\n\n"
        while True:
            data = await event_queue.get()
            yield f"data: {data}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- 下载 ----------
@app.get("/download/{file_id}")
async def download(file_id: str):
    """用 id 下载"""
    file = next((f for f in UPLOADED_FILES if f["id"] == file_id), None)
    if not file:
        return JSONResponse({"error": "File not found"}, status_code=404)
    midi_path = file.get("output") or os.path.join(
        "outputs", remove_extension(os.path.basename(file["file"])) + ".mid"
    )
    if not os.path.exists(midi_path):
        return JSONResponse({"error": "MIDI not ready"}, status_code=404)
    return FileResponse(
        midi_path,
        filename=os.path.basename(midi_path),
        media_type="audio/midi",
    )

# ---------- 转录 ----------
async def process_transcription():
    global running
    running = True
    for file in UPLOADED_FILES:
        if file["status"] == "pending":
            try:
                file["status"] = "processing"
                await broadcast_status()
                file["output"] = await transcriber.transcribe(
                    os.path.join("uploads", file["file"]), "outputs"
                )
                file["status"] = "transcribed"
                await broadcast_status()
            except Exception:
                traceback.print_exc()
                file["status"] = "error"
                await broadcast_status()
    running = False

@app.post("/transcribe")
async def transcribe(background_tasks: BackgroundTasks) -> Message:
    global running
    if running:
        return Message(message="already running")
    background_tasks.add_task(process_transcription)
    return Message(message="Transcription started")

# ---------- YouTube 下载 ----------
def my_hook(d):
    if d['status'] == 'finished':
        UPLOADED_FILES.append({
            'id': str(uuid.uuid4()),
            'file': os.path.splitext(os.path.basename(d['filename']))[0] + '.ogg',
            'status': 'pending'
        })

        
async def download_youtube_audio(url: str):
    ydl_opts = {
        'format': 'bestaudio/best',
        'progress_hooks': [my_hook],
        'outtmpl': 'uploads/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'vorbis',
            'preferredquality': '192'
        }]
    }
    await asyncio.get_event_loop().run_in_executor(executor, lambda: youtube_dl.YoutubeDL(ydl_opts).download([url]))

@app.get("/download-youtube")
async def download_youtube(url: str = Query(..., description="YouTube 视频链接")) -> Message:
    try:
        await download_youtube_audio(url)
        await broadcast_status()
        return Message(message="File downloaded and added to list")
    except Exception as e:
        traceback.print_exc()
        return Message(message=str(e))

# ---------- 启动 ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
 
