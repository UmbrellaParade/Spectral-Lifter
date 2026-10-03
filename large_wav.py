import math
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from processor import AudioProcessor


MAX_FILE_SIZE = 300 * 1024 * 1024
MAX_CHUNK_SIZE = 9 * 1024 * 1024
JOB_LIFETIME_SECONDS = 2 * 60 * 60
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "spectral_lifter_large_wav"

JOBS = {}
JOBS_LOCK = threading.Lock()


class UploadStart(BaseModel):
    name: str
    size: int
    chunks: int


def _safe_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="アップロードが見つかりません。")
        return job


def _cleanup_old_jobs():
    cutoff = time.time() - JOB_LIFETIME_SECONDS
    with JOBS_LOCK:
        expired = [job_id for job_id, job in JOBS.items() if job["created_at"] < cutoff]
        for job_id in expired:
            job = JOBS.pop(job_id)
            shutil.rmtree(job["directory"], ignore_errors=True)


def _run_processing(job_id):
    try:
        job = _safe_job(job_id)

        def report(value, description):
            with JOBS_LOCK:
                current = JOBS.get(job_id)
                if current:
                    current["status"] = "processing"
                    current["progress"] = 50 + round(value * 49)
                    current["message"] = description

        output_path = AudioProcessor().process(str(job["input_path"]), progress_callback=report)
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if current:
                current["status"] = "complete"
                current["progress"] = 100
                current["message"] = "処理が完了しました。WAVをダウンロードできます。"
                current["output_path"] = Path(output_path)
    except Exception as exc:
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if current:
                current["status"] = "error"
                current["message"] = f"処理に失敗しました: {exc}"


def create_app(demo):
    app = FastAPI(title="Spectral Lifter")

    @app.get("/large-wav", response_class=HTMLResponse)
    async def large_wav_page():
        return HTMLResponse(LARGE_WAV_HTML)

    @app.post("/api/large-wav/start")
    async def start_upload(payload: UploadStart):
        _cleanup_old_jobs()
        filename = Path(payload.name).name
        if Path(filename).suffix.lower() != ".wav":
            raise HTTPException(status_code=400, detail="WAVファイルを選択してください。")
        if payload.size <= 0 or payload.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="ファイルは300MB以内にしてください。")
        expected_chunks = math.ceil(payload.size / MAX_CHUNK_SIZE)
        if payload.chunks != expected_chunks:
            raise HTTPException(status_code=400, detail="分割数が正しくありません。")

        job_id = uuid.uuid4().hex
        directory = UPLOAD_ROOT / job_id
        directory.mkdir(parents=True, exist_ok=False)
        input_path = directory / filename
        input_path.touch()
        with JOBS_LOCK:
            JOBS[job_id] = {
                "created_at": time.time(),
                "directory": directory,
                "input_path": input_path,
                "original_name": filename,
                "expected_size": payload.size,
                "expected_chunks": payload.chunks,
                "next_chunk": 0,
                "received_bytes": 0,
                "status": "uploading",
                "progress": 0,
                "message": "アップロードを開始しました。",
                "output_path": None,
            }
        return {"job_id": job_id, "chunk_size": MAX_CHUNK_SIZE}

    @app.post("/api/large-wav/{job_id}/chunk/{chunk_index}")
    async def upload_chunk(job_id: str, chunk_index: int, request: Request):
        data = await request.body()
        if not data or len(data) > MAX_CHUNK_SIZE:
            raise HTTPException(status_code=400, detail="分割データのサイズが正しくありません。")

        job = _safe_job(job_id)
        with JOBS_LOCK:
            if job["status"] != "uploading" or chunk_index != job["next_chunk"]:
                raise HTTPException(status_code=409, detail="分割データの順番が正しくありません。")
            if job["received_bytes"] + len(data) > job["expected_size"]:
                raise HTTPException(status_code=400, detail="ファイルサイズが申告値を超えました。")
            with job["input_path"].open("ab") as output:
                output.write(data)
            job["received_bytes"] += len(data)
            job["next_chunk"] += 1
            job["progress"] = round(50 * job["received_bytes"] / job["expected_size"])
            job["message"] = f"アップロード中 ({job['next_chunk']}/{job['expected_chunks']})"
            progress = job["progress"]
        return {"progress": progress}

    @app.post("/api/large-wav/{job_id}/process")
    async def process_upload(job_id: str):
        job = _safe_job(job_id)
        with JOBS_LOCK:
            if (
                job["status"] != "uploading"
                or job["received_bytes"] != job["expected_size"]
                or job["next_chunk"] != job["expected_chunks"]
            ):
                raise HTTPException(status_code=409, detail="アップロードが完了していません。")
            job["status"] = "queued"
            job["progress"] = 50
            job["message"] = "処理を開始します。"
        threading.Thread(target=_run_processing, args=(job_id,), daemon=True).start()
        return {"status": "queued"}

    @app.get("/api/large-wav/{job_id}/status")
    async def job_status(job_id: str):
        job = _safe_job(job_id)
        with JOBS_LOCK:
            result = {
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"],
            }
            if job["status"] == "complete":
                result["download_url"] = f"/api/large-wav/{job_id}/download"
        return result

    @app.get("/api/large-wav/{job_id}/download")
    async def download_result(job_id: str):
        job = _safe_job(job_id)
        output_path = job.get("output_path")
        if job["status"] != "complete" or not output_path or not output_path.exists():
            raise HTTPException(status_code=404, detail="出力ファイルがありません。")
        filename = f"{Path(job['original_name']).stem}_lifter.wav"
        return FileResponse(output_path, media_type="audio/wav", filename=filename)

    return gr.mount_gradio_app(app, demo, path="/")


LARGE_WAV_HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>大容量WAV | Spectral Lifter</title>
  <style>
    body { margin: 0; background: #f7f7f8; color: #202124; font-family: system-ui, sans-serif; }
    main { max-width: 760px; margin: 48px auto; padding: 0 20px; }
    .card { background: white; border: 1px solid #ddd; border-radius: 16px; padding: 28px; box-shadow: 0 4px 18px #0000000d; }
    h1 { margin-top: 0; font-size: 28px; }
    input { width: 100%; box-sizing: border-box; padding: 16px; border: 1px dashed #999; border-radius: 10px; }
    button, .download { display: inline-block; margin-top: 18px; padding: 12px 20px; border: 0; border-radius: 9px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; text-decoration: none; }
    button:disabled { opacity: .5; cursor: wait; }
    progress { width: 100%; height: 18px; margin-top: 22px; }
    #status { min-height: 3em; white-space: pre-wrap; }
    .note { color: #555; line-height: 1.7; }
    .back { display: inline-block; margin-bottom: 18px; color: #2563eb; }
  </style>
</head>
<body>
<main>
  <a class="back" href="/">← 通常画面へ戻る</a>
  <div class="card">
    <h1>大容量WAVを処理</h1>
    <p class="note">6分以内・300MB以内のWAVを、小分けにアップロードして処理します。画面を閉じずにお待ちください。</p>
    <input id="file" type="file" accept="audio/wav,.wav">
    <button id="start">アップロードして処理</button>
    <progress id="progress" max="100" value="0"></progress>
    <p id="status">WAVファイルを選択してください。</p>
    <a id="download" class="download" hidden>処理済みWAVをダウンロード</a>
  </div>
</main>
<script>
const fileInput = document.getElementById('file');
const startButton = document.getElementById('start');
const progress = document.getElementById('progress');
const status = document.getElementById('status');
const download = document.getElementById('download');

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `通信エラー (${response.status})`);
  return body;
}

async function poll(jobId) {
  while (true) {
    const result = await jsonRequest(`/api/large-wav/${jobId}/status`);
    progress.value = result.progress;
    status.textContent = result.message;
    if (result.status === 'complete') {
      download.href = result.download_url;
      download.hidden = false;
      startButton.disabled = false;
      return;
    }
    if (result.status === 'error') throw new Error(result.message);
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
}

startButton.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return status.textContent = 'WAVファイルを選択してください。';
  if (!file.name.toLowerCase().endsWith('.wav')) return status.textContent = 'WAVファイルを選択してください。';
  startButton.disabled = true;
  download.hidden = true;
  progress.value = 0;
  try {
    const chunkSize = 9 * 1024 * 1024;
    const chunks = Math.ceil(file.size / chunkSize);
    const started = await jsonRequest('/api/large-wav/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: file.name, size: file.size, chunks})
    });
    for (let index = 0; index < chunks; index++) {
      status.textContent = `アップロード中 (${index + 1}/${chunks})`;
      const part = file.slice(index * chunkSize, Math.min((index + 1) * chunkSize, file.size));
      const response = await fetch(`/api/large-wav/${started.job_id}/chunk/${index}`, {
        method: 'POST', body: part, headers: {'Content-Type': 'application/octet-stream'}
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `アップロードエラー (${response.status})`);
      }
      progress.value = Math.round(50 * (index + 1) / chunks);
    }
    await jsonRequest(`/api/large-wav/${started.job_id}/process`, {method: 'POST'});
    await poll(started.job_id);
  } catch (error) {
    status.textContent = error.message;
    startButton.disabled = false;
  }
});
</script>
</body>
</html>"""
