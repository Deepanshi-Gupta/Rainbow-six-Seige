import asyncio
import json
import logging
import aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import r6s Task
from tasks import run_analysis_task
from celery.result import AsyncResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://playformance.gg", "https://api.playformance.gg"], # Configure for your r6s frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    video_url: str

@app.get("/health")
def health_check():
    return {"status": "ok", "game": "Rainbow Six Siege"}

# --- r6s Analysis Endpoint ---
@app.post("/analyze")
async def analyze_endpoint(data: VideoRequest):
    video_url = data.video_url
    if not video_url:
        return JSONResponse(content={"error": "Missing video_url"}, status_code=400)

    logger.info(f"Received r6s analysis request for {video_url}")
    
    # Call the r6s Celery Task
    task = run_analysis_task.delay(video_url)

    status_url = f"https://ai.playformance.gg/r6s/status/{task.id}"

    return JSONResponse(content={
        "status": "processing",
        "job_id": task.id,
        "status_url": status_url
    }, status_code=202)

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task_result = AsyncResult(task_id, app=run_analysis_task.app)
    response = {'state': task_result.state}

    if task_result.state == 'SUCCESS':
        response['result'] = task_result.get()
    elif task_result.state == 'FAILURE':
        response['status'] = str(task_result.info)
    else:
        response['status'] = "Task is processing or in queue."
    return JSONResponse(content=response)

# --- WebSocket for r6s ---
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    
    # Note: Make sure redis_utils publishes to THIS channel structure
    # If you use the generic redis_utils, it might default to job_updates:valorant:{id}
    # You should modify redis_utils.py to accept the game prefix or standardize it.
    # Assuming we standardize on `job_updates:valorant:{id}` for now inside the task:
    
    # Recommendation: Change redis_utils.py line 24 to:
    # channel = f"job_updates:{job_id}" (remove 'valorant')
    # Then this works for both games.
    
    redis_conn = None
    pubsub = None
    try:
        redis_conn = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
        pubsub = redis_conn.pubsub()
        
        # IMPORTANT: This must match what r6s_tasks.py publishes to!
        channel = f"job_updates:r6s:{job_id}" # Keeping strictly to your existing format
        
        await pubsub.subscribe(channel)
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                await websocket.send_json(data)
                if data.get("status") in ["complete", "error"]:
                    break
    except Exception as e:
        logger.error(f"WS Error: {e}")
    finally:
        if pubsub: await pubsub.close()
        if redis_conn: await redis_conn.close()
        