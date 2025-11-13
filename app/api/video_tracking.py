from fastapi import APIRouter, UploadFile, File
import uuid, os
from fastapi.responses import FileResponse

from services.video_service import process_video_simple

router = APIRouter(prefix="/video-simple", tags=["Simple Video Tracking"])


@router.post("/process")
async def process_video(file: UploadFile = File(...)):
    os.makedirs("app/temp", exist_ok=True)
    save_path = f"app/temp/{uuid.uuid4().hex}_{file.filename}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    output_path, err = process_video_simple(save_path)
    if err:
        return {"error": err}

    return {"video_url": f"/video-simple/get/{os.path.basename(output_path)}"}


@router.get("/get/{filename}")
async def get_processed_video(filename: str):
    path = f"app/temp/{filename}"
    return FileResponse(path, media_type="video/mp4")