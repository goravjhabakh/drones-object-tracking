# app/api/routes_tracking.py
from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from services.tracking_service import track_object

router = APIRouter(prefix="/track", tags=["Tracking"])

@router.post("/frame")
async def process_frame(file: UploadFile = File(...)):
    frame_bytes = await file.read()
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result = track_object(frame)
    return result