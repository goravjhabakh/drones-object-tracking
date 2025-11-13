import torch
import cv2
import numpy as np
from PIL import Image

from services.embedding_service import encode_images
from services.detection_service import detect_objects
from store.target_store import get_target_embedding
from core.config import DEVICE

SIM_THRESHOLD = 0.80
CLIP_BATCH = 32


def track_object(frame):

    # Load target embedding
    target_emb = get_target_embedding()
    if target_emb is None:
        return {"error": "No target uploaded"}

    if isinstance(target_emb, np.ndarray):
        target_emb = torch.tensor(target_emb, dtype=torch.float32, device=DEVICE)

    target_emb = torch.nn.functional.normalize(target_emb.unsqueeze(0), dim=1)

    # YOLO detection
    results = detect_objects(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []

    # ❗ If no YOLO detections → DO NOTHING
    if len(boxes) == 0:
        return {
            "cx": None,
            "cy": None,
            "dx": 0.0,
            "dy": 0.0,
            "confidence": 0.0,
            "target_found": False
        }

    # Crop all detections for CLIP
    crops = []
    bboxes = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crops.append(pil)
        bboxes.append([x1, y1, x2, y2])

    # ❗ If YOLO found boxes, but all crops invalid
    if len(crops) == 0:
        return {
            "cx": None,
            "cy": None,
            "dx": 0.0,
            "dy": 0.0,
            "confidence": 0.0,
            "target_found": False
        }

    # Batch CLIP embeddings
    crop_emb = encode_images(crops, batch_size=CLIP_BATCH)
    sims = torch.matmul(crop_emb, target_emb.T).squeeze(1).cpu().numpy()

    # Pick best similarity
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    # ❗ Strict threshold check (same as video pipeline)
    if best_sim < SIM_THRESHOLD:
        return {
            "cx": None,
            "cy": None,
            "dx": 0.0,
            "dy": 0.0,
            "confidence": float(best_sim),
            "target_found": False
        }

    # Extract best box
    x1, y1, x2, y2 = bboxes[best_idx]

    # Compute center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Save
    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame_with_box, f"sim={best_sim:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite('app/temp/frame.png', frame_with_box)

    # Normalize offsets for drone control
    h, w = frame.shape[:2]
    dx = (cx - w / 2) / (w / 2)
    dy = (cy - h / 2) / (h / 2)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "dx": float(dx),
        "dy": float(dy),
        "confidence": float(best_sim),
        "target_found": True
    }