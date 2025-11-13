import cv2
import numpy as np
from PIL import Image
import uuid
import os
import torch
from ultralytics import YOLO
from tqdm.auto import tqdm

from services.embedding_service import encode_images
from store.target_store import get_target_embedding
from core.config import YOLO_MODEL_PATH, DEVICE

# Load YOLO once
yolo = YOLO(YOLO_MODEL_PATH)

SIM_THRESHOLD = 0.80   # strict threshold
CLIP_BATCH = 32


def process_video_simple(video_path: str):
    target_emb = get_target_embedding()
    if target_emb is None:
        return None, "No target uploaded"

    # Normalize target embedding
    if isinstance(target_emb, np.ndarray):
        target_emb = torch.tensor(target_emb, dtype=torch.float32, device=DEVICE)

    target_emb = torch.nn.functional.normalize(target_emb.unsqueeze(0), dim=1)

    # Read input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    output_name = f"processed_{uuid.uuid4().hex}.mp4"
    os.makedirs("app/temp", exist_ok=True)
    output_path = os.path.join("app/temp", output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame", dynamic_ncols=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = yolo(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else []

        crops = []
        bboxes = []

        # Create crops for CLIP
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append(pil_crop)
            bboxes.append([x1, y1, x2, y2])

        # No detections → save frame and continue
        if len(crops) == 0:
            out.write(frame)
            pbar.update(1)
            continue

        # Encode all crops using CLIP
        crop_emb = encode_images(crops, batch_size=CLIP_BATCH)
        sims = torch.matmul(crop_emb, target_emb.T).squeeze(1).cpu().numpy()

        # Best similarity
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        # Only proceed if best_sim >= 0.80
        if best_sim < SIM_THRESHOLD:
            out.write(frame)
            pbar.update(1)
            continue

        # Otherwise, draw the bounding box
        chosen_box = bboxes[best_idx]
        chosen_sim = best_sim

        x1, y1, x2, y2 = chosen_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"sim={chosen_sim:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    return output_path, None