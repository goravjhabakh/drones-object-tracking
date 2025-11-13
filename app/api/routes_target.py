from fastapi import APIRouter, UploadFile, File
from PIL import Image
from services.embedding_service import get_image_embedding
from store.target_store import save_target_embedding

router = APIRouter(prefix="/target", tags=["Target"])

@router.post("/upload")
async def upload_target(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    emb = get_image_embedding(image)
    save_target_embedding(emb)
    return {"message": "Target uploaded successfully"}