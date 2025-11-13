# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import routes_target, routes_tracking, video_tracking

app = FastAPI(title="Drone Object Tracking Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_target.router)
app.include_router(routes_tracking.router)
app.include_router(video_tracking.router)

@app.get("/")
async def root():
    return {"message": "Drone tracking backend is live"}
