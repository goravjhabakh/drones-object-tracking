# app/main.py
from fastapi import FastAPI
from api import routes_target, routes_tracking, video_tracking

app = FastAPI(title="Drone Object Tracking Backend")

app.include_router(routes_target.router)
app.include_router(routes_tracking.router)
app.include_router(video_tracking.router)

@app.get("/")
async def root():
    return {"message": "Drone tracking backend is live"}
