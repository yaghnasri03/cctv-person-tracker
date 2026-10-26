from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.upload import router as upload_router

app = FastAPI(
    title="CCTV Person Tracker API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload API
app.include_router(upload_router)

# Serve processed videos
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

@app.get("/")
def root():
    return {
        "message": "CCTV Person Tracker Backend Running"
    }