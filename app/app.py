import os
import sys
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, Path, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from figure_registry import registry
from step_config import STEP_CLASSES

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "dist")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AppState:
    def __init__(self):
        self.registry = registry
        self.verifiers = {}
        self.default_classes = list(STEP_CLASSES)

    def load(self):
        self.registry.load()
        print(f"[INFO] Loaded {len(self.registry)} origami figure specifications: {list(self.registry.keys())}")
        verifier = self.registry.get_verifier("crane", device=str(DEVICE))
        self.verifiers["crane"] = verifier

    def get_verifier(self, figure_id: str):
        return self.verifiers.get(figure_id)

    def get_classes(self, figure_id: str):
        verifier = self.get_verifier(figure_id)
        if verifier and hasattr(verifier, "classes"):
            return list(verifier.classes)
        return self.default_classes


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield
    state.verifiers.clear()


app = FastAPI(title="GamiSenpai API", lifespan=lifespan)


@app.get("/api/figures")
async def list_figures(reload: bool = Query(False, description="Force reload registry from disk")):
    if reload:
        state.registry.load(force_reload=True)
    return state.registry.list_summaries()


@app.get("/api/figures/{figure_id}")
async def get_figure_detail(
    figure_id: str = Path(..., pattern=r"^[a-z0-9_-]{1,64}$", description="Figure identifier"),
):
    fig = state.registry.get(figure_id)
    if not fig:
        raise HTTPException(status_code=404, detail=f"Origami figure '{figure_id}' not found.")
    return fig


@app.get("/api/model-info")
async def model_info(
    figure: str = Query("crane", pattern=r"^[a-z0-9_-]{1,64}$", description="Target origami figure ID"),
):
    target_verifier = state.get_verifier(figure)
    return {
        "task": "step-classification",
        "figure": figure,
        "classes": state.get_classes(figure),
        "model_loaded": target_verifier is not None,
        "available_figures": list(state.registry.keys()),
    }


@app.post("/api/predict")
async def predict_frame(
    file: UploadFile = File(...),
    figure: str = Query("crane", pattern=r"^[a-z0-9_-]{1,64}$", description="Target origami figure ID"),
):
    fig_config = state.registry.get(figure)
    if not fig_config:
        return JSONResponse(status_code=404, content={"error": f"Origami figure '{figure}' does not exist."})

    if fig_config.get("status") != "available":
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Model for '{figure}' is currently under development (Coming Soon).",
                "figure": figure,
                "status": fig_config.get("status"),
            },
        )

    if file.content_type and file.content_type.lower() not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported media type '{file.content_type}'. Allowed types: {sorted(list(ALLOWED_IMAGE_TYPES))}"},
        )

    try:
        contents = await file.read()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to read uploaded file: {str(e)}"})

    if not contents or len(contents) == 0:
        return JSONResponse(status_code=400, content={"error": "Uploaded image file is empty."})

    if len(contents) > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB."},
        )

    target_verifier = state.get_verifier(figure)
    if not target_verifier:
        return JSONResponse(status_code=503, content={"error": f"Model for '{figure}' is not loaded yet."})

    try:
        result = await run_in_threadpool(target_verifier.verify_bytes, contents)
        return result.to_dict()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image format: {str(e)}"})


if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    print(f"[WARNING] Static directory {STATIC_DIR} not found. Build frontend first!")
