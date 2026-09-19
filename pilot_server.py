"""Customer pilot entry point. No personal Telegram webhooks or legacy public APIs."""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

os.environ["RIPPED_PILOT_MODE"] = "1"
os.environ.setdefault("REVIEW_HUB_CHAT_ID", "pilot")
os.environ.setdefault("REVIEW_HUB_USER_ID", "operator")

from review_hub import router
from customer_portal import router as customer_router

app = FastAPI(title="Ripped Shorts customer pilot", docs_url=None, redoc_url=None)
app.include_router(router)
app.include_router(customer_router)


@app.middleware("http")
async def private_responses(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/app"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def offer():
    page = (Path(__file__).parent / "review_static" / "landing.html").read_text(encoding="utf-8")
    return page.replace("{{CHECKOUT_URL}}", "/app")
