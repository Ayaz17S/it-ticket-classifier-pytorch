"""REST API for ticket classification.

Run:  uvicorn app.api:app --reload    ->  http://127.0.0.1:8000/docs
"""
import json
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.predictor import MODEL_DIR, TicketPredictor

app = FastAPI(title="IT Ticket Classifier", version="1.0.0")
predictor = TicketPredictor()

# Which team receives each category.
ROUTING = {
    "network": "Network Operations",
    "hardware": "Desktop Support",
    "access": "Identity & Access Management",
    "password": "Service Desk L1",
    "email": "Messaging Team",
    "software": "Application Support",
}


class Ticket(BaseModel):
    text: str = Field(min_length=5, max_length=2000)


class TicketBatch(BaseModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=500)


def route(result: dict) -> dict:
    team = "Service Desk Triage (human review)" if result["needs_human_review"] \
        else ROUTING[result["category"]]
    return {**result, "assigned_team": team}


@app.get("/health")
def health():
    return {"status": "ok", "classes": predictor.labels}


@app.get("/model/info")
def model_info():
    return json.loads((MODEL_DIR / "metrics.json").read_text())


@app.post("/classify")
def classify(ticket: Ticket):
    return route(predictor.classify([ticket.text])[0])


@app.post("/classify/batch")
def classify_batch(batch: TicketBatch):
    start = time.perf_counter()
    results = [route(r) for r in predictor.classify([t.text for t in batch.tickets])]
    return {"results": results, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}
