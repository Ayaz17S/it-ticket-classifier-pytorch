"""Load the trained model and classify tickets (used by the API and tests)."""
import json
from pathlib import Path

import torch

from app.model import TicketClassifier, Vocab, collate

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Below this confidence the ticket goes to a human agent instead of being
# auto-routed: a wrong automatic route costs more than a short manual check.
REVIEW_THRESHOLD = 0.6


class TicketPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR, weights_file: str = "model.pt"):
        self.vocab = Vocab.load(model_dir / "vocab.json")
        self.labels = json.loads((model_dir / "labels.json").read_text())
        state = torch.load(model_dir / weights_file, weights_only=True)
        # A float16 checkpoint (see benchmark.py) is upcast to float32 here,
        # because float16 maths is slow on most CPUs.
        state = {k: v.float() for k, v in state.items()}
        model = TicketClassifier(len(self.vocab), len(self.labels))
        model.load_state_dict(state)
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> torch.Tensor:
        ids, offsets = collate([self.vocab.encode(t) for t in texts])
        return torch.softmax(self.model(ids, offsets), dim=1)

    def classify(self, texts: list[str], top_k: int = 3) -> list[dict]:
        results = []
        for probs in self.predict_proba(texts):
            conf, idx = probs.topk(top_k)
            best = float(conf[0])
            results.append({
                "category": self.labels[int(idx[0])],
                "confidence": round(best, 4),
                "needs_human_review": best < REVIEW_THRESHOLD,
                "top_k": [{"category": self.labels[int(i)], "confidence": round(float(c), 4)}
                          for c, i in zip(conf, idx)],
            })
        return results
