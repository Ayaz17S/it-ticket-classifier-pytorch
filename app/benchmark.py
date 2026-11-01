"""Optimisation experiments for serving the model.

Run:  python -m app.benchmark

1. Compression: save the weights as float16 instead of float32 -> half the
   file size. Check that accuracy does not drop.
2. Batching: measure latency per ticket for different batch sizes. One forward
   pass over 64 tickets costs far less than 64 separate passes, which is why
   the API has a /classify/batch endpoint.
"""
import statistics
import time

import pandas as pd
import torch

from app.predictor import MODEL_DIR, TicketPredictor

DATA = MODEL_DIR.parent / "data" / "tickets.csv"
FP16_FILE = "model_fp16.pt"


def accuracy(p: TicketPredictor, df: pd.DataFrame) -> float:
    pred = [p.labels[i] for i in p.predict_proba(df.text.tolist()).argmax(1).tolist()]
    return float((pd.Series(pred, index=df.index) == df.category).mean())


def per_ticket_ms(p: TicketPredictor, texts: list[str], batch: int, runs: int = 50) -> float:
    times = []
    for r in range(runs):
        chunk = [texts[(r * batch + i) % len(texts)] for i in range(batch)]
        start = time.perf_counter()
        p.predict_proba(chunk)
        times.append((time.perf_counter() - start) * 1000 / batch)
    return statistics.median(times)


def main() -> None:
    torch.set_num_threads(1)  # typical for a small CPU inference container
    df = pd.read_csv(DATA).sample(1000, random_state=0)

    state = torch.load(MODEL_DIR / "model.pt", weights_only=True)
    torch.save({k: v.half() for k, v in state.items()}, MODEL_DIR / FP16_FILE)

    fp32, fp16 = TicketPredictor(), TicketPredictor(weights_file=FP16_FILE)
    size = lambda f: round((MODEL_DIR / f).stat().st_size / 1024, 1)  # noqa: E731
    print(pd.DataFrame([
        {"weights": "float32", "file_kb": size("model.pt"), "accuracy": accuracy(fp32, df)},
        {"weights": "float16", "file_kb": size(FP16_FILE), "accuracy": accuracy(fp16, df)},
    ]).to_string(index=False))

    print("\nLatency per ticket vs batch size (1 CPU thread):")
    texts = df.text.tolist()
    rows = [{"batch_size": b, "ms_per_ticket": round(per_ticket_ms(fp32, texts, b), 4)}
            for b in (1, 8, 32, 128)]
    for r in rows:
        r["speedup_vs_1"] = round(rows[0]["ms_per_ticket"] / r["ms_per_ticket"], 1)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
