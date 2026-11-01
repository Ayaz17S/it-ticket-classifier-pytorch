"""Train the ticket classifier and compare it with a classical baseline.

Run:  python -m app.train
Outputs (in models/): model.pt, vocab.json, labels.json, metrics.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

from app.model import TicketClassifier, Vocab, collate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "tickets.csv"
HANDWRITTEN = ROOT / "data" / "handwritten_tickets.csv"
MODEL_DIR = ROOT / "models"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LR = 5e-3
PATIENCE = 4  # early stopping: stop if val loss does not improve for 4 epochs


def batches(texts, labels, vocab, shuffle, rng):
    order = rng.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        ids, offsets = collate([vocab.encode(texts[i]) for i in idx])
        yield ids, offsets, torch.tensor(labels[idx])


@torch.no_grad()
def predict(model, texts, vocab) -> np.ndarray:
    model.eval()
    ids, offsets = collate([vocab.encode(t) for t in texts])
    return model(ids, offsets).argmax(1).numpy()


def evaluate_loss(model, texts, labels, vocab, loss_fn) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ids, offsets, y in batches(texts, labels, vocab, False, None):
            total += loss_fn(model(ids, offsets), y).item() * len(y)
    return total / len(texts)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(DATA)
    labels = sorted(df.category.unique())
    label_to_id = {c: i for i, c in enumerate(labels)}
    X, y = df.text.to_numpy(), df.category.map(label_to_id).to_numpy()

    # 70% train / 15% validation (early stopping) / 15% test (final report)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    vocab = Vocab.build(list(X_train))  # vocabulary from training data only
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)} vocab={len(vocab)}")

    model = TicketClassifier(len(vocab), len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for ids, offsets, yb in batches(X_train, y_train, vocab, True, rng):
            optimizer.zero_grad()
            loss = loss_fn(model(ids, offsets), yb)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, X_val, y_val, vocab, loss_fn)
        val_acc = accuracy_score(y_val, predict(model, X_val, vocab))
        print(f"epoch {epoch:2d}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)

    test_pred = predict(model, X_test, vocab)
    print(classification_report(y_test, test_pred, target_names=labels))

    # Baseline: always compare a neural model against a simple classical one.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    baseline = LogisticRegression(max_iter=2000).fit(tfidf.fit_transform(X_train), y_train)
    base_pred = baseline.predict(tfidf.transform(X_test))

    metrics = {
        "labels": labels,
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_macro_f1": f1_score(y_test, test_pred, average="macro"),
        "baseline_tfidf_logreg_accuracy": accuracy_score(y_test, base_pred),
        "train_seconds": round(train_seconds, 2),
        "vocab_size": len(vocab),
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    # Out-of-distribution check: tickets written by hand, not by the generator.
    if HANDWRITTEN.exists():
        hw = pd.read_csv(HANDWRITTEN)
        hw_true = hw.category.map(label_to_id).to_numpy()
        metrics["handwritten_accuracy"] = accuracy_score(
            hw_true, predict(model, hw.text.to_numpy(), vocab))
        metrics["handwritten_baseline_accuracy"] = accuracy_score(
            hw_true, baseline.predict(tfidf.transform(hw.text)))

    for k, v in metrics.items():
        if k != "labels":
            print(f"{k:32s} {v:.4f}" if isinstance(v, float) else f"{k:32s} {v}")

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    vocab.save(MODEL_DIR / "vocab.json")
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
