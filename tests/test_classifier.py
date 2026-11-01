import torch
from fastapi.testclient import TestClient

from app.api import app
from app.model import TicketClassifier, Vocab, collate, tokenize
from app.predictor import TicketPredictor

client = TestClient(app)


def test_tokenize_adds_bigrams_and_char_ngrams():
    tokens = tokenize("VPN down")
    assert "vpn" in tokens and "vpn_down" in tokens and "#<vp" in tokens


def test_unknown_words_map_to_unk():
    vocab = Vocab.build(["hello world", "hello there"], min_freq=2)
    assert set(vocab.encode("zzzz")) == {1}  # word and its char-grams are all <unk>


def test_model_output_shape():
    ids, offsets = collate([[2, 3, 4], [5, 6]])
    logits = TicketClassifier(vocab_size=10, n_classes=6)(ids, offsets)
    assert logits.shape == (2, 6)


def test_probabilities_sum_to_one():
    probs = TicketPredictor().predict_proba(["vpn keeps disconnecting", "printer jammed"])
    assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-5)


def test_clear_tickets_are_classified_correctly():
    p = TicketPredictor()
    cases = {
        "forgot my password and my account is locked": "password",
        "cannot connect to vpn from home": "network",
        "outlook is not syncing emails": "email",
        "laptop screen is flickering": "hardware",
    }
    preds = [r["category"] for r in p.classify(list(cases))]
    assert preds == list(cases.values())


def test_float16_weights_give_same_predictions(tmp_path):
    import shutil
    from app.predictor import MODEL_DIR
    for f in ("vocab.json", "labels.json"):
        shutil.copy(MODEL_DIR / f, tmp_path / f)
    state = torch.load(MODEL_DIR / "model.pt", weights_only=True)
    torch.save({k: v.half() for k, v in state.items()}, tmp_path / "model.pt")

    texts = ["need access to sharepoint", "jira crashes on export", "wifi is slow"]
    full = TicketPredictor().predict_proba(texts).argmax(1)
    half = TicketPredictor(model_dir=tmp_path).predict_proba(texts).argmax(1)
    assert torch.equal(full, half)


def test_api_routes_ticket_to_team():
    body = client.post("/classify", json={"text": "my vpn keeps dropping from home"}).json()
    assert body["category"] == "network"
    assert body["assigned_team"] == "Network Operations"


def test_api_rejects_empty_ticket():
    assert client.post("/classify", json={"text": ""}).status_code == 422
