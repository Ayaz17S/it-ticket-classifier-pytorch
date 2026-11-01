"""Text preprocessing and the PyTorch model.

Architecture (similar to Facebook's fastText):

    "vpn keeps dropping" -> tokens + bigrams -> ids [12, 87, 301, 4410, 5012]
        -> nn.EmbeddingBag (looks up a 64-d vector per id and averages them)
        -> Dropout -> Linear(64->64) -> ReLU -> Linear(64->6 classes)

Why this and not a Transformer? Tickets are short, the data set is small,
and we need low latency on CPU. A bag of embeddings trains in seconds,
infers in well under a millisecond and is easy to explain. A fine-tuned
BERT would be the next step if accuracy on real tickets was not enough.
"""
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn

PAD, UNK = "<pad>", "<unk>"


def char_ngrams(word: str, n: int = 3) -> list[str]:
    # "<mail>" -> "<ma", "mai", "ail", "il>": lets "emails", "mailbox" and the
    # typo "outlok" share features with words seen in training.
    w = f"<{word}>"
    return [f"#{w[i:i + n]}" for i in range(len(w) - n + 1)]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Word bigrams capture short phrases: "not working", "access denied".
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    chars = [g for w in words if len(w) > 2 for g in char_ngrams(w)]
    return words + bigrams + chars


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2) -> "Vocab":
        # Rare tokens (seen once, often typos) map to <unk> to reduce overfitting.
        counts = Counter(t for text in texts for t in tokenize(text))
        return cls([PAD, UNK] + sorted(t for t, c in counts.items() if c >= min_freq))

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in tokenize(text)]
        return ids or [1]  # empty text -> a single <unk>

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text()))


class TicketClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, emb_dim: int = 64,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # EmbeddingBag takes all token ids of the batch concatenated in one 1-D
        # tensor plus the start offset of each ticket: no padding needed.
        return self.head(self.embedding(ids, offsets))


def collate(encoded: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.tensor([0] + [len(e) for e in encoded[:-1]]).cumsum(0)
    ids = torch.tensor([i for e in encoded for i in e])
    return ids, offsets
