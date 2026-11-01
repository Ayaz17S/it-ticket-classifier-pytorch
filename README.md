# IT Support Ticket Classifier (PyTorch)

**🚀 Live demo: [ayaz-ticket-classifier.streamlit.app](https://ayaz-ticket-classifier.streamlit.app)**

A deep-learning text classifier that reads an IT helpdesk ticket and routes it to the
right team (network, hardware, access, password, email, software). Unsure predictions go
to a human instead of being routed automatically.

**Stack:** Python, PyTorch, scikit-learn (baseline), pandas, FastAPI, pytest, Docker

## Problem
Large IT service desks (exactly the kind Tech Mahindra runs for clients) receive thousands
of tickets a day. Manually reading and assigning each one takes time and often lands
tickets in the wrong queue, which delays fixes.

## Data
Real tickets are confidential, so `generate_data.py` creates 3,600 realistic tickets from
templates with synonyms, filler words, typos, mixed case and 3% wrong labels.
To check whether the model works beyond the generator, `data/handwritten_tickets.csv`
holds 30 tickets written by hand in different words (out-of-distribution test).

## Model
```
"vpn keeps dropping" -> words + word bigrams + character 3-grams -> ids
   -> nn.EmbeddingBag (64-d, mean) -> Dropout -> Linear -> ReLU -> Linear -> 6 classes
```
A fastText-style model: 168K parameters, trains in about 4 seconds on CPU. Character
3-grams make it robust to typos and unseen word forms ("outlok", "emails").
Training: Adam, cross-entropy loss, 70/15/15 split, early stopping on validation loss.

## Results
| | PyTorch model | TF-IDF + Logistic Regression baseline |
|---|---|---|
| Test accuracy (540 generated tickets) | 97.8% | 97.8% |
| Hand-written tickets (30) | 70% | 80% |

With the confidence gate (below 0.6 goes to human review): **19 of 30** hand-written
tickets are routed automatically, and **18 of those 19 (95%) are correct**.

**The main lesson (say this in the interview):** 98% on the test set did not mean 98% in
the real world. The test set came from the same generator as the training set, so it
over-estimated performance. The hand-written set exposed the gap. Adding character
n-grams raised it from 63% to 70%. I did *not* tune on the hand-written set, because then
it would stop being an honest test. The real fixes are real ticket data or pretrained
language models, and meanwhile the confidence gate keeps wrong automatic routes rare.

## Optimisation (`python -m app.benchmark`)
| Experiment | Result |
|---|---|
| float16 weights instead of float32 | file 659 KB -> 331 KB, accuracy unchanged |
| batch of 128 instead of 1 | about 3x less time per ticket (0.11 ms -> 0.04 ms) |

I also tried PyTorch dynamic int8 quantization: the file was only 2% smaller and inference
was *slower*, because quantization only affects Linear layers and most weights here sit in
the embedding table. Measuring before optimising showed it was not worth it.

## Web demo
Interactive Streamlit app: type a ticket and see the category, confidence, assigned team and probability chart; classify many tickets at once in the Batch tab.

```bash
streamlit run streamlit_app.py      # opens http://localhost:8501
```
On first start the app trains the model if no saved model exists (same code as `app/train.py`).

## Run it
```bash
pip install -r requirements.txt
python -m app.generate_data      # creates data/tickets.csv
python -m app.train              # trains, prints metrics, saves models/
python -m app.benchmark          # compression + batching experiments
python -m pytest -q              # 8 tests
uvicorn app.api:app --reload     # http://127.0.0.1:8000/docs
```

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/classify` | one ticket -> category, confidence, top 3, assigned team |
| POST | `/classify/batch` | up to 500 tickets in one forward pass |
| GET | `/model/info` | training metrics |
| GET | `/health` | liveness check |

## Next steps
- Fine-tune a small pretrained transformer (DistilBERT) and compare on the hand-written set.
- Use an LLM zero-shot classifier for tickets the model is unsure about.
- Add a priority model (P1-P4) and track accuracy from agents' corrections over time.
