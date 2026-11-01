"""Streamlit web UI for the ticket classifier.

Run locally:  streamlit run streamlit_app.py
"""
import json

import pandas as pd
import streamlit as st

from app.predictor import MODEL_DIR, REVIEW_THRESHOLD, TicketPredictor

st.set_page_config(page_title="IT Ticket Classifier", page_icon="🎫", layout="wide")

ROUTING = {
    "network": "Network Operations",
    "hardware": "Desktop Support",
    "access": "Identity & Access Management",
    "password": "Service Desk L1",
    "email": "Messaging Team",
    "software": "Application Support",
}
EXAMPLES = [
    "my vpn keeps dropping when working from home",
    "forgot my password and my account is locked",
    "outlook is not receiving new emails",
    "need access to the finance sharepoint site",
    "the touchpad is not responding",
]


@st.cache_resource(show_spinner="Training the model (first start only, a few seconds)...")
def load_predictor() -> TicketPredictor:
    # Model weights are not committed to Git, so a fresh deployment trains once.
    if not (MODEL_DIR / "model.pt").exists():
        from app.train import main as train
        train()
    return TicketPredictor()


predictor = load_predictor()
metrics = json.loads((MODEL_DIR / "metrics.json").read_text())

st.title("🎫 IT Support Ticket Classifier")
st.write("A PyTorch neural network reads a helpdesk ticket and routes it to the right team. "
         f"If it is less than {REVIEW_THRESHOLD:.0%} sure, the ticket goes to a human instead.")

tab_one, tab_batch, tab_model = st.tabs(["Classify a ticket", "Batch", "About the model"])

with tab_one:
    choice = st.selectbox("Pick an example or write your own below", ["(write my own)"] + EXAMPLES)
    text = st.text_area("Ticket text", value="" if choice == "(write my own)" else choice,
                        placeholder="e.g. printer on 3rd floor is jammed")
    if text.strip():
        r = predictor.classify([text])[0]
        team = "Service Desk Triage (human review)" if r["needs_human_review"] else ROUTING[r["category"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", r["category"])
        c2.metric("Confidence", f"{r['confidence']:.0%}")
        c3.caption("Assigned team")  # plain text: long team names get cut off in st.metric
        c3.markdown(f"#### {team}")
        if r["needs_human_review"]:
            st.warning("The model is unsure, so this ticket is sent to a human for review.")
        else:
            st.success(f"Auto-routed to **{team}**.")
        probs = predictor.predict_proba([text])[0].tolist()
        st.bar_chart(pd.DataFrame({"probability": probs}, index=predictor.labels))

with tab_batch:
    st.write("Paste one ticket per line — they are classified in a single batch.")
    lines = st.text_area("Tickets", value="\n".join(EXAMPLES), height=180)
    tickets = [t for t in lines.splitlines() if t.strip()]
    if tickets:
        rows = predictor.classify(tickets)
        st.dataframe(pd.DataFrame([{
            "ticket": t, "category": r["category"], "confidence": f"{r['confidence']:.0%}",
            "team": "Human review" if r["needs_human_review"] else ROUTING[r["category"]],
        } for t, r in zip(tickets, rows)]), use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy (generated tickets)", f"{metrics['test_accuracy']:.1%}")
    m2.metric("Hand-written tickets", f"{metrics.get('handwritten_accuracy', 0):.0%}")
    m3.metric("Parameters", f"{metrics['parameters']:,}")
    st.write(
        "**Why the gap?** Training data is synthetic (real tickets are confidential), and the test set "
        "came from the same generator, so it over-estimates real performance. Tickets written by hand "
        "in different words are harder. The confidence gate keeps wrong automatic routes rare: on the "
        "hand-written set, 18 of the 19 auto-routed tickets were correct.")
    st.markdown("**Architecture:** words + word pairs + 3-letter chunks → `EmbeddingBag` (64-d average) "
                "→ Dropout → Linear → ReLU → Linear → 6 classes. Trained with Adam and early stopping.")
    st.caption(f"Baseline (TF-IDF + Logistic Regression): {metrics['baseline_tfidf_logreg_accuracy']:.1%} "
               f"on the test set, {metrics.get('handwritten_baseline_accuracy', 0):.0%} on hand-written tickets.")

st.divider()
st.caption("Built with PyTorch and Streamlit · "
           "[Source code](https://github.com/Ayaz17S/it-ticket-classifier-pytorch)")
