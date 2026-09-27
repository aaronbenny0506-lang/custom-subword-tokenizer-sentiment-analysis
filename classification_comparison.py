"""
Text Classification: Custom Subword Tokenizer + Custom Embeddings
                      vs.
                      Pre-trained Tokenizer + Pre-trained Embeddings (BERT)

Both pipelines follow the same recipe so the comparison is fair:
    1. Tokenize each document.
    2. Look up each token's embedding vector.
    3. Average the token vectors to get one fixed-length document vector.
    4. Train the SAME classifier (Logistic Regression) on these features.
    5. Evaluate with Accuracy, Precision, Recall, F1-score.

Custom pipeline:  BPETokenizer (bpe_tokenizer.py) + custom_embeddings.vec
                  (both trained from scratch in this project on the full
                  80,000-document dataset)
Pre-trained pipeline: a genuine pre-trained tokenizer + pre-trained
                  embeddings — BERT's own WordPiece subword tokenizer
                  (google-bert/bert-base-uncased, via HuggingFace
                  `transformers`) paired with BERT's own pre-trained token
                  embedding matrix. This replaces an earlier version of
                  this script that used a hand-rolled regex word tokenizer
                  alongside GloVe; both the tokenizer *and* the embeddings
                  are now genuinely pre-trained artifacts, and — like the
                  custom pipeline — BERT's is a subword tokenizer, which
                  makes the comparison apples-to-apples (subword tokenizer
                  + embeddings, from scratch vs. pre-trained).

Document vectors are built the same way in both pipelines: tokenize, look
up each token's embedding, and mean-pool. For the pre-trained pipeline
this uses BERT's static input embedding layer (not a full contextual
forward pass through the transformer). That's a deliberate choice: it
keeps the two pipelines methodologically comparable (both are "mean-pooled
static subword embeddings") and is dramatically faster to run over the
full dataset, since it's a plain embedding lookup rather than 80,000
forward passes through a 12-layer transformer. If you want a stronger
(but much slower) pre-trained baseline, swap `doc_vector_pretrained` for a
version that runs the full BERT model and mean-pools the last hidden
state instead of the embedding layer.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from bpe_tokenizer import BPETokenizer

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
BERT_MODEL_NAME = "bert-base-uncased"


# --------------------------------------------------------------------
# Load data — the FULL dataset, not a 4,000-document sample
# --------------------------------------------------------------------
def load_data():
    df = pd.read_csv("Sentiment_Analysis.csv")
    df = df.dropna(subset=["text", "sentiment"]).drop_duplicates(subset=["text"])
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# --------------------------------------------------------------------
# Pipeline A: custom BPE tokenizer + custom embeddings
# --------------------------------------------------------------------
def load_vectors_txt(path: str):
    vecs = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        dim = int(header[1])
        for line in f:
            parts = line.rstrip("\n").split(" ")
            vecs[parts[0]] = np.array(parts[1:], dtype=np.float64)
    return vecs, dim


def doc_vector_custom(text: str, tokenizer: BPETokenizer, embeddings: dict, dim: int) -> np.ndarray:
    tokens = tokenizer.tokenize(text)
    vecs = [embeddings[t] for t in tokens if t in embeddings]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


# --------------------------------------------------------------------
# Pipeline B: genuine pre-trained tokenizer + pre-trained embeddings (BERT)
# --------------------------------------------------------------------
def load_bert(model_name: str = BERT_MODEL_NAME):
    """Load BERT's pre-trained WordPiece tokenizer and its input embedding matrix."""
    from transformers import AutoTokenizer, AutoModel
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()
    embedding_matrix = bert_model.get_input_embeddings().weight.detach().numpy()
    return bert_tokenizer, embedding_matrix


def pretrained_tokenize(text: str, bert_tokenizer) -> list:
    """BERT's own pre-trained WordPiece subword tokenizer."""
    return bert_tokenizer.tokenize(text)


def doc_vector_pretrained(text: str, bert_tokenizer, embedding_matrix: np.ndarray, dim: int) -> np.ndarray:
    tokens = pretrained_tokenize(text, bert_tokenizer)
    if not tokens:
        return np.zeros(dim)
    ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    vecs = embedding_matrix[ids]
    return np.mean(vecs, axis=0)


# --------------------------------------------------------------------
# Evaluation helper
# --------------------------------------------------------------------
def evaluate(name, y_true, y_pred, reports):
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }
    reports[name] = classification_report(y_true, y_pred, target_names=["negative", "positive"])
    return metrics


def main():
    print("Loading full dataset...")
    df = load_data()
    print(f"Using {len(df)} documents "
          f"({df['sentiment'].value_counts().to_dict()})")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"]
    )

    results = []
    reports = {}

    # ---------------- Pipeline A: custom BPE + custom embeddings ----------------
    print("\n[Custom pipeline] Loading BPE tokenizer + custom embeddings...")
    tokenizer = BPETokenizer()
    tokenizer.load("subword_vocab.json")
    custom_vecs, custom_dim = load_vectors_txt("custom_embeddings.vec")
    print(f"  Custom embedding vocab size: {len(custom_vecs)}, dim: {custom_dim}")

    print("[Custom pipeline] Building document vectors...")
    X_train_custom = np.vstack([
        doc_vector_custom(t, tokenizer, custom_vecs, custom_dim) for t in train_df["text"]
    ])
    X_test_custom = np.vstack([
        doc_vector_custom(t, tokenizer, custom_vecs, custom_dim) for t in test_df["text"]
    ])

    print("[Custom pipeline] Training Logistic Regression classifier...")
    clf_custom = LogisticRegression(max_iter=1000)
    clf_custom.fit(X_train_custom, train_df["sentiment"])
    preds_custom = clf_custom.predict(X_test_custom)
    results.append(evaluate("Custom BPE + Custom Embeddings", test_df["sentiment"], preds_custom, reports))

    # ---------------- Pipeline B: pre-trained tokenizer + pre-trained BERT embeddings ----------------
    print(f"\n[Pre-trained pipeline] Loading BERT tokenizer + embeddings ({BERT_MODEL_NAME})...")
    bert_tokenizer, bert_embeddings = load_bert()
    bert_dim = bert_embeddings.shape[1]
    print(f"  BERT vocab size: {bert_embeddings.shape[0]}, dim: {bert_dim}")

    print("[Pre-trained pipeline] Building document vectors...")
    X_train_bert = np.vstack([
        doc_vector_pretrained(t, bert_tokenizer, bert_embeddings, bert_dim) for t in train_df["text"]
    ])
    X_test_bert = np.vstack([
        doc_vector_pretrained(t, bert_tokenizer, bert_embeddings, bert_dim) for t in test_df["text"]
    ])

    print("[Pre-trained pipeline] Training Logistic Regression classifier...")
    clf_bert = LogisticRegression(max_iter=1000)
    clf_bert.fit(X_train_bert, train_df["sentiment"])
    preds_bert = clf_bert.predict(X_test_bert)
    results.append(evaluate("Pre-trained Tokenizer (BERT WordPiece) + Pre-trained Embeddings (BERT)",
                             test_df["sentiment"], preds_bert, reports))

    # ---------------- Save results ----------------
    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print("\n=== Comparison: Custom pipeline vs. Pre-trained pipeline ===")
    print(results_df)
    results_df.to_csv(OUTPUT_DIR / "comparison_results.csv")

    with open(OUTPUT_DIR / "classification_reports.txt", "w") as f:
        for name, rep in reports.items():
            f.write(f"=== {name} ===\n{rep}\n\n")

    # ---------------- Visualization: bar chart ----------------
    ax = results_df.plot(kind="bar", figsize=(9, 6), rot=15)
    ax.set_title("Custom Pipeline vs. Pre-trained Pipeline")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_barchart.png", dpi=150)
    plt.close()

    # ---------------- Visualization: confusion matrices ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, preds) in zip(axes, [("Custom BPE + Custom Embeddings", preds_custom),
                                         ("Pre-trained (BERT)", preds_bert)]):
        cm = confusion_matrix(test_df["sentiment"], preds)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_confusion_matrices.png", dpi=150)
    plt.close()

    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
