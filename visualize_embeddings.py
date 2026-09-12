"""
Visualize the custom-trained subword embeddings using PCA.

Projects the learned 50-dimensional embeddings down to 2D and plots a
sample of tokens to inspect whether semantically related words cluster
together (e.g. sentiment-bearing words, common stopwords, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_vectors_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        vocab_size, dim = int(header[0]), int(header[1])
        tokens = []
        vectors = np.zeros((vocab_size, dim))
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split(" ")
            tokens.append(parts[0])
            vectors[i] = np.array(parts[1:], dtype=np.float64)
    return tokens, vectors


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Minimal from-scratch PCA (mean-center + SVD) — no sklearn dependency needed."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:2].T


if __name__ == "__main__":
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    tokens, vectors = load_vectors_txt("custom_embeddings.vec")
    print(f"Loaded {len(tokens)} embeddings of dimension {vectors.shape[1]}")

    # Pick a curated + frequency-based sample of tokens to visualize
    highlight_words = [
        "good</w>", "great</w>", "bad</w>", "terrible</w>", "horrible</w>",
        "love</w>", "hate</w>", "amazing</w>", "boring</w>", "wonderful</w>",
        "worst</w>", "best</w>", "happy</w>", "sad</w>", "awful</w>",
        "movie</w>", "film</w>", "show</w>", "story</w>", "acting</w>",
        "the</w>", "and</w>", "was</w>", "is</w>", "not</w>",
    ]
    present = [w for w in highlight_words if w in tokens]
    idx = [tokens.index(w) for w in present]

    # Also add a random sample of other tokens for broader context
    rng = np.random.default_rng(0)
    other_idx = [i for i in range(len(tokens)) if i not in idx]
    sample_other = rng.choice(other_idx, size=min(120, len(other_idx)), replace=False)

    all_idx = list(idx) + list(sample_other)
    sub_vectors = vectors[all_idx]
    sub_tokens = [tokens[i] for i in all_idx]

    coords = pca_2d(sub_vectors)

    plt.figure(figsize=(11, 9))
    plt.scatter(coords[len(idx):, 0], coords[len(idx):, 1],
                c="lightgray", s=15, label="other tokens")
    plt.scatter(coords[:len(idx), 0], coords[:len(idx), 1],
                c="crimson", s=40, label="highlighted words")
    for i, tok in enumerate(sub_tokens[:len(idx)]):
        plt.annotate(tok.replace("</w>", ""), (coords[i, 0], coords[i, 1]),
                     fontsize=9, xytext=(3, 3), textcoords="offset points")

    plt.title("PCA Projection of Custom-Trained Subword Embeddings")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "embedding_pca.png", dpi=150)
    plt.close()
    print(f"Saved visualization to {OUTPUT_DIR / 'embedding_pca.png'}")
