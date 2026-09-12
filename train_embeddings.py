"""
Custom Word Embeddings — Skip-Gram with Negative Sampling, implemented from
scratch using only numpy (no gensim/word2vec training routines).

This trains embeddings for the SUBWORD tokens produced by our custom BPE
tokenizer (bpe_tokenizer.py) on the sentiment dataset, then saves them to a
plain-text .vec/.txt file in the standard word2vec text format:

    <vocab_size> <embedding_dim>
    <token> <v1> <v2> ... <vD>
    <token> <v1> <v2> ... <vD>
    ...

Algorithm (Mikolov et al., 2013):
    For each (center, context) pair drawn from a sliding window over the
    tokenized corpus, maximize the probability of the true context word
    while minimizing the probability of a handful of randomly sampled
    "negative" words (negative sampling), via logistic regression trained
    with plain SGD.
"""

import json
import time
import numpy as np
import collections
from pathlib import Path

from bpe_tokenizer import BPETokenizer


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embed_dim: int = 50, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # "input" embeddings (used as the final word vectors) and
        # "output" embeddings (context-side weights, discarded after training)
        self.W_in = (rng.random((vocab_size, embed_dim)) - 0.5) / embed_dim
        self.W_out = np.zeros((vocab_size, embed_dim))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def train_pair(self, center: int, context: int, negatives: np.ndarray, lr: float):
        """One SGD step for a single (center, true_context, negative_samples) example."""
        v_center = self.W_in[center]                     # (D,)

        # --- positive example (label = 1) ---
        u_pos = self.W_out[context]
        score_pos = self._sigmoid(np.dot(v_center, u_pos))
        grad_pos = (score_pos - 1.0)                      # d loss / d score
        grad_center = grad_pos * u_pos
        self.W_out[context] -= lr * grad_pos * v_center

        # --- negative examples (label = 0) ---
        u_neg = self.W_out[negatives]                      # (K, D)
        scores_neg = self._sigmoid(u_neg @ v_center)        # (K,)
        grad_neg = scores_neg                               # (K,)
        grad_center += grad_neg @ u_neg
        self.W_out[negatives] -= lr * (grad_neg[:, None] * v_center[None, :])

        self.W_in[center] -= lr * grad_center


def build_vocab_and_corpus(tokenized_docs: list, max_vocab: int = 6000, min_freq: int = 2):
    """Build a frequency-capped vocabulary and convert docs to id sequences."""
    counter = collections.Counter()
    for doc in tokenized_docs:
        counter.update(doc)

    most_common = [w for w, c in counter.most_common(max_vocab) if c >= min_freq]
    token2id = {tok: i for i, tok in enumerate(most_common)}

    id_corpus = []
    for doc in tokenized_docs:
        ids = [token2id[t] for t in doc if t in token2id]
        if len(ids) >= 2:
            id_corpus.append(ids)

    freqs = np.array([counter[tok] for tok in most_common], dtype=np.float64)
    return token2id, id_corpus, freqs


def make_negative_sampling_table(freqs: np.ndarray, power: float = 0.75, table_size: int = 1_000_000):
    """Precompute a sampling table biased toward more frequent words (unigram^0.75)."""
    weights = freqs ** power
    probs = weights / weights.sum()
    counts = np.round(probs * table_size).astype(np.int64)
    table = np.repeat(np.arange(len(freqs)), np.maximum(counts, 0))
    return table


def train_word2vec(id_corpus, vocab_size, embed_dim=50, window=3, negatives=5,
                    epochs=3, lr=0.02, freqs=None, seed=42, verbose=True):
    model = SkipGramNegativeSampling(vocab_size, embed_dim, seed=seed)
    rng = np.random.default_rng(seed)
    neg_table = make_negative_sampling_table(freqs)

    start = time.time()
    n_pairs_total = 0
    for epoch in range(epochs):
        n_pairs = 0
        for doc in id_corpus:
            L = len(doc)
            for i, center in enumerate(doc):
                w = rng.integers(1, window + 1)
                lo, hi = max(0, i - w), min(L, i + w + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    context = doc[j]
                    neg_idx = neg_table[rng.integers(0, len(neg_table), size=negatives)]
                    model.train_pair(center, context, neg_idx, lr)
                    n_pairs += 1
        n_pairs_total += n_pairs
        if verbose:
            elapsed = time.time() - start
            print(f"  epoch {epoch + 1}/{epochs}: {n_pairs} training pairs "
                  f"(elapsed {elapsed:.1f}s)")
    return model


def save_vectors_txt(path: str, id2token: dict, vectors: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(id2token)} {vectors.shape[1]}\n")
        for idx in range(len(id2token)):
            vec_str = " ".join(f"{x:.6f}" for x in vectors[idx])
            f.write(f"{id2token[idx]} {vec_str}\n")


if __name__ == "__main__":
    import pandas as pd

    print("Loading dataset and BPE tokenizer...")
    df = pd.read_csv("Sentiment_Analysis.csv")
    sample = df.sample(n=4000, random_state=42).reset_index(drop=True)

    tokenizer = BPETokenizer()
    tokenizer.load("subword_vocab.json")

    print("Tokenizing corpus with custom BPE tokenizer...")
    tokenized_docs = [tokenizer.tokenize(t) for t in sample["text"].astype(str)]

    print("Building word2vec training vocabulary...")
    token2id, id_corpus, freqs = build_vocab_and_corpus(tokenized_docs, max_vocab=1800, min_freq=2)
    id2token = {i: t for t, i in token2id.items()}
    print(f"Word2Vec vocabulary size: {len(token2id)}; training sequences: {len(id_corpus)}")

    print("Training skip-gram with negative sampling (from scratch, numpy)...")
    EMBED_DIM = 50
    model = train_word2vec(
        id_corpus, vocab_size=len(token2id), embed_dim=EMBED_DIM,
        window=2, negatives=5, epochs=2, lr=0.025, freqs=freqs
    )

    print("Saving embeddings to custom_embeddings.vec ...")
    save_vectors_txt("custom_embeddings.vec", id2token, model.W_in)

    # quick sanity check: nearest neighbours for a few tokens
    def most_similar(token, topn=5):
        if token not in token2id:
            return []
        vecs = model.W_in
        norm_vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        idx = token2id[token]
        sims = norm_vecs @ norm_vecs[idx]
        top_idx = np.argsort(-sims)[1:topn + 1]
        return [(id2token[i], float(sims[i])) for i in top_idx]

    print("\nSanity check — nearest neighbours in learned embedding space:")
    for probe in ["good</w>", "bad</w>", "movie</w>", "love</w>"]:
        print(f"  {probe}: {most_similar(probe)}")
