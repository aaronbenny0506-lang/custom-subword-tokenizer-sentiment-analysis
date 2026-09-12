"""
Custom Byte-Pair Encoding (BPE) Subword Tokenizer — implemented from scratch.

No external tokenizer libraries (no HuggingFace tokenizers, no sentencepiece,
no nltk/spaCy word-piece utilities) are used for the subword algorithm itself.
Only Python's standard library (re, collections, json) is used.

Algorithm (Sennrich et al., 2016 — "Neural Machine Translation of Rare Words
with Subword Units"):
    1. Split the training corpus into words; represent each word as a
       sequence of characters plus an end-of-word marker "</w>".
    2. Count how often each adjacent symbol pair occurs across the corpus.
    3. Merge the most frequent pair into a new symbol; add it to the vocab.
    4. Repeat steps 2-3 for a fixed number of merges (a hyperparameter that
       controls final vocabulary size).
    5. To tokenize new text, apply the learned merges in the order they were
       learned (greedy, priority-based merging).
"""

import re
import json
import collections
from pathlib import Path


class BPETokenizer:
    END_OF_WORD = "</w>"
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"

    def __init__(self):
        self.merges = []          # ordered list of merged pairs, e.g. [("l", "o"), ...]
        self.merge_ranks = {}     # pair -> priority (lower = merged earlier)
        self.token2id = {}
        self.id2token = {}

    # ----------------------------------------------------------------
    # Pre-tokenization: corpus/text -> list of words
    # ----------------------------------------------------------------
    @staticmethod
    def _basic_clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)            # strip HTML tags (e.g. <br />)
        text = re.sub(r"http\S+|www\S+", " ", text)    # strip URLs
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _words(text: str) -> list:
        """Split cleaned text into word-like tokens (words + punctuation)."""
        text = BPETokenizer._basic_clean(text)
        return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text)

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def train(self, corpus: list, vocab_size: int = 2000, min_pair_freq: int = 2, verbose: bool = True):
        """
        Learn BPE merges from a list of raw text documents.

        Args:
            corpus: list of raw text strings.
            vocab_size: target number of merge operations (roughly controls
                        final subword vocabulary size).
            min_pair_freq: stop merging once the best pair occurs fewer than
                           this many times (avoids overfitting to noise).
        """
        # 1. Build word frequency table from the whole corpus.
        word_freq = collections.Counter()
        for doc in corpus:
            for w in self._words(doc):
                word_freq[w] += 1

        # 2. Represent each word as a tuple of characters + end marker.
        vocab = {
            tuple(list(word) + [self.END_OF_WORD]): freq
            for word, freq in word_freq.items()
        }

        if verbose:
            print(f"Training BPE on {len(word_freq)} unique words "
                  f"({sum(word_freq.values())} total word occurrences)...")

        merges = []
        for i in range(vocab_size):
            pairs = self._get_pair_stats(vocab)
            if not pairs:
                break
            best_pair, best_freq = max(pairs.items(), key=lambda kv: kv[1])
            if best_freq < min_pair_freq:
                break
            vocab = self._merge_vocab(best_pair, vocab)
            merges.append(best_pair)
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i + 1}/{vocab_size}: {best_pair} (freq={best_freq})")

        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        # 3. Build the final token vocabulary: base characters + all merged symbols.
        symbols = set()
        for word_tuple in vocab:
            symbols.update(word_tuple)
        # also include any base characters seen before merges (safety net)
        for word in word_freq:
            symbols.update(list(word))
        symbols.add(self.END_OF_WORD)

        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
        all_tokens = special_tokens + sorted(symbols)
        self.token2id = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

        if verbose:
            print(f"Done. Learned {len(self.merges)} merges. "
                  f"Final subword vocabulary size: {len(self.token2id)}")

    @staticmethod
    def _get_pair_stats(vocab: dict) -> dict:
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: tuple, vocab: dict) -> dict:
        a, b = pair
        merged_symbol = a + b
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    # ----------------------------------------------------------------
    # Tokenization (inference)
    # ----------------------------------------------------------------
    def _bpe_word(self, word: str) -> list:
        """Apply learned merges (in priority order) to segment one word."""
        symbols = list(word) + [self.END_OF_WORD]
        if len(symbols) == 1:
            return symbols

        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # pick the pair with the lowest merge rank (i.e. learned earliest)
            candidate = None
            candidate_rank = None
            for p in pairs:
                if p in self.merge_ranks:
                    r = self.merge_ranks[p]
                    if candidate_rank is None or r < candidate_rank:
                        candidate, candidate_rank = p, r
            if candidate is None:
                break
            a, b = candidate
            merged = a + b
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def tokenize(self, text: str) -> list:
        """Tokenize raw text into a flat list of subword tokens."""
        tokens = []
        for word in self._words(text):
            tokens.extend(self._bpe_word(word))
        return tokens

    def encode(self, text: str) -> list:
        """Tokenize + map to integer ids (unknown subwords -> <unk>)."""
        unk_id = self.token2id[self.UNK_TOKEN]
        return [self.token2id.get(tok, unk_id) for tok in self.tokenize(text)]

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------
    def save(self, path: str):
        data = {
            "merges": [list(pair) for pair in self.merges],
            "token2id": self.token2id,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token2id = data["token2id"]
        self.id2token = {int(v) if isinstance(v, str) else v: k for k, v in self.token2id.items()}
        # ensure id2token keys are ints matching token2id values
        self.id2token = {v: k for k, v in self.token2id.items()}


# ----------------------------------------------------------------------
# Demo / self-test: train on a small corpus and tokenize 10 sample sentences
# ----------------------------------------------------------------------
SAMPLE_SENTENCES = [
    "The movie was absolutely wonderful and heartwarming.",
    "This film was a complete waste of time, terribly boring.",
    "I loved the acting, the story was captivating from start to finish!",
    "Worst movie I have ever seen, the plot made no sense at all.",
    "A masterpiece of modern cinema, beautifully directed and acted.",
    "The special effects were amazing but the dialogue felt awkward.",
    "I would not recommend this film to anyone, very disappointing.",
    "An emotional rollercoaster with brilliant performances throughout.",
    "The pacing was slow and the characters were poorly developed.",
    "Absolutely thrilling from beginning to end, a must watch!",
]


if __name__ == "__main__":
    import pandas as pd

    # Train on a sample of the actual sentiment dataset for a more
    # representative vocabulary, falling back to the demo sentences only.
    try:
        df = pd.read_csv("Sentiment_Analysis.csv")
        corpus = df["text"].astype(str).sample(n=6000, random_state=42).tolist()
    except FileNotFoundError:
        corpus = SAMPLE_SENTENCES * 50  # tiny fallback corpus

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=1500, min_pair_freq=3)
    tokenizer.save("subword_vocab.json")

    print("\n=== Tokenizing 10 sample sentences ===")
    for i, sent in enumerate(SAMPLE_SENTENCES, 1):
        tokens = tokenizer.tokenize(sent)
        print(f"\n[{i}] {sent}")
        print("   ->", tokens)
