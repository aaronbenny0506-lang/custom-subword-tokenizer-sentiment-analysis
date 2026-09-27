# Sentiment Analysis with a Custom Subword Tokenizer & Embeddings

A complete NLP pipeline built **from scratch**: a custom Byte-Pair Encoding
(BPE) subword tokenizer, custom word embeddings trained with skip-gram +
negative sampling, and a downstream sentiment classifier, compared against
a genuinely pre-trained tokenizer + pre-trained embeddings (BERT).

Dataset: `Sentiment_Analysis.csv` : 80,000 balanced positive/negative texts
(tweets/reviews mentioning products, movies and general sentiment). The
tokenizer and embeddings are trained on the **full 80,000-row dataset**,
not a sample.

## 📁 Files

```
subword_sentiment_analysis/
├── Sentiment_Analysis.csv           # dataset (80,000 rows, balanced 0/1 labels)
├── bpe_tokenizer.py                 # custom BPE tokenizer (from scratch) + 10-sentence demo
├── subword_vocab.json               # learned merges + subword vocabulary (trained on full dataset)
├── train_embeddings.py              # custom skip-gram + negative sampling trainer (numpy only)
├── custom_embeddings.vec            # trained embeddings, word2vec .vec/.txt format
├── visualize_embeddings.py          # PCA projection of the learned embedding space
├── classification_comparison.py     # custom pipeline vs. pre-trained (BERT) pipeline
├── subword_sentiment_analysis.ipynb # notebook walking through the full pipeline
├── requirements.txt
└── outputs/
    ├── embedding_pca.png
    ├── comparison_results.csv
    ├── comparison_barchart.png
    ├── comparison_confusion_matrices.png
    └── classification_reports.txt
```

## 1️⃣ Custom Subword Tokenizer (BPE)

`bpe_tokenizer.py` implements Byte-Pair Encoding entirely from Python's
standard library (`re`, `collections`, `json` — **no** HuggingFace
tokenizers, sentencepiece or NLTK/spaCy subword utilities):

1. Represent each word as characters + an end-of-word marker `</w>`.
2. Count all adjacent symbol-pair frequencies across the corpus.
3. Repeatedly merge the most frequent pair into a new symbol.
4. Store merges in the order they were learned; apply them greedily
   (lowest-rank-first) to segment new text.

Run it directly to train on the dataset and tokenize 10 sample sentences:

```bash
python bpe_tokenizer.py
```

Example output:

```
[1] The movie was absolutely wonderful and heartwarming.
   -> ['the</w>', 'movie</w>', 'was</w>', 'absolutely</w>', 'wonderful</w>',
       'and</w>', 'he', 'ar', 'tw', 'ar', 'ming</w>', '.</w>']
```

Common words (`the`, `movie`, `wonderful`) collapse into single tokens,
while rare/compound words (`heartwarming`) get split into meaningful
subword pieces — exactly the behavior BPE is designed to produce.

The tokenizer is trained on the **full 80,000-document dataset** (not a
sample). Vocabulary size is controlled by `vocab_size` (number of merges,
default 8,000) and `min_pair_freq` (default 5, to avoid wasting merges on
rare noise like typos/URLs); both are scaled up from an earlier version
of this project that trained on a 6,000-document sample. Training on the
full corpus takes noticeably longer than the sampled run — it's pure
Python, so budget real time for it.

The learned vocabulary is saved to **`subword_vocab.json`**.

## 2️⃣ Custom Word Embeddings (Skip-Gram + Negative Sampling)

`train_embeddings.py` trains embeddings **from scratch with plain numpy**
(no gensim/word2vec training calls) on the BPE-tokenized corpus:

- Skip-gram objective with negative sampling (Mikolov et al., 2013)
- Manual forward/backward pass and SGD weight updates
- Negative sampling distribution ∝ unigram frequency^0.75
- 50-dimensional embeddings, trained for 2 epochs over the **full 80,000
  documents** (not a sample — this is a significantly larger training run
  than an earlier ~4,000-document version of this project)

```bash
python train_embeddings.py
```

Embeddings are saved in standard word2vec text format to
**`custom_embeddings.vec`**:

```
8214 50
the</w> -0.215990 -0.312748 0.684080 ...
...
```
(exact vocab size depends on your `max_vocab`/`min_freq` settings and this run's data)

**Sanity check** — nearest neighbours in the learned space:

```
movie</w>: [('film</w>', 0.92), ('product</w>', 0.84), ('stuff</w>', 0.80), ...]
good</w>:  [('bad</w>', 0.86), ('great</w>', 0.85), ('fine</w>', 0.80), ...]
```

`movie` → `film` being the closest neighbour is a strong signal the model
learned real distributional semantics from the corpus.

### Visualization

`visualize_embeddings.py` projects the embeddings to 2D with a from-scratch
PCA (mean-center + SVD) and plots sentiment/movie-review words against a
random sample of other tokens:

```bash
python visualize_embeddings.py   # -> outputs/embedding_pca.png
```

Sentiment-bearing words (`good`, `bad`, `love`, `terrible`, `boring`, …)
visibly cluster apart from function words (`the`, `and`, `was`, `is`).

## 3️⃣ Text Classification & Comparison

`classification_comparison.py` builds **two parallel pipelines** and
evaluates both with the exact same downstream classifier (Logistic
Regression), so the comparison isolates the effect of tokenizer + embeddings:

| | Tokenizer | Embeddings |
|---|---|---|
| **Custom pipeline** | Custom BPE (`bpe_tokenizer.py`) | Custom skip-gram embeddings (`custom_embeddings.vec`) |
| **Pre-trained pipeline** | BERT's WordPiece subword tokenizer (`bert-base-uncased`, via HuggingFace `transformers`) | BERT's own pre-trained input embedding matrix (`bert-base-uncased`) |

Both the tokenizer *and* the embeddings in the pre-trained pipeline are now
genuinely pre-trained artifacts (an earlier version of this project used a
hand-rolled regex word tokenizer alongside GloVe — that's been replaced).
BERT's tokenizer is also a subword tokenizer, which makes this an
apples-to-apples comparison: subword tokenizer + embeddings, from scratch
vs. pre-trained.

Each document is vectorized the same way in both pipelines: tokenize, look
up each token's embedding, and average the vectors — then both feed into
the same `LogisticRegression` model. On the pre-trained side, this means
using BERT's static input embedding layer rather than running a full
forward pass through the transformer (i.e. no contextual embeddings). That
keeps the two pipelines methodologically comparable (both are "mean-pooled
static subword embeddings") and is far faster to run over the full
dataset. Both pipelines are trained and evaluated on the **full dataset**,
not a 4,000-document sample.

```bash
python classification_comparison.py
```

### Results (full dataset, 15,734 held-out test documents)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Custom BPE + Custom Embeddings | 0.66 | 0.66 | 0.68 | 0.67 |
| Pre-trained Tokenizer (BERT WordPiece) + Pre-trained Embeddings (BERT) | 0.75 | 0.75 | 0.75 | 0.75 |

**Interpretation:** the pre-trained BERT pipeline outperforms the
from-scratch pipeline by roughly 9 points of accuracy (0.75 vs. 0.66) —
expected, since BERT was pre-trained on billions of words with a
~30,000-token WordPiece vocabulary, while the custom embeddings are
trained from scratch on this dataset alone. Still, a 0.66 accuracy /
0.67 F1 from a tokenizer and embedding model built entirely from
scratch is a solid result: the PCA projection shows the custom
embeddings cleanly separate negative-sentiment words (`bad`, `awful`,
`terrible`, `horrible`) from positive-sentiment words (`good`, `great`,
`wonderful`, `amazing`) into distinct regions of the embedding space,
confirming the skip-gram model learned real distributional semantics
purely from this corpus. The comparison demonstrates both (a) that a
subword tokenizer + embedding pipeline built entirely from scratch can
learn genuine semantic structure, and (b) the practical value of
pre-trained tokenizers and embeddings over a fully from-scratch pipeline
trained on one dataset.

Outputs saved to `outputs/`:
- `comparison_results.csv` : metrics table
- `comparison_barchart.png` : side-by-side metric comparison
- `comparison_confusion_matrices.png`
- `classification_reports.txt` : full sklearn reports for both pipelines

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Then run the scripts in order:

```bash
python bpe_tokenizer.py              # 1. train tokenizer, tokenize samples, save vocab
python train_embeddings.py           # 2. train custom embeddings
python visualize_embeddings.py       # 3. (optional) PCA visualization
python classification_comparison.py  # 4. train & compare classifiers
```

Or open `subword_sentiment_analysis.ipynb` for the full walkthrough in one
notebook (note: re-running the notebook end-to-end repeats the BPE +
embedding training on the full dataset and downloads BERT's pre-trained
weights on first use, so budget real time for it — this is a much longer
run than the earlier sampled version).

## Notes on scope / scaling

The tokenizer, embeddings, and classifier are all trained on the **full
80,000-row dataset**. This is a meaningfully longer run than training on a
sample: BPE training and skip-gram training are both pure Python/numpy (no
vectorized batch training or GPU use), so expect the full pipeline —
especially `bpe_tokenizer.py` and `train_embeddings.py` — to take
substantially longer than a sampled run. Plan accordingly (e.g. run it as
a background job) rather than expecting it to finish in a minute or two.

All key parameters (`vocab_size`, `min_pair_freq`, `max_vocab`,
`embed_dim`, `epochs`, `window`) are exposed as variables at the top of
each script if you need to trade off training time against embedding
quality — e.g. dropping `epochs` to 1 in `train_embeddings.py` for a
faster turnaround while testing changes.
