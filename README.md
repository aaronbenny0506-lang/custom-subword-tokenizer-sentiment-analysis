# Sentiment Analysis with a Custom Subword Tokenizer & Embeddings

A complete NLP pipeline built **from scratch**: a custom Byte-Pair Encoding
(BPE) subword tokenizer, custom word embeddings trained with skip-gram +
negative sampling, and a downstream sentiment classifier, compared against
a pre-trained tokenizer + pre-trained GloVe embeddings.

Dataset: `Sentiment_Analysis.csv` : 80,000 balanced positive/negative texts
(tweets/reviews mentioning products, movies and general sentiment).

## 📁 Files

```
subword_sentiment_analysis/
├── Sentiment_Analysis.csv           # dataset (80,000 rows, balanced 0/1 labels)
├── bpe_tokenizer.py                 # custom BPE tokenizer (from scratch) + 10-sentence demo
├── subword_vocab.json               # learned merges + subword vocabulary (1,783 tokens)
├── train_embeddings.py              # custom skip-gram + negative sampling trainer (numpy only)
├── custom_embeddings.vec            # trained embeddings, word2vec .vec/.txt format
├── visualize_embeddings.py          # PCA projection of the learned embedding space
├── classification_comparison.py     # custom pipeline vs. pre-trained (GloVe) pipeline
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

The learned vocabulary (1,500 merges, 1,783 total subword tokens) is saved
to **`subword_vocab.json`**.

## 2️⃣ Custom Word Embeddings (Skip-Gram + Negative Sampling)

`train_embeddings.py` trains embeddings **from scratch with plain numpy**
(no gensim/word2vec training calls) on the BPE-tokenized corpus:

- Skip-gram objective with negative sampling (Mikolov et al., 2013)
- Manual forward/backward pass and SGD weight updates
- Negative sampling distribution ∝ unigram frequency^0.75
- 50-dimensional embeddings, trained for 2 epochs over ~4,000 documents
  (~1.85M training pairs/epoch)

```bash
python train_embeddings.py
```

Embeddings are saved in standard word2vec text format to
**`custom_embeddings.vec`**:

```
1622 50
the</w> -0.215990 -0.312748 0.684080 ...
...
```

**Sanity check** — nearest neighbours in the learned space:

```
movie</w>: [('film</w>', 0.92), ('product</w>', 0.84), ('stuff</w>', 0.80), ...]
good</w>:  [('bad</w>', 0.86), ('great</w>', 0.85), ('fine</w>', 0.80), ...]
```

`movie` → `film` being the closest neighbour is a strong signal the model
learned real distributional semantics from a fairly small corpus.

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
| **Pre-trained pipeline** | Regex word tokenizer (representative of a standard tokenizer) | GloVe 50d (`glove-wiki-gigaword-50`, via `gensim.downloader`) |

Each document is vectorized by averaging its token embeddings, then fed to
the same `LogisticRegression` model.

```bash
python classification_comparison.py
```

### Results (4,000-document balanced sample)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Custom BPE + Custom Embeddings | 0.573 | 0.562 | 0.658 | 0.606 |
| Pre-trained Tokenizer + GloVe-50d | 0.706 | 0.693 | 0.740 | 0.716 |

**Interpretation:** the pre-trained GloVe pipeline outperforms the
from-scratch pipeline, which is expected — GloVe was trained on billions of
words with a 400,000-word vocabulary, while the custom embeddings were
trained from scratch on only ~4,000 documents for 2 epochs. The comparison
demonstrates both (a) that a subword tokenizer + embedding pipeline built
entirely from scratch *can* learn real semantic structure, and (b) the
practical value of pre-trained embeddings especially on smaller datasets.

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
embedding training and downloads GloVe, so it takes a few minutes).

## Notes on scope / scaling

To keep training time reasonable for this exercise, both the tokenizer and
the embedding trainer are run on a ~4,000-document sample rather than the
full 80,000-row dataset, and BPE is capped at 1,500 merges. All parameters
(`SAMPLE_SIZE`, `vocab_size`, `embed_dim`, `epochs`, `window`) are exposed
as variables at the top of each script — increasing them (especially
sample size and epochs) will improve embedding quality and classification
performance at the cost of longer training time.
