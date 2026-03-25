# Legal Document Summarization using InCaseLawBERT and Knapsack Optimization

An extractive summarization system for Indian Supreme Court judgments combining LDA topic clustering, LexRank graph ranking, InCaseLawBERT semantic embeddings, and knapsack-based sentence selection. This project implements a novel pipeline evaluated across 473 real legal documents.

---

## Project Overview

Indian Supreme Court judgments are long, dense, and difficult to parse quickly. This project builds an automated summarization pipeline that:

1. Extracts text from raw PDF judgment files
2. Cleans and preprocesses legal text (removes page numbers, citations, noise)
3. Groups sentences by legal topic using LDA clustering
4. Selects the most important sentence from each topic cluster using TextRank
5. Augments with globally important sentences using LexRank
6. Scores the final candidate pool using InCaseLawBERT semantic embeddings
7. Selects the optimal summary within a 50% word budget using knapsack dynamic programming
8. Compares BERT-based approach vs TF-IDF baseline using Kendall's Tau

---

## Algorithm — URL-KnapSum (Novel Approach)

This project implements a custom summarization algorithm combining:

```
PDF Judgments (500 documents)
        ↓
  Text extraction (pdfplumber, first 10 pages)
        ↓
  Legal text cleaning (page numbers, citations, noise removal)
        ↓
  Sentence tokenization (min 5 words per sentence)
        ↓
  LDA Topic Clustering (7 topics)
  → Best sentence per cluster via TextRank
        ↓
  LexRank Global Ranking (top 5 sentences)
        ↓
  Candidate Pool = Cluster candidates ∪ Global top-5
        ↓
  InCaseLawBERT Sentence Embeddings
        ↓
  Knapsack DP Optimizer (50% word budget)
        ↓
  Final Summary
```

### Why Knapsack?
Each sentence has a **value** (semantic similarity to the document mean vector) and a **weight** (word count). The knapsack algorithm finds the optimal combination of sentences that maximizes total value within the word budget — this is provably optimal unlike greedy selection.

---

## Two Approaches Compared

| Approach | Sentence Scoring | Clustering | Ranking |
|---|---|---|---|
| TF-IDF Baseline | TF-IDF cosine similarity | LDA | LexRank (TF-IDF) |
| BERT-KnapSum (Novel) | InCaseLawBERT embeddings | LDA | Semantic PageRank |

---

## Technical Components

| Component | Detail |
|---|---|
| Embedding model | `law-ai/InCaseLawBERT` (legal domain BERT) |
| Topic model | LDA (Gensim), 7 topics, 20 passes |
| Graph ranking | NetworkX PageRank on cosine similarity matrix |
| Optimization | 0/1 Knapsack dynamic programming |
| Word budget | 50% of original document length |
| PDF extraction | pdfplumber, first 10 pages per document |
| Evaluation metric | Kendall's Tau (semantic consistency) |

---

## Evaluation

Evaluated on 473 valid documents from the Indian Supreme Court Judgments dataset.

| Method | Avg Kendall's Tau | Interpretation |
|---|---|---|
| TF-IDF Baseline | lower | Less consistent ranking |
| BERT-KnapSum (Ours) | ~0.9872 | High semantic consistency |

Kendall's Tau measures the agreement between graph-based sentence ranking and semantic centroid similarity — higher means the summary is more semantically coherent with the original document.

---

## Dataset

**Indian Supreme Court Judgments** from Kaggle:
```
kaggle datasets download -d vangap/indian-supreme-court-judgments
```

> Add your own Kaggle API credentials to the notebook before running:
> ```python
> os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
> os.environ['KAGGLE_KEY'] = 'your_api_key_here'
> ```

---

## Setup

```bash
pip install kaggle pdfplumber nltk scikit-learn networkx rouge-score gensim
pip install sentence-transformers torch scipy seaborn matplotlib pandas
python -m nltk.downloader punkt punkt_tab
```

---

## Project Structure

```
├── legal_goooooooddddddd__1_.py    # Main pipeline
└── README.md                       # This file
```

---

## Output

- `legal_summaries_output.txt` — all summaries in one file
- `Legal_Summaries_Archive.zip` — individual `.txt` file per document
- Visualizations:
  - BERT vs TF-IDF performance stability plot (scatter + regression)
  - Distribution box plot with swarm overlay (Kendall's Tau comparison)

---

## Libraries Used

- `pdfplumber` — PDF text extraction
- `nltk` — sentence tokenization
- `gensim` — LDA topic modeling
- `scikit-learn` — TF-IDF vectorization, cosine similarity
- `networkx` — PageRank graph scoring
- `sentence-transformers` — InCaseLawBERT semantic embeddings
- `scipy` — Kendall's Tau and Spearman correlation
- `matplotlib` / `seaborn` — result visualization

---

## Acknowledgements

- [InCaseLawBERT](https://huggingface.co/law-ai/InCaseLawBERT) by law-ai for the Indian legal domain language model
- [Indian Supreme Court Judgments Dataset](https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments) on Kaggle
- LexRank: Graph-based Lexical Centrality as Salience in Text Summarization (Erkan & Radev, 2004)
