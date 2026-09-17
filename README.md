
## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

## Final Project Results

### Dataset

- Total WikiNews records: 15,200
- English-language articles: 5,240
- Exclusive single-label selected-category articles: 3,261
- Articles suitable for detailed NLP analysis: 3,138
- NER and summarization sample: 60 articles
- Topic-classification dataset: 1,044 articles
- Classification dataset balance: 261 articles per category

### Named Entity Recognition

spaCy was used for sentence segmentation, tokenization, lemmatization, part-of-speech tagging, dependency parsing, and named entity recognition.

Named entities were linked to article metadata and analysed by entity type, news category, semantic group, and publication year.

A heuristic review of 50 sampled NER predictions found:

- 49 predictions with no obvious issue detected
- 1 prediction flagged as potentially ambiguous

This is a heuristic error-screening exercise rather than a gold-standard NER accuracy measurement.

### Text Summarization

Extractive summarization was applied to 60 articles using TF-IDF sentence representations and PageRank sentence ranking.

The analysis covered 15 articles from each of:

- Politics and conflicts
- Economy and business
- Science and technology
- Sports

Grammar and style heuristics were also applied to the generated summaries.

### Semantic Similarity

SentenceTransformer embeddings and cosine similarity were used to compare each generated summary with its original article.

Results:

- Mean semantic similarity: 0.841
- Median semantic similarity: 0.860
- Summaries above 0.8 similarity: 45 out of 60

### Topic Classification

A balanced supervised classification dataset of 1,044 articles was used.

The classification pipeline uses:

- TF-IDF unigram and bigram features
- LinearSVC
- stratified train-test splitting
- accuracy, precision, recall, F1-score, and confusion-matrix evaluation

Final test performance:

- Accuracy: 0.909
- Macro precision: 0.910
- Macro recall: 0.909
- Macro F1-score: 0.910

### Turing College Requirements Covered

1. Text preprocessing and grammatical-role identification
2. Named Entity Recognition associated with article metadata
3. Investigation of potentially incorrect or ambiguous NER predictions
4. Summarization of 15 articles from each of four categories
5. Semantic similarity analysis between original and summarized text
6. Similarity-score visualization and highest/lowest case investigation
7. Topic prediction for unseen articles
8. PEP8-oriented modular Python code

## Project Status

Completed.
