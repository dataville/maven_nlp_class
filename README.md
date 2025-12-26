# Natural Language Processing in Python

A comprehensive implementation of traditional and modern NLP techniques, demonstrating the evolution from classical machine learning to transformer-based approaches.

## Project Overview

This project implements NLP techniques across three major areas: text preprocessing and vectorization, traditional machine learning models, and modern transformer-based architectures using Hugging Face. The implementations use three datasets: children's books, movie reviews, and movie metadata with director information.

## Repository Structure

```
├── section3_assignments_working.ipynb    # Text preprocessing and vectorization
├── section4_assignments_working.ipynb    # Traditional ML NLP (VADER, Naive Bayes, NMF)
└── section7_assignments_working.ipynb    # Modern NLP with transformers
```

## Technical Implementation

### Section 3: Text Preprocessing and Vectorization

**Dataset:** Children's books (`childrens_books.csv`)

**Preprocessing Pipeline:**
- Text normalization with pandas: lowercase conversion, Unicode character removal (`\xa0`), punctuation stripping
- Tokenization and lemmatization with spaCy (`en_core_web_sm`)
- Custom preprocessing function `clean_and_normalize` from `maven_text_preprocessing`

**Vectorization:**
- **CountVectorizer:** Implemented with stop word removal and minimum document frequency threshold (10%)
- **TF-IDF Vectorizer:** Applied with identical parameters for comparison
- Generated document-term matrices and identified top 10 most/least common terms
- Visualized term frequencies using horizontal bar charts with matplotlib

### Section 4: Traditional Machine Learning NLP

**Dataset:** Movie reviews with ratings, genres, directors, and critic consensus (`movie_reviews.csv`)

#### 1. Sentiment Analysis (VADER)
- Applied VADER (Valence Aware Dictionary and sEntiment Reasoner) to movie descriptions
- Extracted compound sentiment scores ranging from -1 to +1
- Identified movies with highest positive sentiment (e.g., "Breakthrough": 0.9915) and most negative sentiment (e.g., "Charlie Says": -0.9706)

#### 2. Text Classification
- **Target Variable:** Director gender prediction (male/female)
- **Features:** Cleaned and normalized movie descriptions using spaCy
- **Models Implemented:**
  - Multinomial Naive Bayes
  - Logistic Regression
- Vectorization with TF-IDF (min_df=0.1)
- Model comparison using accuracy scores and classification reports
- Identified movies most likely to be directed by women based on model predictions

#### 3. Topic Modeling
- **Algorithm:** Non-Negative Matrix Factorization (NMF)
- **Parameters:** 6 components, 500 max iterations, random_state=42
- **Preprocessing:** TF-IDF vectorization with min_df=0.02, max_df=0.2
- Generated interpretable topic labels: 
  - Family films
  - True stories
  - Friends narratives
  - Award winners
  - Adventure
  - Horror
- Custom `display_topics` function to extract top 10 terms per topic

### Section 7: Modern NLP with Transformers

**Datasets:** 
- Movie reviews with VADER sentiment scores (`movie_reviews_sentiment.csv`)
- Children's books (`childrens_books.csv`)

All transformer models configured with Metal Performance Shaders (MPS) device acceleration on Apple Silicon.

#### 1. Sentiment Analysis with Transformers
- Compared transformer-based sentiment analysis against VADER baseline
- Pipeline implementation for zero-setup inference

#### 2. Named Entity Recognition (NER)
- **Model:** `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Aggregation Strategy:** SIMPLE
- Applied to children's book descriptions
- Extracted person entities (PER) and filtered to exclude authors
- Generated unique list of character names from book descriptions

#### 3. Zero-Shot Classification
- **Model:** `facebook/bart-large-mnli`
- **Categories:** Adventure & fantasy, animals & nature, mystery, humor, non-fiction
- Applied to children's book descriptions without training data
- Validated classification results through manual spot-checking

#### 4. Text Summarization
- **Model:** `facebook/bart-large-cnn`
- **Parameters:** min_length=10, max_length=50, early_stopping=True, length_penalty=0.8
- Generated abstractive summaries of book descriptions
- Example: "Where the Wild Things Are" description (78 words) → summary (33 words)

#### 5. Document Similarity
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Technique:** Feature extraction to generate 384-dimensional embeddings
- Computed cosine similarity between "Harry Potter and the Sorcerer's Stone" and all books
- Identified top 5 most similar books:
  1. Harry Potter and the Sorcerer's Stone (1.0000)
  2. Harry Potter and the Prisoner of Azkaban (0.8726)
  3. Harry Potter and the Chamber of Secrets (0.8554)
  4. The Witches (0.7991)
  5. The Wonderful Wizard of Oz (0.7885)

## Technologies and Libraries

### Core NLP Libraries
- **spaCy 3.8.0:** Tokenization, lemmatization, linguistic annotations
- **transformers (Hugging Face):** Pre-trained transformer models and pipelines
- **vaderSentiment:** Rule-based sentiment analysis

### Machine Learning
- **scikit-learn:** 
  - Vectorization: `CountVectorizer`, `TfidfVectorizer`
  - Models: `MultinomialNB`, `LogisticRegression`, `NMF`
  - Metrics: `cosine_similarity`
- **PyTorch:** Backend for transformer models

### Data Processing and Visualization
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical operations and array handling
- **matplotlib:** Data visualization

### Specific Models Used
| Task | Model | Source |
|------|-------|--------|
| NER | BERT-large-cased CoNLL03 | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| Zero-Shot | BART-large MNLI | `facebook/bart-large-mnli` |
| Summarization | BART-large CNN | `facebook/bart-large-cnn` |
| Embeddings | MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` |

## Environment Setup

Three separate conda environments were used:

```bash
# Text preprocessing environment
conda create -n nlp_preprocessing python=3.12
conda activate nlp_preprocessing
pip install pandas spacy matplotlib
python -m spacy download en_core_web_sm

# Traditional ML environment  
conda create -n nlp_machine_learning python=3.12
conda activate nlp_machine_learning
pip install pandas scikit-learn vaderSentiment spacy

# Transformers environment
conda create -n nlp_transformers python=3.12
conda activate nlp_transformers
pip install pandas transformers torch
```

## Key Findings

### Sentiment Analysis Comparison
VADER and transformer-based approaches showed different strengths:
- VADER: Fast, interpretable, good for social media text
- Transformers: More nuanced understanding, better context handling

### Text Classification
Logistic Regression and Naive Bayes both achieved competitive accuracy in predicting director gender from movie descriptions, demonstrating that linguistic patterns correlate with directorial gender.

### Topic Modeling
NMF successfully identified coherent topics from unlabeled movie descriptions, with clear thematic separation between genres (family films vs. horror, true stories vs. adventure).

### Document Similarity
Sentence transformers effectively captured semantic similarity, correctly identifying Harry Potter sequels as most similar to the first book, followed by other magical adventure stories (The Witches, Wizard of Oz).

## Implementation Details

### Preprocessing Best Practices
1. Lowercase normalization before vectorization
2. Unicode character handling (`\xa0` removal)
3. Punctuation stripping
4. Stop word removal for dimensionality reduction
5. Minimum document frequency thresholds to eliminate rare terms

### Vectorization Strategy
- Used both Count and TF-IDF vectorization for comparison
- TF-IDF consistently provided better features for classification tasks
- Document frequency thresholds (min_df, max_df) crucial for performance

### Model Configuration
- All transformers configured with `logging.set_verbosity_error()` to reduce output
- MPS device acceleration on Apple Silicon: `device='mps'`
- Pipeline abstraction for simplified inference
- Aggregation strategies in NER to merge subword tokens

## Data Characteristics

### Children's Books Dataset
- 100 books ranked by popularity
- Features: Title, Author, Year, Rating (1-5), Description
- Publication years: 1947-2014
- Average rating: 4.3/5.0

### Movie Reviews Dataset  
- 160 movies from 2019
- Features: Title, Rating, Genre, Release Date, Description, Directors, Director Gender, Tomatometer Rating, Audience Rating, Critics Consensus
- Includes VADER sentiment scores in later versions
- Director gender distribution provides binary classification target

## Performance Considerations

### Computational Requirements
- spaCy processing: Minimal (CPU sufficient)
- VADER sentiment: Extremely fast (rule-based)
- Transformer inference: GPU/MPS recommended for batch processing
- Feature extraction: Most computationally intensive (384-dim embeddings for 100+ documents)

### Memory Usage
- Document-term matrices kept sparse for efficiency
- Transformer models loaded individually (1-2GB each)
- Embedding matrices: ~380KB for 100 documents (float32)

## Jupyter Notebook Execution

All notebooks use `pandas.set_option('display.max_colwidth', None)` for full text display and include inline visualizations with matplotlib.

Processing order:
1. Data loading
2. Text cleaning
3. Vectorization/Model application  
4. Analysis and visualization
5. Results interpretation

## Python Version

All implementations use Python 3.12.11 via miniforge3 (Apple Silicon).

## Course Context

These implementations are based on assignments from the Natural Language Processing in Python course, covering the progression from foundational text processing to state-of-the-art transformer architectures.