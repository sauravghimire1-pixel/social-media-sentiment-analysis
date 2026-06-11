# Social Media Sentiment Analysis

Classification of social media posts and reviews as positive, negative, or neutral using natural language processing and machine learning.

## Project Overview

This project builds a sentiment classifier trained on social media text data. It covers text preprocessing, feature extraction (TF-IDF and embeddings), and comparison of multiple classification models.

## Dataset

| Source | Description |
|--------|-------------|
| [Kaggle — Twitter Sentiment Analysis](https://www.kaggle.com/datasets/kazanova/sentiment140) | 1.6M tweets labelled positive/negative |
| [Kaggle — Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) | Product reviews with star ratings |
| [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) | Sentences from Amazon, Yelp, IMDb |

> **Note:** Raw data files are not committed to this repo. Download sources above and place in `data/raw/`.

## Project Structure

```
sentiment-analysis-social-media/
├── data/
│   ├── raw/              # Source data (gitignored)
│   └── processed/        # Cleaned, tokenized data
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis & class balance
│   ├── 02_features.ipynb     # Text preprocessing & feature extraction
│   ├── 03_modelling.ipynb    # Model training & evaluation
│   └── 04_insights.ipynb     # Error analysis & visualizations
├── src/
│   ├── preprocessing.py      # Text cleaning utilities
│   ├── features.py           # TF-IDF & embedding feature builders
│   └── evaluation.py         # Classification metrics
├── reports/
│   └── final_report.pdf
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/yourusername/sentiment-analysis-social-media.git
cd sentiment-analysis-social-media
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet
jupyter notebook notebooks/01_eda.ipynb
```

## Phases

| Phase | Focus | Milestone |
|-------|-------|-----------|
| 1 | Data collection & EDA | Clean dataset + class distribution analysis |
| 2 | Text preprocessing & features | TF-IDF matrix or embeddings ready |
| 3 | Modelling & evaluation | Best classifier selected, metrics documented |
| 4 | Insights & presentation | Confusion matrix, word clouds, error analysis |

## Results

_To be updated after modelling is complete._

## Tech Stack

Python · pandas · scikit-learn · NLTK · TF-IDF · Logistic Regression · Naive Bayes · matplotlib · seaborn · Jupyter

## Author

Your Name — [GitHub](https://github.com/yourusername)
