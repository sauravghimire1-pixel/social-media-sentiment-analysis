import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


def build_tfidf(
    train_texts: pd.Series,
    test_texts: pd.Series,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
):
    """
    Fit a TF-IDF vectorizer on train and transform both splits.
    Returns (vectorizer, X_train, X_test).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def get_top_ngrams(vectorizer: TfidfVectorizer, n: int = 20) -> list:
    """Return top n feature names by index."""
    return vectorizer.get_feature_names_out()[:n].tolist()
