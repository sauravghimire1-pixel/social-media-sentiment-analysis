import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, mentions, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                  # remove mentions
    text = re.sub(r"#\w+", "", text)                  # remove hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove stopwords from a cleaned text string."""
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def lemmatize(text: str) -> str:
    """Lemmatize each token in the text."""
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())


def full_pipeline(text: str) -> str:
    """Apply clean → remove stopwords → lemmatize."""
    return lemmatize(remove_stopwords(clean_text(text)))


def apply_pipeline(df: pd.DataFrame, text_col: str, out_col: str = "clean_text") -> pd.DataFrame:
    """Apply the full text pipeline to a dataframe column."""
    df[out_col] = df[text_col].astype(str).apply(full_pipeline)
    return df
