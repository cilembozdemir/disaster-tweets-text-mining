"""Classify disaster tweets with an interpretable TF-IDF model."""

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "disaster_tweets.csv"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    """Apply light cleaning while keeping useful words and hashtags."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9#' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    tweets = pd.read_csv(DATA_PATH)
    tweets["clean_text"] = tweets["text"].fillna("").map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        tweets["clean_text"],
        tweets["target"],
        test_size=0.20,
        stratify=tweets["target"],
        random_state=RANDOM_STATE,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train.to_frame(), y_train)
    baseline_predictions = baseline.predict(X_test.to_frame())

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rows": int(len(tweets)),
        "disaster_share": round(float(tweets["target"].mean()), 3),
        "baseline": evaluate(y_test, baseline_predictions),
        "logistic_regression": evaluate(y_test, predictions),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).T.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Disaster tweet classification")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    terms = pd.DataFrame({"term": feature_names, "coefficient": coefficients})
    influential = pd.concat([terms.nsmallest(12, "coefficient"), terms.nlargest(12, "coefficient")])
    influential.to_csv(OUTPUT_DIR / "influential_terms.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
