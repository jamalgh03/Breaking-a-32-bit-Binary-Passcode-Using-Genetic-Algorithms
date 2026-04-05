import re
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def eval_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision(macro)": prec,
        "Recall(macro)": rec,
        "F1(macro)": f1,
    }


def main():
    # 1) Load dataset (AG News) with fallback for reliability
    try:
        ds = load_dataset("wangrongsheng/ag_news")
    except Exception:
        ds = load_dataset("ag_news")

    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    # 2) Clean text
    train_df["text"] = train_df["text"].astype(str).apply(clean_text)
    test_df["text"] = test_df["text"].astype(str).apply(clean_text)

    X = train_df["text"].tolist()
    y = train_df["label"].tolist()

    # 3) split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # 5) Models
    lr = LogisticRegression(
        max_iter=2000
    )
    dt = DecisionTreeClassifier(
        max_depth=25,
        random_state=42
    )

    # 6) Train
    lr.fit(X_train_tfidf, y_train)
    dt.fit(X_train_tfidf, y_train)

    # 7) Predict (validation)
    lr_pred = lr.predict(X_val_tfidf)
    dt_pred = dt.predict(X_val_tfidf)

    # 8) Metrics
    results = []
    results.append(eval_model("Logistic Regression", y_val, lr_pred))
    results.append(eval_model("Decision Tree", y_val, dt_pred))
    results_df = pd.DataFrame(results)

    print("\n=== Results (Validation) ===")
    print(results_df.to_string(index=False))

    print("\n=== Confusion Matrix (Logistic Regression) ===")
    print(confusion_matrix(y_val, lr_pred))

    print("\n=== Confusion Matrix (Decision Tree) ===")
    print(confusion_matrix(y_val, dt_pred))

    print("\n=== Classification Report (Logistic Regression) ===")
    print(classification_report(y_val, lr_pred, target_names=[LABELS[i] for i in range(4)]))

    print("\n=== Classification Report (Decision Tree) ===")
    print(classification_report(y_val, dt_pred, target_names=[LABELS[i] for i in range(4)]))

    # 9) Graphs (bar charts for metrics)
    metrics_cols = ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)"]
    results_df.set_index("Model")[metrics_cols].plot(kind="bar")
    plt.title("Model Comparison on AG News (Validation)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 10) Plot Decision Tree (top levels only)
    plt.figure(figsize=(18, 10))
    plot_tree(
        dt,
        max_depth=3,
        feature_names=None,
        class_names=[LABELS[i] for i in range(4)],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree (Top Levels Only)")
    plt.tight_layout()
    plt.show()

    # 11) Evaluate on official test set
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    X_test_tfidf = tfidf.transform(X_test)

    lr_test_pred = lr.predict(X_test_tfidf)
    dt_test_pred = dt.predict(X_test_tfidf)

    print("\n=== Official Test Set Results ===")
    test_results = pd.DataFrame([
        eval_model("Logistic Regression (TEST)", y_test, lr_test_pred),
        eval_model("Decision Tree (TEST)", y_test, dt_test_pred),
    ])
    print(test_results.to_string(index=False))

    print("\n=== Confusion Matrix (Logistic Regression - TEST) ===")
    print(confusion_matrix(y_test, lr_test_pred))

    print("\n=== Confusion Matrix (Decision Tree - TEST) ===")
    print(confusion_matrix(y_test, dt_test_pred))
              

if __name__ == "__main__":
    main()
