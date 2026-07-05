"""
양정재 담당 모델 학습 코드

Baseline 분석 반영:
- TfidfVectorizer -> LogisticRegression
- 기존 baseline은 current_prompt만 사용
- 이 프로젝트는 current_prompt + history + session_meta를 함께 사용
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from src.preprocess import make_input_text


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_model() -> Pipeline:
    """
    TF-IDF + 선형 분류 모델입니다.

    LogisticRegression baseline을 참고했지만,
    Mac 로컬 환경에서 빠르게 학습되도록 SGDClassifier(log_loss)를 사용했습니다.
    확률 기반 선형 분류라 Logistic Regression과 비슷한 방식으로 동작합니다.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=80000,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=20,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train(
    train_jsonl_path: str = "data/train.jsonl",
    labels_csv_path: str = "data/train_labels.csv",
    model_output_path: str = "model/model.joblib",
):
    train_path = Path(train_jsonl_path)
    label_path = Path(labels_csv_path)

    if not train_path.exists():
        raise FileNotFoundError(f"학습 데이터가 없습니다: {train_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"라벨 데이터가 없습니다: {label_path}")

    print("1) train.jsonl 읽는 중...")
    samples = load_jsonl(str(train_path))

    print("2) train_labels.csv 읽는 중...")
    labels_df = pd.read_csv(label_path)

    if len(samples) != len(labels_df):
        raise ValueError(f"입력 데이터와 라벨 개수가 다릅니다: {len(samples)} vs {len(labels_df)}")

    print("3) 전처리 중...")
    X = [make_input_text(sample) for sample in samples]
    y = labels_df["action"].astype(str).tolist()

    print("4) 모델 학습 중...")
    model = build_model()
    model.fit(X, y)

    print("5) 학습 완료")
    print("train samples:", len(X))
    print("classes:", list(model.named_steps["clf"].classes_))

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"6) 모델 저장 완료: {output_path}")


if __name__ == "__main__":
    train()
