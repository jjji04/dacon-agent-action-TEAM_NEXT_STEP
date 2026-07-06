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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.preprocess import make_input_text


ACTION_TO_GROUP = {
    "list_directory": "search_read",
    "glob_pattern": "search_read",
    "grep_search": "search_read",
    "read_file": "search_read",
    "edit_file": "write_edit",
    "write_file": "write_edit",
    "apply_patch": "write_edit",
    "run_bash": "execute_check",
    "run_tests": "execute_check",
    "lint_or_typecheck": "execute_check",
    "respond_only": "dialog_plan",
    "ask_user": "dialog_plan",
    "plan_task": "dialog_plan",
    "web_search": "dialog_plan",
}


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

    SGDClassifier(log_loss)는 빠른 로지스틱 회귀 계열 모델입니다.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=3e-6,
            l1_ratio=0.25,
            class_weight="balanced",
            max_iter=30,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_hierarchical_model(X, y):
    groups = [ACTION_TO_GROUP[label] for label in y]
    group_model = build_model()
    group_model.fit(X, groups)

    action_models = {}
    group_to_actions = {}
    for group in sorted(set(groups)):
        indexes = [i for i, item in enumerate(groups) if item == group]
        group_X = [X[i] for i in indexes]
        group_y = [y[i] for i in indexes]
        group_to_actions[group] = sorted(set(group_y))

        model = build_model()
        model.fit(group_X, group_y)
        action_models[group] = model

    return {
        "kind": "hierarchical",
        "model_type": "sgd",
        "group_model": group_model,
        "action_models": action_models,
        "action_to_group": ACTION_TO_GROUP,
        "group_to_actions": group_to_actions,
    }


def predict_with_model(model, X):
    if isinstance(model, dict) and model.get("kind") == "hierarchical":
        predicted_groups = model["group_model"].predict(X)
        predictions = []
        for text, group in zip(X, predicted_groups):
            action_model = model["action_models"].get(str(group))
            if action_model is None:
                action_model = next(iter(model["action_models"].values()))
            predictions.append(action_model.predict([text])[0])
        return predictions
    return model.predict(X)


def score_candidate(name, model, X_valid, y_valid):
    predictions = predict_with_model(model, X_valid)
    score = accuracy_score(y_valid, predictions)
    print(f"validation {name} accuracy: {score:.6f}")
    return score


def build_validated_model(X, y, validation_size: float = 0.2):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=42,
        stratify=y,
    )

    print("4-1) 단일 모델 검증 학습 중...")
    single_model = build_model()
    single_model.fit(X_train, y_train)
    single_score = score_candidate("single_sgd", single_model, X_valid, y_valid)

    print("4-2) 계층형 모델 검증 학습 중...")
    hierarchical_model = build_hierarchical_model(X_train, y_train)
    hierarchical_score = score_candidate("hierarchical_sgd", hierarchical_model, X_valid, y_valid)

    if hierarchical_score >= single_score:
        print("4-3) 계층형 모델 선택 후 전체 데이터 재학습 중...")
        return build_hierarchical_model(X, y), {
            "single_accuracy": single_score,
            "hierarchical_accuracy": hierarchical_score,
            "selected": "hierarchical_sgd",
        }

    print("4-3) 단일 모델 선택 후 전체 데이터 재학습 중...")
    final_model = build_model()
    final_model.fit(X, y)
    return final_model, {
        "single_accuracy": single_score,
        "hierarchical_accuracy": hierarchical_score,
        "selected": "single_sgd",
    }


def train(
    train_jsonl_path: str = "data/train.jsonl",
    labels_csv_path: str = "data/train_labels.csv",
    model_output_path: str = "model/model.joblib",
    validation_size: float = 0.2,
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

    unknown_labels = sorted(set(y) - set(ACTION_TO_GROUP))
    if unknown_labels:
        raise ValueError(f"그룹 매핑에 없는 라벨이 있습니다: {unknown_labels}")

    print("4) 모델 학습 및 검증 중...")
    if validation_size > 0:
        model, metrics = build_validated_model(X, y, validation_size=validation_size)
    else:
        model = build_hierarchical_model(X, y)
        metrics = {"selected": "hierarchical_sgd"}

    print("5) 학습 완료")
    print("train samples:", len(X))
    print("selected model:", metrics["selected"])
    if isinstance(model, dict):
        print("groups:", sorted(model["action_models"]))
    else:
        print("classes:", list(model.named_steps["clf"].classes_))

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"6) 모델 저장 완료: {output_path}")


if __name__ == "__main__":
    train()
