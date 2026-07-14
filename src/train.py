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


ACTION_MODEL_ALPHA = {
    "dialog_plan": 1e-4,
    "execute_check": 3e-5,
    "search_read": 3e-5,
    "write_edit": 1e-5,
}

SPECIALIST_ACTION_CLUSTERS = {
    "search": {"read_file", "list_directory", "grep_search", "glob_pattern"},
    "write": {"edit_file", "apply_patch"},
    "execute": {"run_bash", "run_tests", "lint_or_typecheck"},
    "dialog": {"ask_user", "plan_task", "respond_only"},
}
SPECIALIST_CONFIDENCE_THRESHOLD = 0.6
SPECIALIST_MARGIN_THRESHOLD = 0.05


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_model(alpha: float = 3e-6) -> Pipeline:
    """
    TF-IDF + 선형 분류 모델입니다.

    SGDClassifier(log_loss)는 빠른 로지스틱 회귀 계열 모델입니다.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=180000,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=0.25,
            class_weight="balanced",
            max_iter=30,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_search_read_model(alpha: float = 3e-5) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_features=220000,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=0.25,
            class_weight="balanced",
            max_iter=30,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_write_edit_model(alpha: float = 1e-5) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_features=220000,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=0.25,
            class_weight="balanced",
            max_iter=30,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_specialist_model(alpha: float = 3e-5) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_features=160000,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=0.25,
            class_weight="balanced",
            max_iter=30,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_specialist_models(X, y):
    specialist_models = {}
    for cluster_name, actions in SPECIALIST_ACTION_CLUSTERS.items():
        indexes = [i for i, label in enumerate(y) if label in actions]
        cluster_y = [y[i] for i in indexes]
        if len(set(cluster_y)) < 2:
            continue
        cluster_X = [X[i] for i in indexes]
        model = build_specialist_model()
        model.fit(cluster_X, cluster_y)
        specialist_models[cluster_name] = model
    return specialist_models


def build_hierarchical_model(X, y):
    groups = [ACTION_TO_GROUP[label] for label in y]
    group_model = build_model(alpha=1e-6)
    group_model.fit(X, groups)

    action_models = {}
    group_to_actions = {}
    for group in sorted(set(groups)):
        indexes = [i for i, item in enumerate(groups) if item == group]
        group_X = [X[i] for i in indexes]
        group_y = [y[i] for i in indexes]
        group_to_actions[group] = sorted(set(group_y))

        alpha = ACTION_MODEL_ALPHA.get(group, 3e-6)
        if group == "search_read":
            model = build_search_read_model(alpha=alpha)
        elif group == "write_edit":
            model = build_write_edit_model(alpha=alpha)
        else:
            model = build_model(alpha=alpha)
        model.fit(group_X, group_y)
        action_models[group] = model

    return {
        "kind": "hierarchical",
        "model_type": "sgd",
        "group_model": group_model,
        "action_models": action_models,
        "specialist_models": build_specialist_models(X, y),
        "action_to_group": ACTION_TO_GROUP,
        "group_to_actions": group_to_actions,
    }


def candidate_actions_from_text(text: str):
    actions = None

    def add(candidate):
        nonlocal actions
        actions = set(candidate) if actions is None else actions | set(candidate)

    if "HINT_TEST" in text:
        add(["run_tests", "run_bash", "lint_or_typecheck"])
    if "HINT_LINT" in text:
        add(["lint_or_typecheck", "run_tests", "run_bash"])
    if "HINT_FIND_PATTERN" in text or "HINT_SEARCH" in text:
        add(["grep_search", "glob_pattern", "read_file", "list_directory"])
    if "HINT_LIST_DIR" in text or "HINT_LIST" in text:
        add(["list_directory", "glob_pattern", "grep_search"])
    if "HINT_DIRECT_OPEN" in text or "HINT_READ" in text:
        add(["read_file", "grep_search", "glob_pattern", "list_directory"])
    if "HINT_PATCH_STYLE" in text:
        add(["apply_patch", "edit_file"])
    if "HINT_EDIT" in text:
        add(["edit_file", "apply_patch", "write_file"])
    if "HINT_PLAN_STEPS" in text:
        add(["plan_task", "respond_only"])
    if "HINT_ASK_CLARIFY" in text or "HINT_QUESTION" in text:
        add(["ask_user", "respond_only", "plan_task"])

    return actions


def predict_with_model(model, X):
    if isinstance(model, dict) and model.get("kind") == "hierarchical":
        group_model = model["group_model"]
        if hasattr(group_model, "predict_proba"):
            group_classes = list(group_model.named_steps["clf"].classes_)
            group_probs = group_model.predict_proba(X)
            predictions = []
            for text, probs in zip(X, group_probs):
                top_groups = sorted(
                    zip(group_classes, probs),
                    key=lambda item: item[1],
                    reverse=True,
                )[:2]
                best_action = None
                best_score = -1.0
                candidate_actions = candidate_actions_from_text(text)
                action_scores = []
                for group, group_prob in top_groups:
                    action_model = model["action_models"].get(str(group))
                    if action_model is None or not hasattr(action_model, "predict_proba"):
                        continue
                    action_classes = list(action_model.named_steps["clf"].classes_)
                    action_probs = action_model.predict_proba([text])[0]
                    for action, action_prob in zip(action_classes, action_probs):
                        score = float(group_prob) * float(action_prob)
                        if candidate_actions is not None and action not in candidate_actions:
                            score *= 0.9
                        action_scores.append((action, score))
                        if score > best_score:
                            best_score = score
                            best_action = action
                action_scores.sort(key=lambda item: item[1], reverse=True)
                if len(action_scores) >= 2:
                    best_action = apply_specialist_correction(
                        model,
                        text,
                        action_scores,
                        best_action,
                    )
                if best_action is None:
                    best_action = model["action_models"][str(top_groups[0][0])].predict([text])[0]
                predictions.append(best_action)
            return predictions

        predicted_groups = group_model.predict(X)
        predictions = []
        for text, group in zip(X, predicted_groups):
            action_model = model["action_models"].get(str(group))
            if action_model is None:
                action_model = next(iter(model["action_models"].values()))
            predictions.append(action_model.predict([text])[0])
        return predictions
    return model.predict(X)


def apply_specialist_correction(model, text, action_scores, best_action):
    specialist_models = model.get("specialist_models", {})
    if not specialist_models:
        return best_action

    top_action, top_score = action_scores[0]
    second_action, second_score = action_scores[1]
    if top_action != best_action:
        return best_action
    if top_score - second_score > SPECIALIST_MARGIN_THRESHOLD:
        return best_action

    for cluster_name, actions in SPECIALIST_ACTION_CLUSTERS.items():
        if top_action not in actions or second_action not in actions:
            continue
        specialist_model = specialist_models.get(cluster_name)
        if specialist_model is None or not hasattr(specialist_model, "predict_proba"):
            return best_action
        classes = list(specialist_model.named_steps["clf"].classes_)
        probabilities = specialist_model.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        specialist_action = classes[best_index]
        specialist_confidence = float(probabilities[best_index])
        if specialist_action != top_action and specialist_confidence >= SPECIALIST_CONFIDENCE_THRESHOLD:
            return specialist_action
        return best_action

    return best_action


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
