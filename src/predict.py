"""
양정재 담당 예측 코드

학습된 model/model.joblib을 불러와 test.jsonl을 예측하고 submission.csv를 생성합니다.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

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
SPECIALIST_ACTION_CLUSTERS = {
    "search": {"read_file", "list_directory", "grep_search", "glob_pattern"},
    "write": {"edit_file", "apply_patch"},
    "execute": {"run_bash", "run_tests", "lint_or_typecheck"},
    "dialog": {"ask_user", "plan_task", "respond_only"},
}
SPECIALIST_CONFIDENCE_THRESHOLD = 0.6
SPECIALIST_MARGIN_THRESHOLD = 0.05
SPECIALIST_CLUSTER_THRESHOLDS = {
    "search": (0.4, 0.02),
    "write": (0.6, 0.15),
    "execute": (0.45, 0.15),
}
STACK_TOKEN_MODE = "action_group"


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


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


def stack_token_for_prediction(action: str) -> str:
    group = ACTION_TO_GROUP.get(action, "unknown")
    if STACK_TOKEN_MODE == "action_group":
        return f"STACK_PRED_ACTION_{action} STACK_PRED_GROUP_{group}"
    return f"STACK_PRED_ACTION_{action}"


def append_stack_tokens(X, predicted_actions):
    return [
        f"{text}\n\n{stack_token_for_prediction(action)}"
        for text, action in zip(X, predicted_actions)
    ]


def predict_with_model(model, X):
    if isinstance(model, dict) and model.get("kind") == "stacked_hierarchical":
        base_predictions = predict_with_model(model["base_model"], X)
        stacked_X = append_stack_tokens(X, base_predictions)
        return predict_with_model(model["stack_model"], stacked_X)

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
    for cluster_name, actions in SPECIALIST_ACTION_CLUSTERS.items():
        if top_action not in actions or second_action not in actions:
            continue
        confidence_threshold, margin_threshold = SPECIALIST_CLUSTER_THRESHOLDS.get(
            cluster_name,
            (SPECIALIST_CONFIDENCE_THRESHOLD, SPECIALIST_MARGIN_THRESHOLD),
        )
        if top_score - second_score > margin_threshold:
            return best_action
        specialist_model = specialist_models.get(cluster_name)
        if specialist_model is None or not hasattr(specialist_model, "predict_proba"):
            return best_action
        classes = list(specialist_model.named_steps["clf"].classes_)
        probabilities = specialist_model.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        specialist_action = classes[best_index]
        specialist_confidence = float(probabilities[best_index])
        if specialist_action != top_action and specialist_confidence >= confidence_threshold:
            return specialist_action
        return best_action

    return best_action


def predict(
    test_jsonl_path: str = "data/test.jsonl",
    sample_submission_path: str = "data/sample_submission.csv",
    model_path: str = "model/model.joblib",
    output_path: str = "output/submission.csv",
):
    test_path = Path(test_jsonl_path)
    sample_path = Path(sample_submission_path)
    model_file = Path(model_path)

    if not test_path.exists():
        raise FileNotFoundError(f"테스트 데이터가 없습니다: {test_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_file}. 먼저 python3 -m src.train 실행")

    print("1) test.jsonl 읽는 중...")
    samples = load_jsonl(str(test_path))

    print("2) 전처리 중...")
    X = [make_input_text(sample) for sample in samples]

    print("3) 모델 불러오는 중...")
    model = joblib.load(model_file)

    print("4) 예측 중...")
    predictions = predict_with_model(model, X)

    if sample_path.exists():
        submission = pd.read_csv(sample_path)
        if "action" in submission.columns:
            submission["action"] = predictions
        else:
            submission.iloc[:, -1] = predictions
    else:
        ids = [sample.get("id", i) for i, sample in enumerate(samples)]
        submission = pd.DataFrame({"id": ids, "action": predictions})

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_file, index=False)
    print(f"5) submission 저장 완료: {output_file}")


if __name__ == "__main__":
    predict()
