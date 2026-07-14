"""
박진영 담당 제출 실행 파일

데이콘 평가 서버에서 실행되는 파일입니다.
제출용 zip에는 이 script.py, requirements.txt, model/model.joblib이 들어가야 합니다.

로컬 실행:
python3 script.py
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


SPECIALIST_ACTION_CLUSTERS = {
    "search": {"read_file", "list_directory", "grep_search", "glob_pattern"},
    "write": {"edit_file", "apply_patch"},
    "execute": {"run_bash", "run_tests", "lint_or_typecheck"},
    "dialog": {"ask_user", "plan_task", "respond_only"},
}
SPECIALIST_CONFIDENCE_THRESHOLD = 0.6
SPECIALIST_MARGIN_THRESHOLD = 0.05
SPECIALIST_CLUSTER_THRESHOLDS = {
    "execute": (0.45, 0.08),
}


def history_to_text(history: List[Dict[str, Any]], max_history_items: int = 3) -> str:
    if not isinstance(history, list):
        return ""

    selected_history = history[-max_history_items:]
    lines = []

    for item in selected_history:
        if not isinstance(item, dict):
            lines.append(str(item))
            continue

        role = item.get("role", "")

        if role == "user":
            lines.append(f"USER: {item.get('content', '')}")

        elif role == "assistant_action":
            name = item.get("name", "")
            args = json.dumps(item.get("args", {}), ensure_ascii=False, sort_keys=True)
            result = item.get("result_summary", "")
            lines.append(f"ASSISTANT_ACTION: {name} | ARGS: {args} | RESULT: {result}")

        else:
            lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    return "\n".join(lines)


def session_meta_to_text(session_meta: Dict[str, Any]) -> str:
    if not isinstance(session_meta, dict):
        return ""

    workspace = session_meta.get("workspace", {})
    if not isinstance(workspace, dict):
        workspace = {}

    language_mix = workspace.get("language_mix", {})
    if not isinstance(language_mix, dict):
        language_mix = {}

    open_files = workspace.get("open_files", [])
    if not isinstance(open_files, list):
        open_files = []

    return "\n".join([
        f"tier={session_meta.get('user_tier', '')}",
        f"lang_pref={session_meta.get('language_pref', '')}",
        f"git_dirty={workspace.get('git_dirty', '')}",
        f"last_ci={workspace.get('last_ci_status', '')}",
        "open_files=" + " ".join(str(path) for path in open_files),
        "langs=" + " ".join(str(language) for language in language_mix.keys()),
    ])


def last_context_to_text(history: List[Dict[str, Any]]) -> str:
    if not isinstance(history, list):
        return ""

    actions = [
        item for item in history
        if isinstance(item, dict) and item.get("role") == "assistant_action"
    ]
    users = [
        item for item in history
        if isinstance(item, dict) and item.get("role") == "user"
    ]

    lines = []
    if actions:
        last_action = actions[-1]
        lines.append(f"LAST_ACTION_NAME: {last_action.get('name', '')}")
        lines.append(
            "LAST_ACTION_ARGS: "
            + json.dumps(last_action.get("args", {}), ensure_ascii=False, sort_keys=True)
        )
        lines.append(f"LAST_ACTION_RESULT: {last_action.get('result_summary', '')}")
        lines.append(
            "RECENT_ACTION_NAMES: "
            + " ".join(str(action.get("name", "")) for action in actions[-5:])
        )
        lines.append(f"LAST_ACTION_NAME_REPEAT: {last_action.get('name', '')}")
        lines.append(
            "RECENT_ACTION_NAMES_REPEAT: "
            + " ".join(str(action.get("name", "")) for action in actions[-5:])
        )

    if users:
        lines.append(f"LAST_USER: {users[-1].get('content', '')}")

    return "\n".join(lines)


def hint_tokens_to_text(sample: Dict[str, Any]) -> str:
    current_prompt = sample.get("current_prompt", "")
    history_text = history_to_text(sample.get("history", []))
    combined_text = f"{current_prompt} {history_text}".lower()

    session_meta = sample.get("session_meta", {})
    if not isinstance(session_meta, dict):
        session_meta = {}
    workspace = session_meta.get("workspace", {})
    if not isinstance(workspace, dict):
        workspace = {}
    open_files = workspace.get("open_files", [])
    if not isinstance(open_files, list):
        open_files = []

    tokens = []
    if re.search(r"\b(test|tests|pytest|spec|unit|profile)\b", combined_text):
        tokens.append("HINT_TEST")
    if re.search(r"\b(lint|typecheck|type check|mypy|eslint|tsc)\b", combined_text):
        tokens.append("HINT_LINT")
    if re.search(r"\b(grep|search|find|occurrence|pattern)\b", combined_text):
        tokens.append("HINT_SEARCH")
    if re.search(r"\b(read|open|show|inspect|look at)\b", combined_text):
        tokens.append("HINT_READ")
    if re.search(r"\b(edit|change|modify|fix|patch|update|write)\b", combined_text):
        tokens.append("HINT_EDIT")
    if "?" in current_prompt or re.search(
        r"\b(should|would|can you|what|which|어떻게|뭐|질문)\b",
        combined_text,
    ):
        tokens.append("HINT_QUESTION")
    if re.search(r"\b(bash|shell|command|run|execute)\b", combined_text):
        tokens.append("HINT_RUN")
    if re.search(r"\b(directory|folder|list|ls)\b", combined_text):
        tokens.append("HINT_LIST")
    if re.search(r"\b(open|read|show|cat|display|inspect|view)\b", combined_text):
        tokens.append("HINT_DIRECT_OPEN")
    if re.search(
        r"\b(grep|search|find|ripgrep|rg|occurrences?|matches?|pattern|regex)\b",
        combined_text,
    ):
        tokens.append("HINT_FIND_PATTERN")
    if re.search(r"\b(list|ls|tree|directory|folder|files in|what files)\b", combined_text):
        tokens.append("HINT_LIST_DIR")
    if re.search(
        r"[\w./-]+\.(py|tsx?|jsx?|md|json|ya?ml|txt|csv|toml|css|html)",
        combined_text,
    ):
        tokens.append("HINT_PATH_MENTION")
    if re.search(r"\b(patch|diff|apply|hunk|edit_file|apply_patch)\b", combined_text):
        tokens.append("HINT_PATCH_STYLE")
    if re.search(r"\b(fail|failed|error|traceback|exception|red)\b", combined_text):
        tokens.append("HINT_FAILURE")
    if re.search(r"\b(plan|steps|break down|단계|계획)\b", combined_text):
        tokens.append("HINT_PLAN_STEPS")
    if re.search(r"\b(ask|clarify|confirm|question|물어|확인)\b", combined_text):
        tokens.append("HINT_ASK_CLARIFY")
    if current_prompt.strip().lower().startswith(("run ", "execute ", "test ", "lint ")):
        tokens.append("HINT_PROMPT_COMMAND_START")
    if len(open_files) == 1:
        tokens.append("HINT_SINGLE_OPEN_FILE")
    if len(open_files) > 1:
        tokens.append("HINT_MULTI_OPEN_FILES")
    if re.search(r"\b(todo|fixme|function|class|component|hook|schema)\b", combined_text):
        tokens.append("HINT_CODE_SYMBOL")
    if open_files:
        tokens.append("HINT_HAS_OPEN_FILES")
    if workspace.get("git_dirty") is True:
        tokens.append("HINT_GIT_DIRTY")
    if workspace.get("last_ci_status") == "failed":
        tokens.append("HINT_CI_FAILED")

    for path in open_files:
        suffix = str(path).rsplit(".", 1)
        if len(suffix) == 2 and suffix[1]:
            tokens.append("HINT_EXT_" + suffix[1].lower())

    return " ".join(tokens)


def make_input_text(sample: Dict[str, Any]) -> str:
    if not isinstance(sample, dict):
        sample = {}

    current_prompt = sample.get("current_prompt", "")
    hint_text = hint_tokens_to_text(sample)
    last_context_text = last_context_to_text(sample.get("history", []))
    history_text = history_to_text(sample.get("history", []))
    meta_text = session_meta_to_text(sample.get("session_meta", {}))

    return (
        "CURRENT:\n"
        f"{current_prompt}\n\n"
        f"{current_prompt}\n\n"
        f"{current_prompt}\n\n"
        f"{current_prompt}\n\n"
        "HINTS:\n"
        f"{hint_text}\n\n"
        "LAST_CONTEXT:\n"
        f"{last_context_text}\n\n"
        "HISTORY:\n"
        f"{history_text}\n\n"
        "META:\n"
        f"{meta_text}"
    )


def load_jsonl(path: Path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def find_file(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find any of: {candidates}")


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


def main():
    test_path = find_file([
        "data/test.jsonl",
        "./test.jsonl",
        "../data/test.jsonl",
        "/data/test.jsonl",
    ])

    model_path = find_file([
        "model/model.joblib",
        "./model/model.joblib",
    ])

    sample_submission_path = None
    for candidate in [
        "data/sample_submission.csv",
        "./sample_submission.csv",
        "../data/sample_submission.csv",
        "/data/sample_submission.csv",
    ]:
        if Path(candidate).exists():
            sample_submission_path = Path(candidate)
            break

    samples = load_jsonl(test_path)
    X = [make_input_text(sample) for sample in samples]

    model = joblib.load(model_path)
    predictions = predict_with_model(model, X)

    if sample_submission_path is not None:
        submission = pd.read_csv(sample_submission_path)
        if "action" in submission.columns:
            submission["action"] = predictions
        else:
            submission.iloc[:, -1] = predictions
    else:
        ids = [sample.get("id", i) for i, sample in enumerate(samples)]
        submission = pd.DataFrame({"id": ids, "action": predictions})

    output_path = Path("output/submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
