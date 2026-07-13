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
