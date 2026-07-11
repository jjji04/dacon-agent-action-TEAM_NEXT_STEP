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


def history_to_text(history: List[Dict[str, Any]], max_history_items: int = 10) -> str:
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

    # current_prompt 전용 의도 힌트입니다.
    # history에 섞인 과거 단어가 아니라, 지금 사용자 요청에서 직접 드러난 행동만 별도 강조합니다.
    current_lower = str(current_prompt).lower()
    if re.search(r"[\w./-]+\.(py|tsx?|jsx?|md|json|ya?ml|txt|csv|toml|css|html)", current_lower):
        tokens.append("HINT_PROMPT_HAS_FILE_PATH")
    if re.search(r"\*\.[a-zA-Z0-9]+|\*\*/|glob", current_lower):
        tokens.append("HINT_PROMPT_GLOB_INTENT")
    if re.search(r"\b(rg|grep|ripgrep|search for|find occurrences|occurrences?|matches?|regex)\b", current_lower):
        tokens.append("HINT_PROMPT_SEARCH_INTENT")
    if re.search(r"\b(list|ls|tree|directory|folder|files in|what files)\b", current_lower):
        tokens.append("HINT_PROMPT_LIST_INTENT")
    if re.search(r"\b(read|open|show|inspect|view|cat|look at)\b", current_lower):
        tokens.append("HINT_PROMPT_READ_INTENT")
    if re.search(r"\b(edit|modify|change|update|fix|refactor|replace|rename|adjust)\b", current_lower):
        tokens.append("HINT_PROMPT_EDIT_INTENT")
    if re.search(r"\b(create file|new file|write file|save file|generate file|add file|write_file)\b", current_lower):
        tokens.append("HINT_PROMPT_WRITE_INTENT")
    if re.search(r"\b(apply patch|apply_patch|diff|hunk|patch file|patching|unified diff)\b", current_lower):
        tokens.append("HINT_PROMPT_PATCH_INTENT")
    if re.search(r"\b(pytest|run tests?|test suite|unit tests?|npm test|pnpm test|yarn test)\b", current_lower):
        tokens.append("HINT_PROMPT_TEST_INTENT")
    if re.search(r"\b(lint|typecheck|type check|mypy|eslint|tsc|ruff|flake8)\b", current_lower):
        tokens.append("HINT_PROMPT_LINT_INTENT")
    if re.search(r"\b(bash|shell|command|terminal|run command|execute command)\b", current_lower):
        tokens.append("HINT_PROMPT_BASH_INTENT")
    if re.search(r"\b(plan|steps|step by step|roadmap|todo|break down)\b|단계|계획|순서", current_lower):
        tokens.append("HINT_PROMPT_PLAN_INTENT")
    if re.search(r"\b(clarify|confirm|ask user|need more info|question)\b|물어|확인", current_lower):
        tokens.append("HINT_PROMPT_ASK_INTENT")
    if re.search(r"\b(explain|summarize|tell me|what is|why|how does)\b|알려줘|설명|요약", current_lower):
        tokens.append("HINT_PROMPT_RESPOND_INTENT")
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
        tokens.append("HINT_READ_FILE_PATH_STRONG")
    if re.search(r"\*\.[a-zA-Z0-9]+|\*\*/|glob", combined_text):
        tokens.append("HINT_GLOB_PATTERN_STRONG")
    if re.search(r"\b(rg|grep|ripgrep|occurrences?|matches?|search for|find occurrences)\b", combined_text):
        tokens.append("HINT_GREP_SEARCH_STRONG")
    if re.search(r"\b(ls|tree|list directory|folder structure|directory tree|what files|files in)\b", combined_text):
        tokens.append("HINT_LIST_DIRECTORY_STRONG")
    if re.search(r"\b(patch|diff|apply|hunk|edit_file|apply_patch)\b", combined_text):
        tokens.append("HINT_PATCH_STYLE")
    if re.search(r"\b(apply patch|apply_patch|diff|hunk|patch file|patching|unified diff)\b", combined_text):
        tokens.append("HINT_APPLY_PATCH_STRONG")
    if re.search(r"\b(edit|modify|change|update|fix|refactor|replace|rename|adjust)\b", combined_text):
        tokens.append("HINT_EDIT_FILE_STRONG")
    if re.search(r"\b(create file|new file|write file|save file|generate file|add file|write_file)\b", combined_text):
        tokens.append("HINT_WRITE_FILE_STRONG")
    if re.search(r"\b(fail|failed|error|traceback|exception|red)\b", combined_text):
        tokens.append("HINT_FAILURE")
    if re.search(r"\b(plan|steps|break down|단계|계획)\b", combined_text):
        tokens.append("HINT_PLAN_STEPS")
    if re.search(r"\b(ask|clarify|confirm|question|물어|확인)\b", combined_text):
        tokens.append("HINT_ASK_CLARIFY")
    if current_prompt.strip().lower().startswith(("run ", "execute ", "test ", "lint ")):
        tokens.append("HINT_PROMPT_COMMAND_START")

    actions = [
        item for item in sample.get("history", [])
        if isinstance(item, dict) and item.get("role") == "assistant_action"
    ]
    if actions:
        last_action = str(actions[-1].get("name", "")).strip()
        if last_action:
            tokens.append("HINT_LAST_ACTION_" + last_action.upper())
            if last_action in {"list_directory", "glob_pattern", "grep_search", "read_file"}:
                tokens.append("HINT_LAST_GROUP_SEARCH_READ")
            elif last_action in {"edit_file", "write_file", "apply_patch"}:
                tokens.append("HINT_LAST_GROUP_WRITE_EDIT")
            elif last_action in {"run_bash", "run_tests", "lint_or_typecheck"}:
                tokens.append("HINT_LAST_GROUP_EXECUTE_CHECK")
            elif last_action in {"respond_only", "ask_user", "plan_task", "web_search"}:
                tokens.append("HINT_LAST_GROUP_DIALOG_PLAN")

        recent_names = [str(action.get("name", "")).strip() for action in actions[-4:]]
        recent_names = [name for name in recent_names if name]
        for name in recent_names:
            tokens.append("HINT_RECENT_ACTION_" + name.upper())
        if len(set(recent_names)) == 1 and recent_names:
            tokens.append("HINT_REPEAT_SAME_ACTION_" + recent_names[-1].upper())

        action_to_group = {
            "list_directory": "SEARCH_READ",
            "glob_pattern": "SEARCH_READ",
            "grep_search": "SEARCH_READ",
            "read_file": "SEARCH_READ",
            "edit_file": "WRITE_EDIT",
            "write_file": "WRITE_EDIT",
            "apply_patch": "WRITE_EDIT",
            "run_bash": "EXECUTE_CHECK",
            "run_tests": "EXECUTE_CHECK",
            "lint_or_typecheck": "EXECUTE_CHECK",
            "respond_only": "DIALOG_PLAN",
            "ask_user": "DIALOG_PLAN",
            "plan_task": "DIALOG_PLAN",
            "web_search": "DIALOG_PLAN",
        }

        # 최근 Action의 순서/전이를 입력 힌트로 추가합니다.
        # 예측을 강제로 바꾸지 않고, 모델이 자주 이어지는 행동 패턴을 학습하게 합니다.
        if len(recent_names) >= 2:
            last_two = recent_names[-2:]
            tokens.append("HINT_LAST_ACTION_PAIR_" + "__".join(name.upper() for name in last_two))
            for prev_name, next_name in zip(recent_names[:-1], recent_names[1:]):
                tokens.append("HINT_RECENT_ACTION_PAIR_" + prev_name.upper() + "__" + next_name.upper())

            last_two_groups = [action_to_group.get(name, "OTHER") for name in last_two]
            tokens.append("HINT_LAST_GROUP_PAIR_" + "__".join(last_two_groups))

            recent_groups = [action_to_group.get(name, "OTHER") for name in recent_names]
            for prev_group, next_group in zip(recent_groups[:-1], recent_groups[1:]):
                tokens.append("HINT_RECENT_GROUP_PAIR_" + prev_group + "__" + next_group)

            if last_two_groups == ["SEARCH_READ", "WRITE_EDIT"]:
                tokens.append("HINT_FLOW_SEARCH_TO_EDIT")
            if last_two_groups == ["WRITE_EDIT", "EXECUTE_CHECK"]:
                tokens.append("HINT_FLOW_EDIT_TO_EXEC")
            if last_two_groups == ["EXECUTE_CHECK", "WRITE_EDIT"]:
                tokens.append("HINT_FLOW_EXEC_TO_EDIT")
            if last_two_groups == ["SEARCH_READ", "SEARCH_READ"]:
                tokens.append("HINT_FLOW_SEARCH_CHAIN")


        last_result = str(actions[-1].get("result_summary", "")).lower()
        if re.search(r"\b(error|failed|failure|exception|traceback|not found|no such file|cannot|can't)\b", last_result):
            tokens.append("HINT_LAST_ACTION_FAILED")
        if re.search(r"\b(success|completed|done|passed|created|updated|modified)\b", last_result):
            tokens.append("HINT_LAST_ACTION_SUCCESS")

        # 직전 도구 흐름을 모델 입력 힌트로만 전달합니다. 최종 예측을 강제로 바꾸지는 않습니다.
        if len(recent_names) >= 2 and recent_names[-2:] == ["grep_search", "read_file"]:
            tokens.append("HINT_FLOW_SEARCH_THEN_READ")
        if recent_names and recent_names[-1] in {"edit_file", "write_file", "apply_patch"}:
            tokens.append("HINT_AFTER_WRITE_ACTION")
        if recent_names and recent_names[-1] in {"run_tests", "lint_or_typecheck", "run_bash"}:
            tokens.append("HINT_AFTER_EXEC_ACTION")

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
        f"{current_prompt}\n\n"
        f"{current_prompt}\n\n"
        f"{current_prompt}\n\n"
        "HINTS:\n"
        f"{hint_text}\n\n"
        "LAST_CONTEXT:\n"
        f"{last_context_text}\n\n"
        "FLOW_CONTEXT:\n"
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
