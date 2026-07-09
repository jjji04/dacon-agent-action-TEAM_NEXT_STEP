"""
김태영 담당 전처리 코드

목표:
current_prompt, history, session_meta를 하나의 문자열로 합칩니다.
"""

import json
import re
from typing import Any, Dict, List


def history_to_text(history: List[Dict[str, Any]], max_history_items: int = 10) -> str:
    """
    history 리스트를 문자열로 변환합니다.

    history는 list 안에 dictionary가 들어 있는 구조입니다.
    role이 user이면 content를 사용하고,
    role이 assistant_action이면 name, args, result_summary를 사용합니다.
    """
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
            content = item.get("content", "")
            lines.append(f"USER: {content}")

        elif role == "assistant_action":
            name = item.get("name", "")
            args = json.dumps(item.get("args", {}), ensure_ascii=False, sort_keys=True)
            result = item.get("result_summary", "")
            lines.append(f"ASSISTANT_ACTION: {name} | ARGS: {args} | RESULT: {result}")

        else:
            lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    return "\n".join(lines)


def session_meta_to_text(session_meta: Dict[str, Any]) -> str:
    """
    action 예측에 도움 되는 메타 정보만 짧게 문자열화합니다.
    """
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
        tokens.append("HINT_READ_FILE_PATH_STRONG")
    if re.search(r"\*\.[a-zA-Z0-9]+|\*\*/|glob", combined_text):
        tokens.append("HINT_GLOB_PATTERN_STRONG")
    if re.search(r"\b(rg|grep|ripgrep|occurrences?|matches?|search for|find occurrences)\b", combined_text):
        tokens.append("HINT_GREP_SEARCH_STRONG")
    if re.search(r"\b(ls|tree|list directory|folder structure|directory tree|what files|files in)\b", combined_text):
        tokens.append("HINT_LIST_DIRECTORY_STRONG")
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
    """
    sample 하나를 모델 입력 문자열 하나로 변환합니다.

    최종 형태:
    CURRENT:
    ...

    HISTORY:
    ...

    META:
    ...
    """
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
        "HINTS:\n"
        f"{hint_text}\n\n"
        "LAST_CONTEXT:\n"
        f"{last_context_text}\n\n"
        "HISTORY:\n"
        f"{history_text}\n\n"
        "META:\n"
        f"{meta_text}"
    )
