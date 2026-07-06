"""
김태영 담당 전처리 코드

목표:
current_prompt, history, session_meta를 하나의 문자열로 합칩니다.
"""

import json
from typing import Any, Dict, List


def history_to_text(history: List[Dict[str, Any]], max_history_items: int = 20) -> str:
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
    last_context_text = last_context_to_text(sample.get("history", []))
    history_text = history_to_text(sample.get("history", []))
    meta_text = session_meta_to_text(sample.get("session_meta", {}))

    return (
        "CURRENT:\n"
        f"{current_prompt}\n\n"
        f"{current_prompt}\n\n"
        "LAST_CONTEXT:\n"
        f"{last_context_text}\n\n"
        "HISTORY:\n"
        f"{history_text}\n\n"
        "META:\n"
        f"{meta_text}"
    )
