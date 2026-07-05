"""
김태영 담당 전처리 코드

목표:
current_prompt, history, session_meta를 하나의 문자열로 합칩니다.
"""

import json
from typing import Any, Dict, List


def history_to_text(history: List[Dict[str, Any]], max_history_items: int = 40) -> str:
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
    session_meta는 중첩 dictionary이므로 json.dumps로 문자열화합니다.
    """
    if not isinstance(session_meta, dict):
        return ""
    return json.dumps(session_meta, ensure_ascii=False, sort_keys=True)


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
    history_text = history_to_text(sample.get("history", []))
    meta_text = session_meta_to_text(sample.get("session_meta", {}))

    return (
        "CURRENT:\n"
        f"{current_prompt}\n\n"
        "HISTORY:\n"
        f"{history_text}\n\n"
        "META:\n"
        f"{meta_text}"
    )
