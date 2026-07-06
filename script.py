"""
박진영 담당 제출 실행 파일

데이콘 평가 서버에서 실행되는 파일입니다.
제출용 zip에는 이 script.py, requirements.txt, model/model.joblib이 들어가야 합니다.

로컬 실행:
python3 script.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


def history_to_text(history: List[Dict[str, Any]], max_history_items: int = 40) -> str:
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


def make_input_text(sample: Dict[str, Any]) -> str:
    if not isinstance(sample, dict):
        sample = {}

    current_prompt = sample.get("current_prompt", "")
    history_text = history_to_text(sample.get("history", []))
    meta_text = json.dumps(sample.get("session_meta", {}), ensure_ascii=False, sort_keys=True)

    return (
        "CURRENT:\n"
        f"{current_prompt}\n\n"
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
