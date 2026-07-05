"""
양정재 담당 예측 코드

학습된 model/model.joblib을 불러와 test.jsonl을 예측하고 submission.csv를 생성합니다.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import make_input_text


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


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
    predictions = model.predict(X)

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
