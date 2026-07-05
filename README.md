# DACON Agent Action Prediction - TEAM NEXT STEP

데이콘 AI Agent Action Prediction 경진대회를 위한 팀 프로젝트입니다.

## 대회 목표

사용자의 현재 요청(`current_prompt`), 이전 작업 기록(`history`), 세션 정보(`session_meta`)를 바탕으로  
assistant가 다음에 수행할 Action을 예측하는 다중 클래스 분류 문제입니다.

## 팀원 역할

| 팀원 | 역할 | 담당 파일 |
|---|---|---|
| 박진영 | 팀장 / GitHub 관리 / 최종 제출 | `script.py` |
| 소재민 | 데이터 분석 | `notebooks/EDA.ipynb` |
| 김태영 | 전처리 | `src/preprocess.py` |
| 양정재 | 모델 학습 / 예측 | `src/train.py`, `src/predict.py` |

## 프로젝트 구조

초기 GitHub 구조를 그대로 유지합니다.

```text
.
├── data/
├── model/
├── notebooks/
├── src/
├── .gitignore
├── README.md
├── requirements.txt
└── script.py
```

## 데이터 준비

데이콘에서 받은 데이터 파일을 `data/` 폴더에 넣습니다.

```text
data/
├── train.jsonl
├── train_labels.csv
├── test.jsonl
└── sample_submission.csv
```

실제 데이터 파일은 GitHub에 올리지 않습니다.

## 설치

```bash
python3 -m pip install -r requirements.txt
```

## 학습

```bash
python3 -m src.train
```

성공하면 아래 파일이 생성됩니다.

```text
model/model.joblib
```

## 예측

```bash
python3 -m src.predict
```

성공하면 아래 파일이 생성됩니다.

```text
submission.csv
```

## 제출 실행 파일 테스트

```bash
python3 script.py
```

## 데이콘 제출용 submit.zip 만들기

GitHub에는 `submit.zip`을 올리지 않습니다.  
제출 직전에 아래 구조로 직접 압축합니다.

```text
submit.zip
├── model/
│   └── model.joblib
├── requirements.txt
└── script.py
```

Mac 터미널 예시:

```bash
zip -r submit.zip model/model.joblib script.py requirements.txt
```

## 모델 방식

양정재가 분석한 baseline 구조를 기반으로 합니다.

```text
TfidfVectorizer -> 선형 분류 모델
```

기존 baseline은 `current_prompt`만 사용했지만, 이 프로젝트에서는 김태영의 전처리 방식에 따라 아래 세 정보를 하나의 문자열로 합쳐 사용합니다.

```text
CURRENT:
current_prompt

HISTORY:
history

META:
session_meta
```

## 주의사항

- `data/` 실제 파일은 GitHub에 올리지 않습니다.
- `model/model.joblib`은 학습 후 생성되는 파일이라 GitHub에 올리지 않습니다.
- `submission.csv`, `submit.zip`도 생성 결과물이므로 GitHub에 올리지 않습니다.
