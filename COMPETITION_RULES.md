# Competition Rules Checklist

This project must be modified under the following competition constraints.

## Submission Package

- Submit exactly one `submit.zip`.
- Required structure:

```text
submit.zip
├── model/
├── script.py
└── requirements.txt
```

- Directory and file names must match exactly.
- `script.py` is executed automatically by the evaluation server.
- `requirements.txt` must be installable with:

```bash
pip install -r requirements.txt
```

- Final prediction file must be created at:

```text
output/submission.csv
```

- The evaluation server adds `data/` and `output/` automatically.
- `data/` is read-only on the evaluation server.
- Zip size limit: under 1 GB.

## Runtime Environment

- OS: Ubuntu 22.04.5 LTS
- GPU: NVIDIA T4, 16 GB VRAM
- CPU: 3 vCPU
- RAM: 12 GB
- Python: 3.11.15
- CUDA: 12.8
- Internet access: disabled after package installation.
- Package installation time limit: 10 minutes.
- Inference execution time limit: 10 minutes.

## Default Installed Packages

Prefer using these server-installed versions and avoid listing them in `requirements.txt` unless necessary:

- `torch==2.7.1+cu128`
- `pandas==2.0.3`
- `numpy==1.26.4`
- `scipy==1.15.3`
- `scikit-learn==1.8.0`
- `joblib==1.5.3`
- `threadpoolctl==3.6.0`
- `narwhals==2.21.2`
- `transformers==4.46.3`
- `accelerate==1.9.0`
- `sentencepiece==0.1.99`
- `regex==2023.12.25`
- `tqdm==4.66.4`
- `loguru==0.7.2`
- `pyyaml==6.0.1`
- `rich==13.7.1`

## Development Rules

- Use Python.
- Code must use relative paths.
- Training code, inference code, and comments must be UTF-8.
- Do not rely on internet downloads during inference.
- All models or assets needed during inference must be included in `submit.zip`.
- External pretrained models, data, APIs, and paid services may be used during development if legally allowed.
- Track sources and usage scope for external pretrained models, external data, APIs, paid services, and references.
- Avoid adding unnecessary dependencies because installation failures count as installation errors.
- Any error after `script.py` starts counts as a submission error.

## Preliminary Round

- Required submission: model files and inference code as zip.
- Deadline: July 15, 10:00.
- Max submissions: 10 per day.
- If no valid code zip is submitted and shown on the leaderboard, the team is considered non-participating.

## Final Round Preparation

If selected as a finalist or finalist candidate, prepare:

- Reproducible training code for the Private Score.
- Sources for pretrained models, external data, references, and other external resources.
- Development environment and library versions.
- Presentation PDF for a 10-minute presentation.
- Poster session material when the template is released.
