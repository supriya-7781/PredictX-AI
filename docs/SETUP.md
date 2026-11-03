# PredictX — Local Setup Guide

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.11+ |
| Git | any recent |
| Ollama (optional) | 0.3+ |

---

## 1. Clone and create a virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/predictx.git
cd predictx

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Also install the local package in editable mode (lets Python find `src/predictx`):

```bash
pip install -e .
```

---

## 3. Configure environment variables

Copy the example file and edit it:

```bash
cp .env.example .env
```

For a **fully local** run (no AWS), the defaults work as-is:

```
STORAGE_MODE=local
LOCAL_STORE=local_store
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

---

## 4. Train the model

```bash
python training/train_model.py
```

Expected output:
```
PredictX — Isolation Forest Training
Loaded 10,000 rows ...
IsolationForest fitted.
accuracy: 0.93
Scaler  → models/scaler.pkl
Model   → models/isolation_forest.pkl
```

---

## 5. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

- Use the **CSV Upload** tab to test with `data/sample/sample_upload.csv`
- The **Live Feed** tab streams the dataset through the simulator
- The **History** tab loads previously saved results

---

## 6. Run the sensor simulator (optional)

```bash
# Print 20 rows to the console
python simulator/sensor_simulator.py --mode mock --rows 20

# Stream the whole dataset with 0.5 s between readings
python simulator/sensor_simulator.py --mode mock --delay 0.5 --loop
```

---

## 7. Install Ollama for AI explanations (optional)

Download from [https://ollama.com/download](https://ollama.com/download) and run:

```bash
ollama pull llama3.2:3b
ollama serve   # starts the local API on port 11434
```

Once Ollama is running, the dashboard will use it automatically.
If it is not running, the dashboard shows rule-based fallback explanations.

---

## 8. Run the test suite

```bash
pytest tests/ -v
```

All 37 tests should pass without AWS or Ollama running.

---

## Project structure

```
PredictX/
├── data/               Source dataset + sample CSV
├── models/             Trained model artefacts (created by train_model.py)
├── src/predictx/
│   ├── ingestion/      CSV validation
│   ├── model/          Inference wrapper
│   ├── storage/        Local / S3 abstraction
│   ├── iot/            MQTT client (real + mock)
│   └── ollama_client/  LLM explainer
├── training/           Model training script
├── simulator/          Sensor data simulator
├── dashboard/          Streamlit app
├── lambda_function/    AWS Lambda handler + deployment deps
├── tests/              Pytest test suite
└── docs/               This file + AWS setup guide
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'predictx'`**
Run `pip install -e .` from the project root.

**`Model not found` error in dashboard**
Run `python training/train_model.py` first.

**Dashboard is slow / hanging on first load**
The first import of scikit-learn takes a few seconds. This is normal.

**Ollama shows as "Offline" in sidebar**
Start Ollama with `ollama serve` or install it from https://ollama.com/download.
The dashboard will still work with rule-based explanations.
