# PredictX AI — Real-Time Machine Failure Alerts

An industrial predictive maintenance platform that streams sensor data,
detects anomalies with an Isolation Forest model, explains *why* each
anomaly was flagged, scores machine health in real time, and recommends
specific maintenance actions.

---

## Features

| Feature | Description |
|---|---|
| **Anomaly Detection** | Scikit-learn Isolation Forest trained on the AI4I 2020 Predictive Maintenance Dataset |
| **Explainable AI (XAI)** | Per-feature Z-score deviation analysis — identifies which sensor readings are statistically abnormal and by how much |
| **Machine Health Score** | 0–100 dynamic score using weighted feature penalties, updated every reading |
| **Maintenance Recommendations** | Transparent rule-based engine — generates specific inspection actions backed by sensor evidence |
| **AI Explanations** | Ollama LLM (llama3.2:3b) with automatic rule-based fallback when Ollama is offline |
| **Live Monitoring** | Sensor simulator streams the full 10 000-row dataset with real-time anomaly chart |
| **Storage** | Local filesystem (default) or Amazon S3 |
| **AWS IoT Core** | MQTT publishing with TLS, mock mode for local testing |

---

## Project Structure

```
PredictX/
├── src/predictx/
│   ├── model/              Isolation Forest inference wrapper
│   ├── ingestion/          CSV validation (UCI + Kaggle + minimal)
│   ├── storage/            Local / S3 abstraction
│   ├── xai/                Feature deviation analysis (Explainable AI)
│   ├── health/             Dynamic machine health scorer (0–100)
│   ├── recommendations/    Rule-based maintenance recommendation engine
│   ├── ollama_client/      LLM explanation generator
│   └── iot/                AWS IoT Core MQTT client
├── training/               Model training script
├── simulator/              Sensor data replay simulator
├── dashboard/
│   ├── app.py              Entry point + sidebar navigation
│   ├── design.py           Shared dark-theme design system
│   └── pages/
│       ├── overview.py     Project intro + pipeline diagram
│       ├── ingestion.py    CSV upload + validation + inference
│       ├── monitoring.py   Live simulator stream
│       ├── health_monitor.py   Machine Health (NEW)
│       ├── anomalies.py    Anomaly cards + XAI panel (UPDATED)
│       ├── assistant.py    LLM explanations + recommendations (UPDATED)
│       └── history.py      Historical session browser
├── lambda_function/        AWS Lambda handler
├── tests/                  Pytest test suite (119 tests)
├── data/                   AI4I 2020 dataset + sample CSV
├── models/                 Trained model artefacts
└── docs/                   AWS setup guide
```

---

## Quick Start (Local — No AWS Required)

### 1. Clone and activate the virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/predictx.git
cd PredictX

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .          # install the local predictx package
```

### 3. Configure environment

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # macOS / Linux
```

The default `.env` works without any changes for local mode:

```
STORAGE_MODE=local
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### 4. Train the model

```bash
python training/train_model.py
```

This reads `data/ai4i2020.csv` and writes `models/isolation_forest.pkl`
and `models/scaler.pkl`.

### 5. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`.

---

## Dashboard Walkthrough

### Overview
Project introduction, pipeline diagram, system status, and recent results.

### Data Ingestion
1. Click **Use Sample Dataset → Load Sample Dataset** for an instant demo.
2. Or upload any CSV with the five required columns.
3. Click **Run Anomaly Detection** — results populate all other pages.

### Machine Health (NEW)
- Fleet summary: Healthy / Warning / Poor / Critical reading counts.
- Health score timeline across all loaded readings.
- Feature deviation heatmap — Z-scores for every sensor over time.
- Select any reading for a detailed XAI breakdown and recommendations.

### Anomaly Detection
- Severity-filtered anomaly cards.
- Each card now includes a **Feature Deviation (XAI) panel** showing which
  sensors deviated and by how many standard deviations.

### AI Maintenance
- Select an anomaly event, click **Generate Explanation** for an LLM or
  rule-based text explanation.
- Scroll down to **Structured Maintenance Recommendations** for evidence-backed
  inspection actions (works offline, no Ollama required).

### Live Monitoring
- Configure Mock (local) or MQTT (AWS IoT Core) mode.
- Adjust row count and inter-reading delay.
- Click **Start Streaming** — watch the health score chart update live.

### History
- Browse all saved sessions.
- Search, filter by status, sort by anomaly rate.
- Load any session back into the analysis pages.

---

## Running the Tests

```bash
pytest tests/ -v
```

Expected output: **119 tests passing**

| Test file | Tests | What it covers |
|---|---|---|
| `test_validator.py` | 38 | CSV validation, UCI + Kaggle layouts, type coercion |
| `test_predictor.py` | 11 | Isolation Forest inference, error handling |
| `test_explainer.py` | 14 | LLM explainer, fallback, mock HTTP |
| `test_feature_explainer.py` | 24 | XAI Z-scores, ranking, narrative, edge cases |
| `test_health_scorer.py` | 25 | Health score formula, bands, missing values |
| `test_recommendation_engine.py` | 27 | Rule triggers, confidence levels, ordering |

---

## Explainable AI — How It Works

The XAI module uses **standardised deviation (Z-score) analysis**.

The `StandardScaler` fitted during training stores two arrays:
- `mean_` — the average value of each sensor across 10 000 training rows
- `scale_` — the standard deviation of each sensor

For any new reading:

```
Z-score = (current_value - training_mean) / training_std
```

A Z-score of 0 means the reading is exactly at the training average.
A Z-score of +2 means it is 2 standard deviations above the average.

**Significance threshold:** |Z| > 2.0 (a standard engineering criterion).

This is mathematically honest because it uses the exact same numerical
space in which the Isolation Forest was trained — not an approximation.

---

## Machine Health Score — Formula

```
penalty(z) = min(1.0, (|z| / 3.0)²)   for each feature
raw_penalty = Σ weight[i] × penalty(z[i])
health_score = round(100 × (1 - raw_penalty))
```

Default weights:

| Sensor | Weight | Rationale |
|---|---|---|
| Rotational speed | 0.25 | Primary mechanical stress |
| Torque | 0.25 | Primary mechanical stress |
| Air temperature | 0.20 | Process health indicator |
| Process temperature | 0.20 | Direct failure driver |
| Tool wear | 0.10 | Gradual wear |

---

## Optional: Ollama LLM Explanations

Install from [ollama.com/download](https://ollama.com/download), then:

```bash
ollama pull llama3.2:3b
ollama serve
```

The dashboard auto-detects Ollama. If it is not running, rule-based
fallback explanations are shown instead.

---

## AWS Integration

For the full cloud pipeline (IoT Core → Lambda → S3), follow
`docs/AWS_SETUP.md`.

Key environment variables for AWS mode:

```
STORAGE_MODE=s3
S3_BUCKET=your-bucket-name
IOT_ENDPOINT=xxxx.iot.us-east-1.amazonaws.com
IOT_CLIENT_ID=predictx-simulator
IOT_CERT=certs/device.pem.crt
IOT_KEY=certs/private.pem.key
IOT_CA=certs/AmazonRootCA1.pem
```

---

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
— 10 000 synthetic milling-machine sensor records with 5 documented failure modes.

The five sensor features used by the model:

| Column | Unit | Description |
|---|---|---|
| Air temperature [K] | Kelvin | Ambient air temperature |
| Process temperature [K] | Kelvin | Machine process temperature |
| Rotational speed [rpm] | rpm | Spindle rotational speed |
| Torque [Nm] | Newton-metres | Motor torque |
| Tool wear [min] | minutes | Cumulative tool usage time |
