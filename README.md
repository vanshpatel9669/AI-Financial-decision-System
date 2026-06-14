# AI Financial Decision Optimization System

An AI-powered decision optimization platform that simulates portfolio allocation strategies using a reinforcement learning inspired policy engine.

The system evaluates market conditions and portfolio health metrics to recommend optimal actions such as increasing exposure, reducing risk, or maintaining current allocation.

---

## Overview

This project demonstrates how AI-driven decision systems can assist financial analysts and portfolio managers by transforming market signals into actionable recommendations.

The platform:

- Evaluates market volatility and momentum
- Monitors portfolio drawdown levels
- Assesses liquidity through cash ratios
- Generates real-time allocation recommendations
- Exposes predictions through a REST API

---

## Architecture

```text
Market Signals
      │
      ▼
Feature Processing Layer
      │
      ▼
Policy Engine
(Reinforcement Learning Inspired)
      │
      ▼
Decision Scoring
      │
      ▼
Recommended Action
      │
      ▼
REST API Response
```

---

## Features

### Intelligent Portfolio Assessment

Analyzes multiple portfolio and market indicators simultaneously:

- Volatility
- Momentum
- Drawdown
- Cash Ratio

### Decision Optimization Engine

Generates optimized allocation recommendations using a trained policy model.

### REST API

Exposes predictions through a lightweight Flask API for easy integration into dashboards and financial systems.

### Lightweight Deployment

- Python
- Flask
- NumPy
- JSON Model Persistence

No external brokerage integrations required.

---

## Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| API | Flask |
| ML Logic | Reinforcement Learning Inspired Policy |
| Data Processing | NumPy |
| Model Storage | JSON |
| Deployment | Local / Docker Ready |

---

## Project Structure

```text
financial-decision-optimization-system/
│
├── api.py
├── train.py
├── policy.py
├── environment.py
│
├── outputs/
│   └── policy.json
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/financial-decision-optimization-system.git
cd financial-decision-optimization-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Generate policy weights:

```bash
python train.py
```

Output:

```text
outputs/policy.json
```

---

## Start the API

```bash
python -m flask --app api run --port 5001
```

Server:

```text
http://127.0.0.1:5001
```

---

## API Usage

### Prediction Endpoint

```http
POST /predict
```

### Sample Request

```json
{
  "volatility": 0.22,
  "momentum": 0.50,
  "drawdown": 0.10,
  "cash_ratio": 0.30
}
```

### Sample Response

```json
{
  "recommended_action": "increase_exposure"
}
```

---

## Example Curl Request

```bash
curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '{
"volatility":0.22,
"momentum":0.50,
"drawdown":0.10,
"cash_ratio":0.30
}'
```

---

## Business Value

This project demonstrates:

- AI-powered decision systems
- Policy optimization
- Production-ready REST APIs
- End-to-end ML deployment workflows
- Financial analytics applications

---

## Future Enhancements

- PPO / DQN implementation
- Portfolio backtesting engine
- Real market data integration
- Risk-adjusted reward optimization
- Docker deployment
- Cloud deployment on AWS

---

## Author

**Vansh Patel**

M.S. Computer Science (AI & ML)
Stevens Institute of Technology

GitHub: https://github.com/vanshpatel9669
LinkedIn: https://linkedin.com/in/vanshpatel1702