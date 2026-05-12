# Coalition Formation Web App

Flask web app for the AI-generated compromise / coalition formation algorithm.

## What it does

Each agent submits an ideal policy sentence. The algorithm finds a compromise sentence that a majority coalition prefers over the status quo (measured by cosine dissimilarity of text embeddings via OpenAI).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Run

```bash
python app.py
```

Open `http://localhost:5000`.

## Parameters

| Parameter | Description |
|---|---|
| Majority quota | Fraction of agents required to form a coalition (default 0.5) |
| Sigma | Noise added to dissimilarity scores (0 = deterministic) |
| Seed | Random seed for reproducibility |

## Routes

- `/` — main input form
- `/run` — POST, runs the algorithm
- `/about` — algorithm description
- `/logs` — view server logs
