from pathlib import Path
from flask import Flask, request, jsonify
from policy import LinearPolicy
from train import train

if not Path("outputs/policy.json").exists():
    train(episodes=1000)

app = Flask(__name__)
policy = LinearPolicy.load()

@app.get("/health")
def health():
    return {"status": "ok", "service": "financial-optimization"}

@app.post("/predict")
def predict():
    state = request.get_json(force=True)
    action = policy.predict(state)
    scores = policy.predict_scores(state).tolist()
    return jsonify({
        "recommended_action": action,
        "policy_scores": scores,
        "disclaimer": "Educational portfolio demo only. Not financial advice."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
