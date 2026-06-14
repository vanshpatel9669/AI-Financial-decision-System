import numpy as np
from environment import MarketScenarioEnvironment, ACTIONS
from policy import LinearPolicy, FEATURES

def train(episodes=5000, lr=0.05):
    env = MarketScenarioEnvironment()
    policy = LinearPolicy()
    for _ in range(episodes):
        state = env.sample_state()
        rewards = np.array([env.reward(state, i) for i in range(len(ACTIONS))])
        best_action = int(np.argmax(rewards))
        x = np.array([1.0] + [state[f] for f in FEATURES])
        scores = policy.predict_scores(state)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        target = np.zeros(len(ACTIONS)); target[best_action] = 1
        gradient = (target - probs)[:, None] @ x[None, :]
        policy.weights += lr * gradient
    policy.save()
    print("Saved trained policy to outputs/policy.json")

if __name__ == "__main__":
    train()
