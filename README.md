---
title: Fraud Investigator Simulator
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fraud Investigator Simulator – Real-Time AI Decision Environment

> "A single wrong decision can cost millions. Train your AI agent to detect fraud under real-world pressure."

**The investigator reviews high-velocity transactions in a high-stakes simulation where every decision has immediate financial and social consequences. Your agent must protect the system from fraudsters without alienating legitimate customers.**

---

## 🛡️ Why This Project Matters

In modern fintech, AI decision-making isn't just about accuracy—it's about **Trust**. 
- **Financial Impact**: Missed fraud leads to direct capital loss.
- **Human Impact**: False positives (flagging innocent users) cause immediate trust erosion.
- **The Challenge**: Balancing these competing pressures in a dynamic, evolving environment.

---

## 🎮 Narrative & Mechanics

### 📈 The Pressure System
This isn't a static dataset. Every episode is a **Live Narrative**:
- **Episodes**: 20 transactions per session.
- **Narrative Scaling**: As the session progresses, the "Pressure" increases. Transaction amounts spike, and fraudsters become more cunning (simulating a coordinated attack).
- **Evolving Patterns**: Risk markers and merchant categories shift, requiring the agent to adapt in real-time.

### 🤝 Customer Trust Score
The most critical metric.
- **Base Trust**: 100%
- **False Positive Penalty**: `-10` points. Flagging a safe transaction causes heavy user friction.
- **Missed Fraud Penalty**: `-5` points. Customers get angry when their security is breached.
- **Failure Condition**: If Trust hits **0**, the investigator is fired (episode ends early).

### 🔥 Streak Bonuses
Consistency is rewarded. Detecting multiple fraudulent transactions in a row grants a **+0.2 Streak Bonus**, simulating the investigator "finding the thread" of a coordinated attack.

---

## ⚙️ Environment Mechanics

### Observation Space
The investigator receives a rich set of features:
- `transaction`: {`amount`, `merchant_category`, `location_risk`, `frequency_24h`, `is_new_device`, `user_age`, `hour_of_day`}
- `step`: Current progress.
- `trust_score`: Remaining customer trust.

### Action Space
Discrete space:
- `0`: **APPROVE** (Transaction proceeds; builds trust if correct)
- `1`: **FLAG** (Transaction stops; damages trust if false positive)

### Reward Schema
| Outcome | Decision | Actual | Reward |
| :--- | :--- | :--- | :--- |
| **Correct** | Approve | Safe | `+1.0` |
| **Correct** | Flag | Fraud | `+1.0` |
| **Winning Streak** | Flag | Fraud | `+1.2` (on 2+ streak) |
| **False Positive** | Flag | Safe | `-0.5` (and -10 trust) |
| **Missed Fraud** | Approve | Fraud | `-1.0` (and -5 trust) |

---

## 📊 Evaluation & Metrics

The system is evaluated based on three core pillars:
1. **Accuracy**: Overall decision correctness.
2. **Fraud Detection Rate (FDR)**: Percentage of blocked fraud.
3. **False Positive Rate (FPR)**: Percentage of incorrectly flagged safe users.
4. **Final Trust Score**: The remaining user confidence at the end of the shift.

### Example Performance Output:
```text
Final Metrics (100 episodes):
Accuracy: 92.5%
Average Final Trust: 85.0/100
Total False Positives: 12
Total Missed Fraud: 3
Average Reward: 15.40
```


---


## 📊 Analytics & Insights

Use our built-in visualization tools to analyze your agent's decision-making and performance across multiple episodes.

### 📈 Performance Report
Run the visualization script to generate a high-fidelity report (`performance_report.png`):
```bash
./.venv/bin/python3 visualize.py
```
*This report includes Reward Distributions, Accuracy vs False Positives, and average Error Metrics.*

### 🖥️ Real-Time Investigator Dashboard
We've built a sleek, modern dashboard inspired by top-tier fintech analytics platforms. It connects directly to your FastAPI server for live monitoring or manual interaction.

**To Launch:**
1. Ensure the server is running: `PYTHONPATH=src ./.venv/bin/python3 -m uvicorn fraud_env.server.app:app --host 0.0.0.0 --port 7860`
2. Open `dashboard.html` in your web browser.

---

## 🛠️ API & Compliance

### Standardized Reset/Step
- `POST /reset` -> Returns initial `FraudObservation`.
- `POST /step` -> Returns `(observation, reward, done, info)`.

### Grader-Ready Logging
Logs are formatted for automated parsing:
```text
--- START task=Investigator_Main ---
--- STEP 0 ---
Investigator Decision: FLAG
Trust Impact: 90.0
Reward: -0.5
--- END task=Investigator_Main score=0.850 steps=20 ---
```

---
Built with ❤️ for the Meta PyTorch Hackathon.
This represents a **production-ready AI simulation** for high-stakes financial decisioning.
