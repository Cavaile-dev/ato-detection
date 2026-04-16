# Real-Time Login Anomaly Detection Prototype

This project is a local prototype for behavioral-biometrics-based Account Takeover detection. It scores a live post-login session by combining:

- Unsupervised anomaly detection on user behavior
- Context deviation from trusted IP, device, and login hour
- Adaptive response logic to allow, monitor, or trigger MFA

## Project Structure

```text
ato-detection/
├── data/
├── model/
├── server/
│   ├── app.py
│   ├── config.py
│   ├── feature_extraction.py
│   ├── mock_data.py
│   ├── model.py
│   ├── pipeline.py
│   ├── risk_engine.py
│   ├── schemas.py
│   ├── train_model.py
│   ├── evaluation.py
│   ├── generate_features.py
│   ├── simulate_realtime.py
│   └── test_model.py
├── web/
│   ├── login.html
│   ├── dashboard.html
│   ├── logger.js
│   └── styles.css
└── requirements.txt
```

## Components

Important note for the PoC:

- End users do not choose whether they are "normal" or "attacker".
- The `normal` and `attacker` labels only exist inside the synthetic dataset and simulator so we can evaluate the detector offline.
- At runtime, the detector only receives an unlabeled session and infers risk from behavior plus context.

### 1. Frontend Event Collector

`web/logger.js` captures:

- Mouse movement with velocity and acceleration
- Click timestamps and click intervals
- Keystroke intervals
- Navigation events

Events are flushed to the backend in near real time over REST.

### 2. Feature Engineering

`server/feature_extraction.py` converts per-session raw events into:

- `avg_mouse_speed`
- `mouse_acceleration_variance`
- `click_interval_mean`
- `click_interval_std`
- `navigation_entropy`
- `session_duration`
- `deviation_from_typical_time`

### 3. Unsupervised Model

`server/model.py` uses Isolation Forest and trains only on normal behavior rows.

### 4. Real-Time Pipeline

`server/pipeline.py` uses a background queue worker to process streaming event batches per session.

### 5. Risk Engine

`server/risk_engine.py` combines:

- anomaly score
- IP deviation
- device deviation
- login-time deviation

Outputs:

- `LOW` -> `ALLOW_SESSION`
- `MEDIUM` -> `LOG_AND_MONITOR`
- `HIGH` -> `TRIGGER_MFA`

### 6. Mock Data And Simulation

`server/mock_data.py` generates both normal and attacker sessions.

`server/simulate_realtime.py` replays those sessions through the queue-backed pipeline.

## Run Locally

Install dependencies if needed:

```powershell
py -m pip install -r requirements.txt
```

Generate training data and train the model:

```powershell
py -m server.train_model
```

Run the web app:

```powershell
py -m server.app
```

Open:

`http://127.0.0.1:5000/`

## How To Demo The PoC

1. Start a normal monitored session from the login page.
2. Interact with the dashboard normally to show a low-risk session.
3. Click `Inject Suspicious Behavior` to feed attacker-like event patterns into the same detector.
4. For a full attacker-vs-normal replay with suspicious context, run:

```powershell
py -m server.simulate_realtime
```

## Useful Scripts

Generate mock sessions and features:

```powershell
py -m server.mock_data
```

Evaluate the trained model:

```powershell
py -m server.evaluation
```

Simulate normal vs attacker streaming sessions:

```powershell
py -m server.simulate_realtime
```
