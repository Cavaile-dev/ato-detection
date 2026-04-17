import time
import requests
import numpy as np
from datetime import datetime, timezone

class AttackerSimulator:
    def __init__(self, target_username: str, base_url: str = "http://localhost:5000"):
        self.target = target_username
        self.base_url = base_url

    def _generate_events(self, config: dict):
        events = []
        # Move mouse
        for _ in range(config["moves"]):
            # Mimic attacker improvement: Adds noise (variance) to pure mean intervals
            variance = config.get("move_variance", 0.0)
            noise = np.random.normal(0, variance) if variance > 0 else 0
            interval = max(0.01, config["move_interval"] + noise)
            
            events.append({
                "type": "mouse_move",
                "x": int(np.random.randint(100, 1000)), 
                "y": int(np.random.randint(100, 800)),
                "timestamp": time.time()
            })
            time.sleep(interval)
        
        # Click
        decision_variance = config.get("decision_variance", 0.0)
        decision_noise = np.random.normal(0, decision_variance) if decision_variance > 0 else 0
        latency = max(0.0, config["decision_latency"] + decision_noise)
        
        time.sleep(latency)
        events.append({"type": "click", "x": 500, "y": 500, "timestamp": time.time()})
        return events

    def submit_session(self, events: list, session_id: str):
        payload = {
            "session_id": session_id,
            "username": self.target,
            "events": events
        }
        res = requests.post(f"{self.base_url}/api/v1/sessions/start", json=payload)
        res = requests.post(f"{self.base_url}/api/v1/sessions/{session_id}/end", json=payload)
        return res.json()

    def simulate_fast_bot(self):
        """No hesitation, linear/fast tracking, zero dwell time variability"""
        return self._generate_events({
            "moves": 2, "move_interval": 0.01, "move_variance": 0.0,
            "decision_latency": 0.00, "decision_variance": 0.0
        })

    def simulate_random_attacker(self):
        """Erratic, highly variable, chaotic movement"""
        return self._generate_events({
            "moves": 50, 
            "move_interval": np.random.uniform(0.05, 0.5), "move_variance": 0.5,
            "decision_latency": np.random.uniform(0.1, 3.0), "decision_variance": 1.0
        })

    def simulate_mimic_attacker(self, target_baseline: dict):
        """Attempts to match target's statistical mean with synthetic noise injection"""
        return self._generate_events({
            "moves": 15,
            # Target's baseline velocity dictates the interval
            "move_interval": target_baseline.get("velocity_mean", 0.2),
            "move_variance": target_baseline.get("velocity_variance", 0.05),
            "decision_latency": target_baseline.get("decision_latency", 0.8),
            "decision_variance": target_baseline.get("decision_latency_variance", 0.2)
        })

if __name__ == "__main__":
    simulator = AttackerSimulator("alice")
    
    print("Collecting Fast Bot Attack Data...")
    for i in range(5): 
        bot_events = simulator.simulate_fast_bot()
        result = simulator.submit_session(bot_events, f"sim_bot_{i}")
        print(f"Bot {i} Risk Level:", result.get("risk_level"))
        
    print("\nCollecting Mimic Attacker Data...")
    # Simulate known target parameters (e.g. from compromised historical baseline leak)
    dummy_baseline = {"velocity_mean": 0.15, "velocity_variance": 0.02, "decision_latency": 0.4, "decision_latency_variance": 0.1}
    for i in range(5):
        mimic_events = simulator.simulate_mimic_attacker(dummy_baseline)
        result = simulator.submit_session(mimic_events, f"sim_mimic_{i}")
        print(f"Mimic {i} Risk Level:", result.get("risk_level"))
