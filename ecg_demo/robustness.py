import numpy as np


def apply_noise(signal, noise_type, intensity, seed=None):
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=np.float32)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if noise_type == "gaussian":
        scale = max(float(signal.std()), 1e-6) * 0.25 * intensity
        return signal + rng.normal(0, scale, signal.shape).astype(np.float32)
    if noise_type == "baseline":
        time = np.arange(signal.shape[1], dtype=np.float32) / 500.0
        frequency = 0.1 + 0.25 * intensity
        wander = (float(signal.std()) * 0.35 * intensity) * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))
        return signal + wander[None, :]
    if noise_type == "motion":
        output = signal.copy()
        width = max(1, int(signal.shape[1] * 0.08 * intensity))
        start = int(rng.integers(0, max(1, signal.shape[1] - width)))
        output[:, start:start + width] += rng.normal(0, signal.std() * 1.5 * intensity, (signal.shape[0], width)).astype(np.float32)
        return output
    if noise_type == "scaling":
        scales = rng.uniform(1 - 0.25 * intensity, 1 + 0.25 * intensity, signal.shape[0]).astype(np.float32)
        return signal * scales[:, None]
    if noise_type == "lead_noise":
        output = signal.copy()
        lead = int(rng.integers(0, signal.shape[0]))
        output[lead] += rng.normal(0, signal.std() * intensity, signal.shape[1]).astype(np.float32)
        return output
    raise ValueError(f"Unsupported noise type: {noise_type}")


def changed_labels(original, modified):
    original_map = {row["code"]: row for row in original["all_predictions"]}
    modified_map = {row["code"]: row for row in modified["all_predictions"]}
    changes = []
    for code in original_map.keys() & modified_map.keys():
        before, after = original_map[code], modified_map[code]
        if before["detected"] != after["detected"] or abs(before["probability"] - after["probability"]) >= 0.05:
            changes.append({"code": code, "name": after["name"], "before": before["probability"], "after": after["probability"], "status_changed": before["detected"] != after["detected"]})
    return sorted(changes, key=lambda row: abs(row["after"] - row["before"]), reverse=True)
