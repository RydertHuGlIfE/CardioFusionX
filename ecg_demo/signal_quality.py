import numpy as np


def analyze_signal(signal, sample_rate=500):
    signal = np.asarray(signal, dtype=np.float32)
    finite = np.isfinite(signal)
    clean = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    lead_std = clean.std(axis=1)
    lead_range = clean.max(axis=1) - clean.min(axis=1)
    diff = np.diff(clean, axis=1)
    high_frequency_ratio = float(np.mean(np.abs(diff)) / (np.mean(np.abs(clean - clean.mean(axis=1, keepdims=True))) + 1e-8))
    window = max(1, int(sample_rate * 0.2))
    kernel = np.ones(window, dtype=np.float32) / window
    baseline = np.array([np.convolve(lead, kernel, mode="same") for lead in clean])
    baseline_wander = float(np.mean(np.std(baseline, axis=1)))
    amplitude = float(np.max(np.abs(clean)))
    flatline_leads = int(np.sum(lead_std < 1e-6))
    saturation_fraction = float(np.mean(np.abs(clean) >= max(amplitude * 0.999, 1e-6))) if amplitude else 1.0
    finite_fraction = float(finite.mean())

    penalties = 0.0
    penalties += min(0.45, flatline_leads / max(clean.shape[0], 1) * 0.6)
    penalties += min(0.25, max(0.0, high_frequency_ratio - 0.35) * 0.25)
    penalties += min(0.20, baseline_wander / (np.mean(lead_std) + 1e-8) * 0.2)
    penalties += min(0.20, max(0.0, 1 - finite_fraction) * 2)
    score = float(max(0.0, min(1.0, 1.0 - penalties)))
    status = "Good" if score >= 0.8 else "Acceptable" if score >= 0.6 else "Poor" if score >= 0.35 else "Unreliable"
    return {
        "score": round(score, 3),
        "status": status,
        "amplitude_min": round(float(clean.min()), 6),
        "amplitude_max": round(float(clean.max()), 6),
        "mean": round(float(clean.mean()), 6),
        "std": round(float(clean.std()), 6),
        "nan_count": int(np.isnan(signal).sum()),
        "infinite_count": int(np.isinf(signal).sum()),
        "flatline_leads": flatline_leads,
        "saturation_fraction": round(saturation_fraction, 6),
        "baseline_wander": round(baseline_wander, 6),
        "high_frequency_noise_ratio": round(high_frequency_ratio, 6),
        "lead_std": [round(float(value), 6) for value in lead_std],
        "lead_quality": ["Poor" if value < 1e-6 else "Good" for value in lead_std],
        "warning": "Signal quality is unreliable. Prediction may be inaccurate." if status == "Unreliable" else None,
        "disclaimer": "This signal quality score is a technical heuristic and is not medically validated.",
    }
