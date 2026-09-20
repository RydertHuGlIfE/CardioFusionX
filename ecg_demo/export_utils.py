import csv
import io
import json


def prediction_csv(result):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["code", "name", "probability", "threshold", "detected", "near_threshold"])
    writer.writeheader()
    writer.writerows(result.get("all_predictions", []))
    return output.getvalue()


def prediction_json(result, quality, model_info, include_history=False):
    payload = {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "model": model_info,
        "threshold_strategy": result.get("threshold_strategy"),
        "predictions": result.get("all_predictions", []),
        "signal_quality": quality,
        "history_included": bool(include_history),
        "disclaimer": "Experimental AI research prototype; not a medical diagnostic device.",
    }
    return json.dumps(payload, indent=2)
