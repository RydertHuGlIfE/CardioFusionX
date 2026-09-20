
# CardioFusionX ECG Demo

## Setup

1. Put your trained checkpoint in this folder as `model.pth`.
2. Create/activate your virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python app.py
```

5. Open:

http://127.0.0.1:5000

## Input format

Upload a `.mat` file containing:

```python
val.shape == (12, 5000)
```

The demo normalizes each lead exactly like `ECGDataset`, uses a global threshold of `0.65`, and plots all 12 leads. This is a research/demo tool, not a clinical diagnostic system.
