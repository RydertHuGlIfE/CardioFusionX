
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

Run from this directory:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` and upload a MATLAB `.mat` file containing a numeric
`val` array with shape `(12, 5000)`. The server performs the same per-lead
z-score normalization as `ECGDataset` and keeps uploaded data in memory.

The dashboard exposes checkpoints that exist in the repository, validation-derived
tuned/conservative thresholds beside those checkpoints, and a fixed `0.50` option.
Checkpoint architecture, output size, state-dict shapes, and label order are
validated before inference. The dashboard hides absent checkpoints and reports
missing PyTorch/checkpoint configuration as a user-facing error.

This is an experimental AI research prototype, not a medical diagnostic device.
Sigmoid outputs and signal-quality indicators are not clinically calibrated.

Run tests after installing the requirements:

```bash
pytest -q
```
