"""
CardioFusionX Evaluation & Comparison Wrapper
Delegates to evaluate_compare.py for high-speed batched evaluation,
dual threshold testing (F1-tuned & precision-conservative), and PR-AUC/ROC-AUC computation.
"""
import sys
from evaluate_compare import main

if __name__ == "__main__":
    main()
