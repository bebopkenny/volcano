"""Download raw inputs and cache 10 minute seismic features.

Usage (from pipeline/): python -m scripts.fetch [case-id ...]
"""
import sys

from volcano.config import load_cases
from volcano.seismic import fetch_case
from volcano.thermal import fetch_detections

if __name__ == "__main__":
    cases = load_cases()
    for case_id in sys.argv[1:] or cases:
        fetch_case(cases[case_id])
        fetch_detections(cases[case_id])
