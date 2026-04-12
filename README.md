# Hardware-Aware Neural Network Trade-off Analysis Dashboard

## Approach

This project provides an interactive analysis tool to explore the trade-off between hardware efficiency and model performance.

A custom trade-off score is defined as:

Trade-off Score = Pin Reduction − λ × Accuracy Drop

- Pin Reduction represents hardware efficiency gain
- Accuracy Drop represents performance degradation
- λ (lambda) controls the relative importance of accuracy vs hardware efficiency

By adjusting λ, users can simulate different deployment priorities and observe how the optimal configuration changes.

Streamlit live demo:
https://hardware-aware-experiment-dashboard-x95bufhzfrgdusxyf593hv.streamlit.app/

## Note
An earlier version of this project (`legacy_log_dashboard.py`) was initially developed for simple log visualization.  
The current version extends this into a hardware-aware analysis tool with structured trade-off evaluation.

## Key Features

- Trade-off Visualization  
  Visualizes relationships between accuracy, pin reduction, and slice reduction

- Pareto Frontier Analysis  
  Identifies optimal configurations under multi-objective constraints

- Recommendation System  
  Automatically suggests configurations based on:
  - Best trade-off score
  - Best accuracy retention
  - Highest hardware efficiency

- Lambda Sensitivity Analysis  
  Shows how optimal configurations shift as trade-off preference (λ) changes

- Experiment Dashboard  
  Interactive exploration of experiment logs (accuracy curves, metrics, comparisons)

## Example Metrics

•	maximum test accuracy
•	accuracy drop from baseline
•	pin reduction rate
•	slice reduction
•	packing efficiency
•	trade-off score
	
## Files

- `analysis_engine.py`: core analysis logic
- `app.py`: Streamlit interface
- `sample_experiments.json`: sample experiment log file for testing the dashboard

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
