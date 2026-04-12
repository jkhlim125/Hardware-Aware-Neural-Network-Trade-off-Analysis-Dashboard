# Hardware-Aware Experiment Analysis Dashboard

This project is an interactive analysis tool for comparing experiment configurations in LUT-based neural networks from both accuracy and hardware-efficiency perspectives.

It was developed to support structured comparison of different packing and sparsity configurations during FPGA-oriented neural network experiments.

## Note
An earlier version of this project (`legacy_log_dashboard.py`) was initially developed for simple log visualization.  
The current version extends this into a hardware-aware analysis tool with structured trade-off evaluation.

## Main Features

- Parse experiment results from JSON logs
- Compare maximum accuracy and accuracy drop against baseline
- Analyze hardware-related metrics such as pin reduction, slice reduction, and packing efficiency
- Compute a custom trade-off score for balancing hardware gain and accuracy penalty
- Visualize Pareto frontiers for configuration comparison
- Explore epoch-level curves for selected configurations
- Export summary results as CSV

## Motivation

In LUT-based neural network experiments, comparing configurations only by final accuracy is often not enough.
This tool was built to make hardware-aware trade-off analysis easier and more systematic.

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
- `sample_experiments.json`: example input file

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
