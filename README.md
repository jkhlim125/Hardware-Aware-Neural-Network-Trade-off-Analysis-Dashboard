Experiment Log Anomaly Dashboard

An interactive dashboard for analyzing experiment results, detecting anomalous runs, and improving reproducibility.

https://projectsdashbord-epqfjyyegcxytbhlcsfizy.streamlit.app/

⸻

Overview

Machine learning experiments often produce inconsistent results across runs, making it difficult to identify performance issues or compare outcomes effectively.

This project provides a structured dashboard that enables users to analyze experiment logs, detect anomalous runs, and understand performance differences across multiple metrics.

⸻

Problem

Repeated experiments frequently show variation in performance due to randomness, hyperparameter sensitivity, or system-level factors.

Without proper tools, it is difficult to:
	•	Identify which runs are abnormal
	•	Compare performance across multiple metrics
	•	Understand relationships between metrics
	•	Extract meaningful insights from raw logs

⸻

Solution

This dashboard allows users to upload experiment logs and perform structured analysis through:
	•	Run-level performance comparison
	•	Statistical anomaly detection using z-score
	•	Metric selection and filtering
	•	Correlation analysis across metrics
	•	Automated summary generation

⸻

Key Features
	•	Detects anomalous runs based on statistical deviation from the mean
	•	Enables comparison between individual runs and average performance
	•	Provides normalized comparison for cross-metric analysis
	•	Visualizes relationships using scatter plots and correlation heatmaps
	•	Generates structured summaries for faster interpretation
	•	Supports CSV-based workflow for easy integration

⸻

Tech Stack
	•	Python
	•	Streamlit
	•	Pandas
	•	NumPy
	•	Plotly

⸻

How to Run Locally

pip install -r requirements.txt
streamlit run app.py

____

Project Motivation

This project was developed to address the challenge of analyzing unstable experiment results and improving reproducibility.

The goal was to move beyond simple visualization and build a tool that supports structured analysis and decision-making based on experimental data.

⸻

Key Takeaways
	•	Transformed raw experiment logs into an interactive analysis tool
	•	Applied statistical methods to identify anomalous behavior
	•	Designed workflows for comparing performance across multiple metrics
	•	Built and deployed a usable web-based data analysis application

⸻

Author

(Kiheon Lim)
