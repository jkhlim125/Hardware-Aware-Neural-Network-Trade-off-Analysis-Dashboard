# Constraint-Aware Hardware Design Decision Support Dashboard

https://hardware-aware-neural-network-trade-off-analysis-dashboard-esw.streamlit.app/

## Overview

This project is a **constraint-aware decision analytics system** for evaluating hardware design trade-offs.

Rather than acting as a generic dashboard, the tool is designed to reflect how real engineering decisions are made:

- Designs must satisfy **hard constraints** before comparison  
- Trade-offs must be evaluated across **multiple conflicting objectives**  
- Final decisions should be **explainable and robust to changing priorities**

The system transforms heterogeneous inputs (experiment logs, structured trade-off tables, and RTL-derived results) into a unified decision pipeline that produces:

- Feasible vs rejected design candidates (with explicit reasons)
- Pareto-optimal configurations
- Interpretable recommendations
- Sensitivity-aware decision boundaries

---

## Motivation

In hardware and systems design, selecting a configuration is not a simple optimization problem.

A design may:
- Achieve higher accuracy but violate latency constraints  
- Reduce resource usage but degrade performance  
- Perform differently depending on changing priorities  

Most tools either:
- Focus on visualization, or  
- Apply fixed scoring without explaining trade-offs  

This project addresses that gap by building a **constraint-first decision system** that prioritizes:

- Explicit feasibility filtering  
- Multi-objective trade-off analysis  
- Explainable recommendation logic  

---

## System Design

The application follows a structured decision pipeline:

1. **Data ingestion and normalization**
   - Supports CSV, JSON, and RTL-derived inputs
   - Converts all inputs into a unified candidate schema

2. **Constraint filtering**
   - Applies hard engineering constraints (e.g., latency, accuracy)
   - Splits candidates into feasible and rejected sets
   - Tracks rejection reasons explicitly

3. **Pareto frontier analysis**
   - Identifies non-dominated configurations
   - Handles partial data without failure
   - Highlights trade-offs across objectives

4. **Recommendation layer**
   - Selects configurations based on decision modes:
     - Accuracy-focused
     - Latency-focused
     - Efficiency-focused
     - Balanced
   - Provides **explicit trade-off explanations**

5. **Sensitivity analysis**
   - Evaluates how decisions change with weight adjustments
   - Identifies **decision boundaries** between configurations

---

## Key Features

### Constraint-First Decision Flow

Unlike typical dashboards, this system enforces feasibility before optimization:

- Infeasible designs are removed early
- Each rejection is traceable to specific constraints

---

### Multi-Source Data Integration

The tool supports multiple input formats:

- Structured trade-off tables (`CSV`)
- Experiment logs (`JSON`)
- RTL simulation outputs (`CSV`)

All inputs are normalized into a unified representation, enabling consistent comparison.

---

### Robust Pareto Analysis

- Works with incomplete or partial data  
- Avoids failure when some metrics are missing  
- Marks Pareto-optimal candidates directly in the dataset  

---

### Explainable Recommendations

Each recommendation includes:

- Why the configuration was selected  
- What trade-offs were made  
- How it compares to alternative candidates  

This shifts the system from **result reporting → decision explanation**

---

### Sensitivity-Aware Decision Making

The system can:

- Sweep decision weights  
- Track configuration changes  
- Identify transition points where optimal choices shift  

This provides insight into **decision robustness**, not just static results.

---

## Example Workflow

1. Load candidate data (e.g., pruning experiments or RTL results)  
2. Apply constraints (e.g., latency ≤ 1500 cycles, accuracy ≥ 90%)  
3. Inspect feasible candidates and rejection reasons  
4. Analyze Pareto-optimal configurations  
5. Select a recommendation mode  
6. Evaluate sensitivity to changing priorities  

---

## Project Structure

```text
stream/
├── app.py                  # Streamlit UI (decision flow)
├── analysis_engine.py      # Constraints, Pareto, scoring
├── parsers.py              # Data ingestion and normalization
├── recommendation.py       # Recommendation logic + explanations
├── sample_data/
│   ├── sample_tradeoff.csv
│   ├── sample_experiments.json
│   └── sample_rtl_results.csv
├── README.md
└── requirements.txt
```
Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
Technical Highlights
	•	Constraint-based filtering with explicit rejection tracking
	•	Generic Pareto frontier computation across arbitrary objectives
	•	Robust handling of missing or partial data
	•	Modular pipeline separating parsing, analysis, and recommendation
	•	Interactive UI designed for decision workflows rather than visualization

⸻

Takeaways

This project demonstrates how raw engineering data can be transformed into a structured decision-making system.

Key focus areas:
	•	Turning data into actionable decisions
	•	Making trade-offs explicit and explainable
	•	Designing systems that remain robust under incomplete information

⸻

Future Work
	•	Integrate predictive modeling for missing metrics (e.g., accuracy estimation)
	•	Extend to larger-scale design space exploration
	•	Connect directly with hardware synthesis or training pipelines
	•	Add automated trade-off reporting for design documentation
