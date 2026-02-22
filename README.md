# Predictive Maintenance for Wind Turbines

This project applies machine learning techniques to real-time sensor data from wind turbines to enable **predictive maintenance**. The primary objective is to detect early signs of failure and minimize unexpected breakdowns, specifically focusing on generalizing models for unseen turbines (e.g., "Wind Turbine C").

## Project Overview
As wind energy becomes a central pillar of renewable energy, maintaining turbine efficiency is critical. This project explores how high-dimensional sensor data (957 features) can be leveraged to move from reactive to predictive maintenance strategies using both Supervised and Unsupervised learning.

## Methodology

### 1. Data Analysis & Feature Engineering
- Processed high-dimensional, anonymized datasets from German wind farms.
- Performed feature selection to identify variables following a normal distribution for model reliability.
- Utilized **t-SNE** (t-Distributed Stochastic Neighbor Embedding) to reduce dimensionality for cluster visualization.

### 2. Supervised Learning: Random Forest
- **Model:** Random Forest Classifier.
- **Goal:** Classify time-series data into "Normal" vs. "Faulty" states.
- **Result:** High accuracy when labeled data was available, proving effective for known failure modes.

### 3. Unsupervised Learning: DBSCAN
- **Model:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
- **Goal:** Anomaly detection without prior labels.
- **Result:** Superior flexibility in detecting subtle, repeated short-term failures and identifying outliers in unseen data.


## Key Findings
* **Generalization:** The models successfully identified anomalies in "Wind Turbine C," which was entirely excluded from the training phase.
* **DBSCAN vs. Random Forest:** While Random Forest is excellent for classification, **DBSCAN** proved to be the more robust choice for real-world monitoring where labels are often unavailable or incomplete.

## File Structure
* `Project_ML.ipynb`: Jupyter Notebook containing the end-to-end pipeline (Preprocessing, Training, Visualization).
* `Darshan_ML_Report.pdf`: Comprehensive technical report detailing the findings and methodology.
* `pra_mal_s25.pdf`: Project specifications and academic guidelines.

## Install required Python packages:
* pip install pandas numpy scikit-learn plotly matplotlib

## How to run the code
* Run the Project_ML.ipynb notebook in a Jupyter environment.
