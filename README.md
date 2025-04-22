# Contingencies Screening

![Python Version](https://img.shields.io/badge/python-%3E=3.12-blue)

A repository housing scripts and utilities developed for the RTE-AIA Contingencies Screening project.

**(c) 2025 Grupo AIA for RTE**

---

## Overview

The Contingency Screening project addresses the critical need for an efficient method to identify potentially inaccurate contingency results from standard powerflow-based Security Analysis. These "suspect" contingencies warrant further investigation using more sophisticated powerflow tools, such as those incorporating approximate network dynamics and control system behavior over time.

This repository provides Python-based software to tackle this challenge. It is specifically designed to integrate with **Hades** (version 2) as the static powerflow engine and **DynaFlow** for dynamic powerflow simulations. Our approach leverages "supervised learning," not in a purely data-driven, black-box manner, but with a strong emphasis on incorporating power system engineering knowledge to build interpretable models.

The core idea involves a comparative analysis. We run both Hades and DynaFlow across a comprehensive set of operational snapshots and their respective contingencies. The results are then compared using a well-defined metric (a distance function quantifying the discrepancies). This metric becomes the target variable for our predictive models, which are built using a carefully selected set of predictor variables derived solely from Hades results.

The software currently implements two distinct modeling strategies:

* **Human-Interpretable Model (Median-Based Linear Regression):** This approach yields models that are highly transparent to domain experts. The relative influence of each predictor variable is directly observable, allowing for informed adjustments. For instance, if expert judgment suggests a variable should be inconsequential but the regression assigns it a non-zero weight (potentially due to overfitting), this weight can be manually overridden.

* **Machine Learning Models (GBM/EBM):** These models, based on Gradient Boosting Machines (GBM) or Explainable Boosting Machines (EBM), often offer superior predictive accuracy. While they operate more as "black boxes," techniques like SHAP (SHapley Additive exPlanations) values can provide insights into variable importance, although these explanations are typically instance-specific rather than globally representative.

The training phase is computationally intensive, requiring numerous Hades and DynaFlow simulations, followed by the transformation of DynaFlow outputs into a format comparable with Hades results, and finally, the model training itself. However, once trained, these models enable remarkably fast prediction of a "suspiciousness" score for each contingency. This speed is the key advantage, allowing for a hybrid approach: the rapid static powerflow (Hades) is used for initial screening of all contingencies, and the more accurate (but computationally demanding) dynamic powerflow (DynaFlow) is then selectively applied only to the high-scoring, potentially problematic cases.

Therefore, the software within this repository operates in two primary modes:

* **Model Training:** This is an infrequent but crucial process, typically performed every few weeks or months, to update the predictive models and mitigate the effects of data and model drift.

* **Screening:** This is a routine operation performed after each Security Analysis run with Hades. The trained model is used to predict a score for each contingency. These contingencies are then ranked based on their scores, and those exceeding a predefined threshold are flagged for detailed analysis with DynaFlow.

Complementary Jupyter Notebooks are also included for in-depth analysis and exploration of the obtained results.

## Project Structure

The repository is organized into the following high-level directories:

[comment]: <> (tree view obtained with: tree -d -L 3 contingencies-screening)
```
contingencies-screening/
├── doc
│   ├── devel
│   └── user
└── src
    └── contingencies_screening
        ├── analyze_loadflow
        ├── commons
        ├── notebook_analysis
        ├── prepare_basecase
        ├── run_dynawo
        └── run_loadflow
```

## Documentation

Comprehensive documentation is available under the `doc` directory:

* **[INSTALLATION](doc/user/INSTALLATION.md):** Provides detailed instructions on how to install the Python package and its dependencies. This includes information on setting up the required environments for Hades and DynaFlow.

* **[USAGE](doc/user/USAGE.md):** Explains how to use the core scripts, specifically `run_contingencies_screening` (the main pipeline for screening contingencies) and `train_test_loadflow_results` (the module for training and evaluating the predictive models). It covers input data formats, configuration options, and output interpretation.

* **[ANALYSIS](doc/user/ANALYSIS.md):** Details the process of preparing power system cases for analysis and how to utilize the provided Jupyter Notebooks. These notebooks facilitate both global analysis (e.g., model performance evaluation, contingency ranking visualization) and individual case studies (e.g., examining the results of specific contingencies).

* **[SUMMARY_CONCLUSIONS](doc/user/SUMMARY_CONCLUSIONS.md):** Presents key findings and conclusions derived from the modeling results obtained using datasets from June 2024, September 2024, December 2024, and March 2025. Further details and a more formal analysis will be available in an upcoming publication.

## Contact

For any questions or inquiries, please contact [omsg@aia.es].

