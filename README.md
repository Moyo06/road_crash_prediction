# Road Crash Severity Prediction

## Project Overview
Road traffic collisions are a major public safety concern. This project develops a supervised Machine Learning classification model to predict the severity of road collisions (**Fatal, Serious, or Slight**) using environmental, temporal, and vehicle-related factors. 

Specifically, we trained a **Random Forest Classifier** (using Python, `pandas`, and `scikit-learn`) on the UK Department for Transport (DfT) STATS19 open data for 2023 and 2024. We used a multiclass classification approach, applying `class_weight='balanced'` to address the severe class imbalance in the dataset (where Fatal crashes make up only 1.5% of the data).

### Machine Learning & Technical Stack
*   **Algorithm:** Random Forest Classifier (Ensemble Learning)
*   **Task:** Multiclass Classification (1 = Fatal, 2 = Serious, 3 = Slight)
*   **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
*   **Evaluation Metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Gini Feature Importance.

## Team Members & Roles
This project was a collaborative effort divided into four main stages:

*   **Iyanda Moyosore(Data Cleaning & Merging):** Handled data quality assessment, handled missing/unknown codes, aggregated vehicle data to the collision level, and merged the 2023 and 2024 datasets into a single clean CSV.
*   **Goodness Kolawole (EDA & Feature Engineering):** Conducted Exploratory Data Analysis to uncover patterns in weather, lighting, and time. Decoded categorical variables using the DfT data guide and analyzed the severe class imbalance in the dataset (e.g., KSI ratio).
*   **Kenechukwu Uba (Model Development):** Built and trained the `RandomForestClassifier`. Handled missing values via median imputation, used `class_weight='balanced'` to address data imbalance, and experimented with hyperparameters (`n_estimators`, `max_depth`).
*   **Morgan Attoe (Model Evaluation):** Evaluated the best model (`max_depth=10`). Generated the Confusion Matrix, analyzed Feature Importances, and documented the model's limitations.

## Repository Structure
*   `1_Cleaning_and_Merging.ipynb` – Notebook for data cleaning and integration.
*   `02_eda.ipynb` – Notebook for Exploratory Data Analysis and visualizations.
*   `03_model_development.ipynb` – Notebook for Random Forest model training and tuning.
*   `04_model_evaluation.ipynb` – Notebook for final model evaluation and feature importance.
*   `data/` – Folder containing the raw CSV files, the DfT Data Guide, and the final merged dataset (`collisions_merged_clean.csv`).

## Key Results & Findings
*   **Data Imbalance:** The dataset is highly imbalanced (75.1% Slight, 23.3% Serious, 1.5% Fatal). Using `class_weight='balanced'` was crucial to prevent the model from just predicting "Slight" for every crash.
*   **Model Performance:** The baseline model (100 trees) achieved **69% overall accuracy**, but it struggled to identify Fatal crashes (1% Recall). 
*   **Improvement via Tuning:** By restricting the model to `max_depth=10`, the model was able to correctly predict **59% of Fatal crashes** (Recall), which is a massive improvement for real-world road safety planning, even though overall accuracy dropped to 52%. *In road safety, catching fatal crashes is more important than overall accuracy.*
*   **Top Predictors:** The most important features driving crash severity were:
    1.  `did_police_officer_attend_scene_of_accident` (proxy for severity)
    2.  `speed_limit`
    3.  `urban_or_rural_area`
*   **Conclusion:** Higher speed limits and rural roads are strongly associated with more severe collision outcomes. Road safety planners should target these specific areas for intervention.

## Limitations
*   **Class Imbalance:** Even with balanced weights, predicting the rare "Fatal" class remains difficult due to the lack of data.
*   **Missing Data:** Missing driver ages were imputed using the median, which may introduce slight bias.
*   **Data Scope:** The model relies on STATS19 police reporting, which may contain human error or under-reporting of minor collisions.

## How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
3. Open the notebooks in numerical order (1 to 4) and run the cells. The datasets are already provided in the `data/` folder.
