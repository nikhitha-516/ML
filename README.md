# Machine Learning Assignment 2

## 1. Mandatory Submission Links
- GitHub Repository Link: `ADD_YOUR_GITHUB_REPO_LINK_HERE`
- Live Streamlit App Link: `ADD_YOUR_STREAMLIT_APP_LINK_HERE`
- BITS Virtual Lab Screenshot: add screenshot file in repo and in final PDF (example path: `inv/bits_lab_screenshot.png`)

## 2. Problem Statement
Build and compare 6 classification models on one common dataset, evaluate them using required metrics, and deploy an interactive Streamlit app for model selection, evaluation, and prediction.

## 3. Dataset Description
- Dataset Name: Cardiovascular Disease Dataset
- Source: Kaggle / open-source mirror
- Kaggle URL: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- Open-source CSV URL used in code: https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv
- Problem Type: Binary Classification
- Target: Cardio (`0 = no disease`, `1 = disease`)
- Instances used (after cleaning): 68,641
- Features used: 14 (`11` base + `3` engineered: `age_years`, `bmi`, `pulse_pressure`)
- Why selected: Satisfies assignment constraints (instances > 500 and features >= 12).

## 4. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## 5. Comparison Table (Required Metrics)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7224 | 0.7888 | 0.7495 | 0.6592 | 0.7014 | 0.4473 |
| Decision Tree | 0.6248 | 0.6247 | 0.6208 | 0.6209 | 0.6208 | 0.2495 |
| kNN | 0.7090 | 0.7655 | 0.7175 | 0.6793 | 0.6979 | 0.4182 |
| Naive Bayes | 0.7072 | 0.7789 | 0.7541 | 0.6056 | 0.6717 | 0.4212 |
| Random Forest (Ensemble) | 0.7121 | 0.7721 | 0.7181 | 0.6880 | 0.7028 | 0.4242 |
| XGBoost (Ensemble) | 0.7301 | 0.7988 | 0.7545 | 0.6737 | 0.7118 | 0.4622 |

## 6. Observations on Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline; high precision and balanced overall performance. |
| Decision Tree | Lowest performance; likely overfitting and weaker generalization than other models. |
| kNN | Better than Decision Tree and Naive Bayes F1, but below Logistic Regression and XGBoost. |
| Naive Bayes | Good precision and AUC, but recall is lower than most other models. |
| Random Forest (Ensemble) | Stable ensemble performance; better than single tree and close to Logistic Regression. |
| XGBoost (Ensemble) | Best overall metrics in this run (highest Accuracy, AUC, F1, MCC). |

## 7. Streamlit Features Implemented
- CSV test dataset upload option (`.csv`)
- Model selection dropdown for all 6 models
- Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix visualization
- Classification report table
- URL-based dataset loading when no file is uploaded (cardiovascular dataset source URL)

## 8. Project Structure
```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- default_test.csv
|   |-- sample_upload_test.csv
|-- model/
|   |-- train_models.py
|   |-- metrics.csv
|   |-- feature_columns.json
|   |-- dataset_info.json
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- random_forest_ensemble.pkl
|   |-- xgboost_ensemble.pkl
```

## 9. How to Run Locally
```bash
pip install -r requirements.txt
python3 model/train_models.py
streamlit run app.py
```

## 10. Streamlit Community Cloud Deployment
1. Push this repository to GitHub.
2. Open https://streamlit.io/cloud and sign in.
3. Click **New app**.
4. Select repository and branch.
5. Set main file path as `app.py`.
6. Deploy and copy the live app URL.

## 11. Notes for Final PDF Submission
- Include GitHub repo link.
- Include live Streamlit app link.
- Include BITS Virtual Lab screenshot.
- Include this full README content in the PDF.
