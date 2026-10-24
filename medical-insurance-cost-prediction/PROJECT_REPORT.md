# Project Report: Medical Insurance Cost Prediction

## 1. Executive summary

This project develops an end-to-end regression workflow and a Streamlit web application to estimate medical insurance charges from age, sex, body mass index (BMI), number of dependents, smoking status, and US region. It is intended for education and portfolio demonstration only.

**Important:** Run the training pipeline before filling in the results section. This report intentionally does not claim any model scores that have not been measured.

## 2. Problem statement

Can a regression model estimate billed medical insurance charges using the attributes included in the Medical Cost Personal Dataset? The practical objective is to compare several model families under a consistent evaluation protocol and expose the trained model through an interactive web interface.

## 3. Dataset

Source: [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance).

The expected fields are:

- `age`: age of the primary beneficiary
- `sex`: sex category in the supplied dataset
- `bmi`: body mass index
- `children`: number of dependents covered
- `smoker`: smoking status
- `region`: US region
- `charges`: billed medical costs, used as the regression target

The training code validates the schema, converts numeric fields, normalizes category text, removes exact duplicate rows, and drops rows with missing required fields. Record the final row count produced by the run in the results section.

## 4. Tools and architecture

- Python for implementation
- pandas and NumPy for data manipulation
- scikit-learn for preprocessing, model training, cross-validation, and metrics
- Matplotlib for figures
- joblib for model serialization
- Streamlit for the web interface

The preprocessing transformations are inside each model pipeline. This is important because transformations must be fitted only on each training fold during cross-validation, rather than on the full dataset.

## 5. Methodology

1. Load and validate `insurance.csv`.
2. Separate the predictor columns from `charges`.
3. Create a reproducible train/test split using random state 42.
4. Compare Linear Regression, Ridge Regression, Random Forest, Extra Trees, and Histogram Gradient Boosting using K-fold cross-validation on the training set.
5. Select the candidate with the lowest mean cross-validation MAE.
6. Fit that candidate on the full training split.
7. Calculate MAE, RMSE, and R² once on the hold-out test set.
8. Calculate permutation importance on the hold-out set and export plots and CSV files.
9. Load the saved pipeline in Streamlit for interactive predictions.

## 6. Results — fill in after running the code

Run:

```bash
python train_model.py
```

Then copy the values from `reports/metrics.json` and `reports/model_comparison.csv` into this section.

| Result | Measured value |
|---|---|
| Rows after cleaning | See `dataset_rows_after_cleaning` in `reports/metrics.json` |
| Training rows | See `train_rows` |
| Test rows | See `test_rows` |
| Selected model | See `selected_model` |
| Test MAE | See `test_mae` |
| Test RMSE | See `test_rmse` |
| Test R² | See `test_r2` |

Do not substitute expected, copied, or fabricated values for the measured results from your run.

## 7. Interpretation

- MAE summarizes the typical magnitude of absolute prediction error in the target's currency units.
- RMSE gives additional weight to large errors.
- R² compares the model against a constant-mean baseline; a negative test-set R² is possible.
- Permutation importance estimates how predictive performance changes when one feature is shuffled. It is not a causal effect and may be affected by correlated predictors.

Use the generated actual-versus-predicted plot and model-comparison table to discuss observed strengths and weaknesses after running the experiment.

## 8. Web application

The Streamlit app includes four sections:

1. **Predict charges:** collect user inputs and display a model estimate.
2. **Dataset explorer:** inspect rows, descriptive statistics, and charge distributions.
3. **Model performance:** view computed hold-out metrics, cross-validation comparison, and permutation importance.
4. **About & limitations:** document intended use and cautions.

## 9. Limitations and ethics

The dataset is relatively small, historical, and US-specific. It does not capture all factors that affect actual insurance costs, policy design, provider prices, or changes over time. The model should not be treated as causal or as a substitute for a quote. The inclusion of sex and smoking status warrants particular care because predictive relationships can create fairness and legal concerns. Predictions may be unreliable for populations or input ranges not represented in the data.

## 10. Conclusion and future work

This project demonstrates an end-to-end machine-learning workflow from validation through deployment-style interaction. The model with the lowest cross-validation MAE is evaluated on a separate test split, and all metrics are computed by code. Future work could include hyperparameter tuning, uncertainty estimates, subgroup error analysis, automated tests, model monitoring, and a formal fairness and privacy review before any real-world use.

## References

1. Choi, M. *Medical Cost Personal Datasets*. Kaggle. https://www.kaggle.com/datasets/mirichoi0218/insurance
2. scikit-learn documentation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/
