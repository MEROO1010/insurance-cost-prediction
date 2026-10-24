[README.md](https://github.com/user-attachments/files/33904312/README.md)
# Medical Insurance Cost Prediction (with explanations)

Predicting individual medical insurance charges from `age, sex, bmi, children, smoker, region`, and explaining what drives the prediction.

**Live demo:** _add your Streamlit Cloud link here_
**Screenshots:** _add 2-3 images here (key EDA chart, model comparison, the app)_

## Key findings
- Smokers pay about **3.8x** more on average than non-smokers.
- BMI matters mostly **for smokers**: smokers with BMI >= 30 average about $41.6k vs about $21.4k for smokers below 30. For non-smokers the gap is small ($8.9k vs $8.0k).
- Adding one engineered feature (`smoker_obese`) moved a plain linear model from R2 0.72 to 0.84 in cross-validation.
- Tuned Gradient Boosting and the linear model with that feature are essentially tied (CV RMSE about 4,700). The insight mattered more than the model.
- Removing `sex` and `region` did not hurt accuracy (test R2 0.92 vs 0.918), so a fairer model costs nothing here.

## Approach
1. Cleaning (1 duplicate removed, no missing values)
2. EDA and feature engineering
3. Model comparison with 5-fold CV (Linear, Ridge, Random Forest, Gradient Boosting)
4. Hyperparameter tuning, error analysis, permutation importance (+ optional SHAP)
5. Streamlit app with a what-if comparison

## Limitations
About 1,300 rows from a US setting, no clinical history, and the largest errors come from costly cases the data cannot explain. This is a learning project, not a pricing or medical tool.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook insurance_cost_analysis.ipynb   # run all cells (creates model.joblib)
streamlit run app.py
```
