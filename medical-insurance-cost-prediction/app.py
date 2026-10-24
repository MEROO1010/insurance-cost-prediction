import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Cost Estimator", page_icon="🩺", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


def build_row(age, sex, bmi, children, smoker, region):
    # Must match add_features() in the notebook
    return pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region,
        "obese": int(bmi >= 30),
        "smoker_obese": int(smoker == "yes" and bmi >= 30),
    }])


model = load_model()

st.title("Medical insurance cost estimator")
st.caption("A learning project, not a pricing or medical tool. Estimates have wide error margins.")

c1, c2 = st.columns(2)
age = c1.slider("Age", 18, 64, 35)
bmi = c2.slider("BMI", 15.0, 50.0, 27.0, 0.1)
children = c1.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = c2.radio("Smoker", ["no", "yes"], horizontal=True)
sex = c1.selectbox("Sex", ["female", "male"])
region = c2.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

row = build_row(age, sex, bmi, children, smoker, region)
estimate = float(model.predict(row)[0])
st.metric("Estimated yearly charges", f"${estimate:,.0f}")

st.subheader("What-if: change one thing")
alt = row.copy()
alt_smoker = "no" if smoker == "yes" else "yes"
alt_row = build_row(age, sex, bmi, children, alt_smoker, region)
alt_est = float(model.predict(alt_row)[0])
st.write(f"If smoking status were **{alt_smoker}**: **${alt_est:,.0f}** "
         f"({alt_est - estimate:+,.0f} vs. current).")
