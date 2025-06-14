import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")
st.title("\U0001F4CA HR Analytics Dashboard")

uploaded_file = st.file_uploader("Upload your HR CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("### \U0001F9FE Column Preview")
    st.write(df.columns.tolist())

    st.subheader("\U0001F680 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", df.shape[0])

    if 'Attrition' in df.columns:
        attrition = df[df['Attrition'].str.lower() == 'yes'].shape[0] / df.shape[0]
        col2.metric("Attrition Rate", f"{attrition*100:.2f}%")
    else:
        col2.metric("Attrition Rate", "N/A")

    if 'JobSatisfaction' in df.columns:
        col3.metric("Avg Satisfaction", round(df['JobSatisfaction'].mean(), 2))
    else:
        col3.metric("Avg Satisfaction", "N/A")

    if 'YearsAtCompany' in df.columns:
        col4.metric("Avg Tenure", round(df['YearsAtCompany'].mean(), 2))
    else:
        col4.metric("Avg Tenure", "N/A")

    st.markdown("---")

    st.subheader("\U0001F4CA Attrition by Department")
    if 'Attrition' in df.columns and 'Department' in df.columns:
        attr_dept = df[df["Attrition"].str.lower() == "yes"]['Department'].value_counts()
        st.bar_chart(attr_dept)

    st.subheader("\U0001F4CA Gender Distribution")
    if 'Gender' in df.columns:
        gender = df['Gender'].value_counts().reset_index()
        gender.columns = ['Gender', 'Count']
        fig = px.pie(gender, names='Gender', values='Count', title='Gender Distribution')
        st.plotly_chart(fig)

    st.subheader("\U0001F4C9 Job Satisfaction vs Attrition")
    if 'Attrition' in df.columns and 'JobSatisfaction' in df.columns:
        fig = px.box(df, x='Attrition', y='JobSatisfaction', color='Attrition',
                     title="Job Satisfaction by Attrition")
        st.plotly_chart(fig)

    st.subheader("\U0001F4B8 Avg Monthly Income by Department")
    if 'MonthlyIncome' in df.columns and 'Department' in df.columns:
        income_dept = df.groupby('Department')['MonthlyIncome'].mean().reset_index()
        fig = px.bar(income_dept, x='Department', y='MonthlyIncome',
                     title='Avg Monthly Income by Department')
        st.plotly_chart(fig)

    st.subheader("\U0001F9E0 Correlation Heatmap")
    num_df = df.select_dtypes(include='number')
    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # ------------------ ML Prediction Section ------------------
    st.markdown("---")
    st.subheader("\U0001F52E Predict Employee Attrition")

    if st.checkbox("\U0001F4DD Show ML-Based Predictions"):
        df_ml = df.copy().dropna()

        if 'Attrition' not in df_ml.columns:
            st.warning("⚠️ 'Attrition' column missing.")
            st.stop()

        df_ml['Attrition'] = df_ml['Attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

        features = ['Age', 'MonthlyIncome', 'JobLevel', 'JobSatisfaction', 'YearsAtCompany', 'OverTime']
        df_ml = df_ml[features + ['Attrition']]

        for col in df_ml.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col])

        X = df_ml.drop('Attrition', axis=1)
        y = df_ml['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"✅ Model Accuracy: {acc*100:.2f}%")

        df_pred = df.copy()
        df_pred_features = df[features].copy()

        for col in df_pred_features.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_pred_features[col] = le.fit_transform(df_pred_features[col].astype(str))

        df_pred['Predicted Attrition'] = model.predict(df_pred_features)
        df_pred['Attrition Probability (%)'] = (model.predict_proba(df_pred_features)[:,1] * 100).round(2)

        st.dataframe(df_pred[['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany',
                              'Predicted Attrition', 'Attrition Probability (%)']].head(20))

        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", data=csv,
                           file_name="attrition_predictions.csv", mime='text/csv')

    # ------------------ PDF Generation ------------------
    st.subheader("📄 Generate PDF Report")

    def generate_pdf_report(df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="HR Analytics Report", ln=True, align='C')
        pdf.ln(10)

        total_employees = df.shape[0]
        pdf.cell(200, 10, txt=f"Total Employees: {total_employees}", ln=True)

        if 'YearsAtCompany' in df.columns:
            avg_tenure = df['YearsAtCompany'].mean()
            pdf.cell(200, 10, txt=f"Avg Tenure: {avg_tenure:.2f} years", ln=True)
        if 'JobSatisfaction' in df.columns:
            avg_sat = df['JobSatisfaction'].mean()
            pdf.cell(200, 10, txt=f"Avg Job Satisfaction: {avg_sat:.2f} / 4", ln=True)
        if 'Attrition' in df.columns:
            attr_rate = df[df['Attrition'].str.lower() == 'yes'].shape[0] / total_employees
            pdf.cell(200, 10, txt=f"Attrition Rate: {attr_rate*100:.2f}%", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(200, 10, txt="Data Insights", ln=True)
        pdf.set_font("Arial", size=11)

        if 'Attrition' in df.columns and 'Department' in df.columns:
            top_dept = df[df['Attrition'].str.lower() == 'yes']['Department'].value_counts().idxmax()
            pdf.multi_cell(0, 10, f"- Highest attrition: {top_dept}")

        if 'MonthlyIncome' in df.columns and 'Department' in df.columns:
            top_income_dept = df.groupby('Department')['MonthlyIncome'].mean().idxmax()
            top_income_val = df.groupby('Department')['MonthlyIncome'].mean().max()
            pdf.multi_cell(0, 10, f"- Highest avg salary: {top_income_dept} (${top_income_val:.2f}/month)")

        if 'JobSatisfaction' in df.columns:
            low_satis = df[df['JobSatisfaction'] <= 2]
            low_satis_attr = low_satis[low_satis['Attrition'].str.lower() == 'yes']
            if not low_satis.empty:
                percent = (low_satis_attr.shape[0] / low_satis.shape[0]) * 100
                pdf.multi_cell(0, 10, f"- Low satisfaction group has {percent:.2f}% attrition rate.")

        if 'Department' in df.columns:
            fig, ax = plt.subplots()
            df['Department'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title("Employees by Department")
            img_path = os.path.join(tempfile.gettempdir(), "dept_chart.png")
            fig.savefig(img_path)
            plt.close(fig)
            pdf.image(img_path, x=10, y=120, w=180)

        return pdf

    if st.button("📥 Generate & Download PDF"):
        try:
            st.info("⏳ Generating report...")
            pdf = generate_pdf_report(df)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf.output(tmp_file.name)
                with open(tmp_file.name, "rb") as f:
                    st.success("✅ Report ready. Click below to download:")
                    st.download_button("⬇️ Download HR Report", f, file_name="HR_Analytics_Report.pdf")
        except Exception as e:
            st.error(f"❌ Failed to generate PDF: {e}")
