#student performance dashboard
#display the data
#show student performance - trends
#charts and filters

import streamlit as st
import pandas as pd
import plotly.express as px

#page - config
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Dashboard")
#load data
df = pd.read_csv("student_performance.csv")
#sidebar filters
st.sidebar.header("Filters")
#filtered data as per grade
grade_filter = st.sidebar.multiselect("Select Grade", options=df["grade"].unique())
#filtered data as per gender
gender_filter = st.sidebar.multiselect("Select Gender", options=df["gender"].unique())
#apply filters
if grade_filter:
    df = df[df["grade"].isin(grade_filter)]
if gender_filter:
    df = df[df["gender"].isin(gender_filter)]
#display data
st.subheader("Filtered Student Data")
st.dataframe(df)

#charts
st.subheader("Performance Trends")
#average score by grade
avg_score_grade = df.groupby("grade")["final_grade"].mean().reset_index()
fig_grade = px.bar(avg_score_grade, x="grade", y="final_grade", title="Average Score by Grade",color="grade" )
st.plotly_chart(fig_grade, use_container_width=True)
#average score by gender line chart
avg_score_gender = df.groupby("gender")["final_grade"].mean().reset_index()
fig_gender = px.bar(avg_score_gender, x="gender", y="final_grade", title="Average Score by Gender")
st.plotly_chart(fig_gender, use_container_width=True)



#linear regression - final grade vs previour_grade, gender, study time, internet access, attendance_percentage

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[["previous_grade", "study_hours_per_day", "attendance_percentage"]]
y = df["final_grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
#predictions
y_pred = model.predict(X_test)
#display predictions
st.subheader("Predicted Final Grades")
pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
st.dataframe(pred_df)




#new data input for prediction
st.subheader("Predict Final Grade for New Student")
previous_grade_input = st.number_input("Previous Grade", min_value=0, max_value=100)
study_hours_input = st.number_input("Study Hours per Day", min_value=0, max_value=24)
attendance_input = st.number_input("Attendance Percentage", min_value=0, max_value=100)
if st.button("Predict Final Grade"):
    new_data = [[previous_grade_input, study_hours_input, attendance_input]]
    predicted_grade = model.predict(new_data)[0]
    st.write(f"Predicted Final Grade: {predicted_grade:.2f}")
