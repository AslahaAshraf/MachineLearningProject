import streamlit as st
import joblib
import numpy as np

# Load model and scaler
adaboost = joblib.load("Employee_attrition_data.pkl")
scaler = joblib.load("scaler.pkl")

# Load label encoders
le = joblib.load('le.pkl')               #Gender
le1 = joblib.load('le1.pkl')             #Job Role
le2 = joblib.load('le2.pkl')             #Work-Life Balance
le3 = joblib.load('le3.pkl')             #Job Satisfaction
le4 = joblib.load('le4.pkl')             #Performance Rating
le5 = joblib.load('le5.pkl')             #Overtime
le6 = joblib.load('le6.pkl')             #Education Level
le7 = joblib.load('le7.pkl')             #Marital Status
le8 = joblib.load('le8.pkl')             #Job Level
le9 = joblib.load('le9.pkl')             #Company Size
le10 = joblib.load('le10.pkl')           #Remote Work
le11 = joblib.load('le11.pkl')           #Leadership Opportunities
le12 = joblib.load('le12.pkl')           #Innovation Opportunities
le13 = joblib.load('le13.pkl')           #Company Reputation
le14 = joblib.load('le14.pkl')           #Employee Recognition
le15 = joblib.load('le15.pkl')           #Attrition


st.title("Employee Attrition Prediction")
st.header("Enter Employee Details:")

# Numerical inputs
Age=st.number_input('Age', min_value=1)
Years_at_Company=st.number_input('Years at Company',min_value=1,max_value=60)
Monthly_Income=st.number_input('Monthly Income')
Number_of_Promotions=st.number_input('Number of Promotions',min_value=0)
Distance_from_Home=st.number_input('Distance from Home',min_value=1,max_value=100)
Number_of_Dependents=st.number_input('Number of Dependents',min_value=0)

# categorical inputs
Gender=st.selectbox('Gender',le.classes_)
Job_Role=st.selectbox('Job Role',le1.classes_)
Work_Life_Balance = st.selectbox('Work Life Balance', le2.classes_)
Job_Satisfaction=st.selectbox('Job Satisfaction',le3.classes_)
Performance_Rating=st.selectbox('Performance Rating',le4.classes_)
Overtime=st.selectbox('Overtime',le5.classes_)
Education_Level=st.selectbox('Education Level',le6.classes_)
Marital_Status=st.selectbox('Marital Status',le7.classes_)
Job_Level=st.selectbox('Job Level',le8.classes_)
Company_Size=st.selectbox('Company Size',le9.classes_)
Remote_Work=st.selectbox('Remote Work',le10.classes_)
Leadership_Opportunities=st.selectbox('Leadership Opportunities',le11.classes_)
Innovation_Opportunities=st.selectbox('Innovation Opportunities',le12.classes_)
Company_Reputation=st.selectbox('Company Reputation',le13.classes_)
Employee_Recognition=st.selectbox('Employee Recognition',le14.classes_)


# Encode cateorical features
Gender_enc=le.transform([Gender])[0]
Job_Role_enc=le1.transform([Job_Role])[0]
Work_Life_Balance_enc = le2.transform([Work_Life_Balance])[0]
Job_Satisfaction_enc=le3.transform([Job_Satisfaction])[0]
Performance_Rating_enc=le4.transform([Performance_Rating])[0]
Overtime_enc=le5.transform([Overtime])[0]
Education_Level_enc=le6.transform([Education_Level])[0]
Marital_Status_enc=le7.transform([Marital_Status])[0]
Job_Level_enc=le8.transform([Job_Level])[0]
Company_Size_enc=le9.transform([Company_Size])[0]
Remote_Work_enc=le10.transform([Remote_Work])[0]
Leadership_Opportunities_enc=le11.transform([Leadership_Opportunities])[0]
Innovation_Opportunities_enc=le12.transform([Innovation_Opportunities])[0]
Company_Reputation_enc=le13.transform([Company_Reputation])[0]
Employee_Recognition_enc=le14.transform([Employee_Recognition])[0]


# Combine into final feature vector (21 features)
features = [[
    Age,Years_at_Company,Monthly_Income,Number_of_Promotions,Distance_from_Home,Number_of_Dependents,Gender_enc,Job_Role_enc,
    Work_Life_Balance_enc,
    Job_Satisfaction_enc,Performance_Rating_enc,Overtime_enc,Education_Level_enc,Marital_Status_enc,Job_Level_enc,
    Company_Size_enc,Remote_Work_enc,Leadership_Opportunities_enc,Innovation_Opportunities_enc,Company_Reputation_enc,Employee_Recognition_enc
]]

# Scale the features
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict"):
    prediction = adaboost.predict(features_scaled)[0]
    result = "🟠 Stayed" if prediction == 1 else "🟢 Left"
    st.success(result)


# Predict
# if st.button("Predict Attrition"):
    # prediction = adaboost.predict(features_scaled)[0]
    # if prediction == 1:
        # st.success("Prediction: Employee is likely to leave (Attrition = Yes)")
    # else:
        # st.success("Prediction: Employee is likely to stay (Attrition = No)")





