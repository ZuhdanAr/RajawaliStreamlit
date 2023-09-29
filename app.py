import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import pickle

warnings.filterwarnings("ignore")

gendermapping = {
    'Female': 0,
    'Male': 1
}

edumapping = {
    'Bachelor': 0,
    'High School': 1,
    'Master': 2,
    'PhD': 3
}

# Load the pre-trained model
filename = 'best_regression_model.pkl'
model = pickle.load(open(filename, 'rb'))

# Home page
def home():
    st.title('Customer Spending Prediction')
    st.subheader('Made by: Team Rajawali')
    st.caption('Zuhdan Arif I Syafa Saphira I Zen Fikri')
    st.write('The customer spending prediction web app uses gender, age, purchase frequency,'
             'and income to accurately predict customer spending. In the real world, this model can help businesses make data-driven decisions.')
    st.subheader('Files')
    st.markdown(
        """
        <style>
        .stMarkdown a {
            color: red !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display hyperlink
    st.markdown("[Google Colab](https://colab.research.google.com/drive/1muh2AeRj54kfXghGNPS5Mi75Uw3zB36A?usp=sharing#scrollTo=x-jsJwAfA1Gy)")
    st.markdown('[CSV file](https://www.kaggle.com/datasets/goyaladi/customer-spending-dataset/download?datasetVersionNumber=1)')

# Data input and prediction page
def predict():
    st.title('Customer Spending Prediction')

    # Data inputation
    st.subheader('Input Data')
    name = st.text_input('Name', 'Your Name')
    gender = st.selectbox("Gender", ("Male", "Female"))
    age = st.number_input('Age', 0, 65)
    education = st.selectbox('Last Education', ("High School", 'Bachelor', 'Master', 'PhD'))
    income = st.number_input('Income', 0, 100000)
    purchase_frequency = st.number_input('Purchase Frequency', 0.0, 1.0)

    ## Perform prediction
    gender = gendermapping[gender]
    education = edumapping[education]
    input_data = pd.DataFrame([[age, gender, education, income, purchase_frequency]], columns=['age', 'gender', 'education', 'income', 'purchase_frequency'])
    prediction = model.predict(input_data)
    rounded_prediction = round(prediction[0], 3)

    st.subheader('Prediction')
    st.write(name + ', The predicted spending is: ')
    st.markdown(f"<h3 style='color: red; font-family: Arial; font-size: 52px;'>{rounded_prediction}</h3>", unsafe_allow_html=True)


# App navigation
def main():
    st.sidebar.title('Navigation')
    pages = ['Home', 'Prediction']
    choice = st.sidebar.selectbox('Go to', pages)

    if choice == 'Home':
        home()
    elif choice == 'Prediction':
        predict()


if __name__ == '__main__':
    main()
