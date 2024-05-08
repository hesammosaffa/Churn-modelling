import preprocess
import streamlit as st
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir,'Models')


st.header('Software fo predicting customer loyalty in a store')

st.subheader('User Input')


# Slider for Credit Score
credit_score = st.number_input('Credit Score', min_value=350, max_value=850, step=1)

# Slider for Estimated Salary
estimated_salary = st.number_input('Estimated Salary', min_value=11.0, max_value=199993.0, step=1.5)

col1, col2, col3 = st.columns([1, 2, 2])

# Selectbox for Geography
with col1:
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    

# Selectbox for Gender
with col1:
    balance = st.number_input('Balance', min_value=0.0, max_value=250899.0, step=1.5)

# Number Input for Age
with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    

# Number Input for Number Of Products
with col2:
    num_of_products = st.number_input('Number Of Products', min_value=1, max_value=4, step=1)

# Number Input for Balance
with col3:
    geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
    

# Number Input for Tenure
with col3:
    tenure = st.number_input('Tenure', min_value=0, max_value=10, step=1)

# Checkbox for Has Credit Card
with col2:
    has_credit_card = st.checkbox('Has Credit Card')

# Checkbox for Is Active Member
with col3:
    is_active_member = st.checkbox('Is Active Member')

# Button for Calculation
btn = st.button('Calculate')


if btn:   
    if not os.listdir(file_path):
          st.error("Oops! Due to the large size of the models, we couldn't send them over. Please run the Project2.ipynb file to access the Model Output section!")

    else:        
        out = preprocess.process(credit_score,estimated_salary,age,num_of_products,balance,
                                tenure,has_credit_card,is_active_member,geography,gender)
        if out:
            st.success('The model predicts that this customer is loyal.')
            st.balloons()
        else:
            st.error('The model predicts that this customer is not loyal.')