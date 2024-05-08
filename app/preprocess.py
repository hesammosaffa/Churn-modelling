import pandas as pd
import os
import pickle

# Get the current directory path
current_dir = os.path.dirname(os.path.realpath(__file__))

# Function to process user data and make predictions
def process(credit_score, estimated_salary, age, num_of_products, balance, tenure, has_credit_card, is_active_member, geography, gender):
    # Convert boolean values to integer for modeling
    has_credit_card_value = 1 if has_credit_card else 0
    is_active_member_value = 1 if is_active_member else 0

    # Paths to the saved models
    path_Gender = os.path.join(current_dir, 'Models', 'le_Gender.pkl')
    path_Geography = os.path.join(current_dir, 'Models', 'le_Geography.pkl')
    path_rfc = os.path.join(current_dir, 'Models', 'rfc_mode.pkl')
    path_sc = os.path.join(current_dir, 'Models', 'StandardScaler.pkl')

    # Load the saved models
    with open(path_Gender, 'rb') as lion_Gender:
        model_Gender = pickle.load(lion_Gender)

    with open(path_Geography, 'rb') as lion_Geography:
        model_Geography = pickle.load(lion_Geography)

    with open(path_sc, 'rb') as sc_model:
        sc = pickle.load(sc_model)

    with open(path_rfc, 'rb') as rfc_model:
        rfc = pickle.load(rfc_model)

    # Prepare user data
    user_data = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card_value],
        'IsActiveMember': [is_active_member_value],
        'EstimatedSalary': [estimated_salary]
    }

    # Create a DataFrame from user data
    data = pd.DataFrame(user_data)

    # Transform categorical variables using Label Encoders
    data['Gender'] = model_Gender.transform(data['Gender'])
    data['Geography'] = model_Geography.transform(data['Geography'])

    # Scale the data
    data = sc.transform(data)

    # Make predictions using the Random Forest Classifier model
    y_pred = rfc.predict(data)

    return y_pred
