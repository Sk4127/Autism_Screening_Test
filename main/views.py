from django.shortcuts import render
import joblib  # For loading the saved XGBoost model
import numpy as np

# Load the trained XGBoost model
MODEL_PATH = "main/utililities/xgb_model.sav"
xgb_model = joblib.load(MODEL_PATH)

# Encoded lookup lists for ethnicity and relation
ENCODED_ETHNICITY = [
    'Asian', 'Black', 'Hispanic', 'Latino', 'Middle Eastern', 
    'Mixed', 'Native Indian', 'Others', 'Pacifica', 'South Asian', 
    'Turkish', 'White European', 'White-European'
]

ENCODED_RELATION = [
    'Family Member', 'Health Care Professional', 'Others', 
    'Parent', 'Relative', 'School and NGO', 'Self'
]

def test_aq10(request):
    if request.method == 'POST':
        data = request.POST

        try:
            # Retrieve and process input data
            user_name = data.get('user_name', 'Unknown')
            gender = 1 if data.get('gender') == 'm' else 0
            age = int(data.get('age', 0))
            ethnicity = data.get('ethnicity')
            jaundice = 1 if data.get('jaundice') == 'yes' else 0
            autism = 1 if data.get('autism') == 'yes' else 0
            relation = data.get('relation')

            # Encode ethnicity and relation with validation
            if ethnicity in ENCODED_ETHNICITY:
                ethnicity_index = ENCODED_ETHNICITY.index(ethnicity) + 1
            else:
                raise ValueError(f"Invalid ethnicity: {ethnicity}")

            if relation in ENCODED_RELATION:
                relation_index = ENCODED_RELATION.index(relation) + 1
            else:
                raise ValueError(f"Invalid relation: {relation}")

            # Retrieve AQ-10 question responses
            aq_scores = [
                int(data.get(f'q{i}', 0)) for i in range(1, 11)
            ]

            # Construct the feature vector
            feature_vector = np.array([[
                *aq_scores,  # A1 to A10
                age,         # Age_Years
                gender,      # Sex (encoded: Male -> 1, Female -> 0)
                ethnicity_index,
                jaundice,    # Jaundice
                autism,      # Family_mem_with_ASD
                relation_index
            ]])

            # Make a prediction
            prediction = xgb_model.predict(feature_vector)
            prediction_result = (
                "Likely to have Autism Spectrum Disorder" 
                if prediction[0] == 1 
                else "Unlikely to have Autism Spectrum Disorder"
            )

            # Context for the template
            context = {
                'user_name': user_name,
                'prediction': prediction_result,
                'aq_scores': aq_scores,
                'age': age,
                'gender': 'Male' if gender == 1 else 'Female',
                'relation': relation,
            }
            return render(request, 'test_result.html', context)

        except ValueError as e:
            # Handle invalid input gracefully
            return render(request, 'test_aq10.html', {
                'error': f"Input error: {str(e)}"
            })
        except Exception as e:
            # Handle unexpected errors
            return render(request, 'test_aq10.html', {
                'error': f"An unexpected error occurred: {str(e)}"
            })

    # Render the form page for GET requests
    return render(request, 'test_aq10.html')