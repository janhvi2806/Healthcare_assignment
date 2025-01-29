from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the trained model and encoders
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])

        # Encode categorical columns
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_columns:
            if col in input_data:
                input_data[col] = label_encoders[col].transform([input_data[col].values[0]])[0]

        # Ensure input data has the same feature columns as the training data
        required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        input_data = input_data[required_columns]
        prediction = model.predict(input_data)[0]

        result = {"prediction": int(prediction)}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
