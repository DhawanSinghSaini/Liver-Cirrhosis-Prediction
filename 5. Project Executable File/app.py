from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('liver_model.pkl')

# Replace this list with your actual 39 feature names in the correct order
features = [
    "serial_number", "age", "gender", "place_of_residence", "alcohol_consumption_years",
    "alcohol_consumption_quantity", "alcohol_type", "hepatitis_b_status", "hepatitis_c_status",
    "diabetes_status", "obesity_status", "family_cirrhosis_history", "total_cholesterol",
    "triglycerides", "ldl_cholesterol", "hdl_cholesterol", "hemoglobin_gdl", "pcv_percent",
    "mcv_femtoliters_per_cell", "white_blood_cell_count", "polymorphs_percent", "lymphocytes_percent",
    "monocytes_percent", "eosinophils_percent", "basophils_percent", "platelet_count_lakhs_per_mm",
    "total_bilirubin_mgdl", "direct_bilirubin_mgdl", "indirect_bilirubin_mgdl", "total_protein_gdl",
    "albumin_gdl", "globulin_gdl", "albumin_globulin_ratio", "alkaline_phosphatase_ul",
    "sgot_ast_ul", "sgpt_alt_ul", "usg_abdomen_liver_condition", "systolic_bp", "diastolic_bp"
]

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in features]
        prediction = model.predict([input_data])[0]

        prediction_text = {
            0: "No Liver Cirrhosis",
            1: "Mild Liver Cirrhosis",
            2: "Severe Liver Cirrhosis"
        }.get(prediction, "Unknown Prediction")

        return render_template('index.html', features=features, prediction=prediction_text)

    except Exception as e:
        return render_template('index.html', features=features, prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
