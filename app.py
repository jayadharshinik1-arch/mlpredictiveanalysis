from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the exported pipeline (preprocessor + model)
with open('logistic_regression_model.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert request data to Pandas DataFrame
        # If input is a single dictionary, wrap it in a list
        if isinstance(data, dict):
            data = [data]
        
        input_df = pd.DataFrame(data)
        
        # Generate prediction
        predictions = model_pipeline.predict(input_df)
        
        # Return predictions as JSON
        return jsonify({
            'predictions': predictions.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
if __name__ == "__main__":
    app.run(debug=True)
