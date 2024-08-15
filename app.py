from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from catboost import CatBoostRegressor

app = Flask(__name__)

# Load your trained CatBoost model
model = CatBoostRegressor()
model.load_model('models/catboost_model.cbm')

# @app.route('/') is called when a user navigates to the base URL of the application (e.g., http://localhost:5000/)
@app.route('/')
# def index() handles incoming requests to this URL
def index():
# renders the index.html template and returns it as the response to the client
    return render_template('index.html')
# @app.route('/predict', methods=['POST']) called whenever a POST request is made to the '/predict' URL
@app.route('/predict', methods=['POST'])
def predict():
    # treats <input type="file" name="file" accept=".csv" required> from index.html as a dictionary where name='file' is a key and the uploaded file is a value
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(file)
        
        # Ensure the data has the correct columns
        categorical = ['feature_1']
        numeric = ['feature_2', 'feature_3']
        
        # Prepare the data for prediction
        X = data[categorical + numeric]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Convert predictions to a list for easy rendering
        predictions_list = predictions.tolist()
        
        return render_template('results.html', predictions=predictions_list)

if __name__ == '__main__':
    app.run(debug=True)
