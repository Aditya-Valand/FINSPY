# from flask import Flask, send_from_directory, request, jsonify
# from flask_socketio import SocketIO, emit
# from flask_cors import CORS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.prompts import ChatPromptTemplate
# # from langchain.schema.output_parser import StrOutputParser
# import pandas as pd
# import joblib
# import fitz  # PyMuPDF
# import re
# import os
# app = Flask(__name__)
# CORS(app)  # Allow all domains for development; adjust for production

# # Load the model
# ml_model_path = r'D:\MaverickAI\backend\model\isolation_forest_model.pkl'
# ml_model = joblib.load(ml_model_path)
# # Define relevant features for ML model
# relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file"""
#     document = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(document.page_count):
#         page = document.load_page(page_num)
#         text += page.get_text()
#     return text

# def parse_transactions(text):
#     """Parse transaction amounts from extracted text"""
#     transaction_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b')
#     transactions = [float(amount.replace(',', '')) for amount in transaction_pattern.findall(text)]
#     return transactions

# def calculate_features(transactions):
#     """Calculate features from transactions"""
#     transaction_amount = transactions[-1]  # Assuming the last transaction is the one to be analyzed
#     average_transaction_amount = sum(transactions) / len(transactions)
#     frequency_of_transactions = len(transactions)
#     return [transaction_amount, average_transaction_amount, frequency_of_transactions]


# @app.route('/')
# def serve_index():
#     return send_from_directory(app.static_folder, 'index.html')

# @app.route('/<path:path>')
# def serve_static(path):
#     if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
#         return send_from_directory(app.static_folder, path)
#     else:
#         return send_from_directory(app.static_folder, 'index.html')
    
# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "Backend is running"}), 200   
     
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if request.content_type == 'multipart/form-data':
# #         file = request.files['file']
# #         # Process the file
# #         # ...
# #         # Return prediction result
# #         return jsonify({"message": "Prediction result from file"})
# #     elif request.content_type == 'application/x-www-form-urlencoded':
# #         transaction_amount = float(request.form['transaction_amount'])
# #         average_transaction_amount = float(request.form['average_transaction_amount'])
# #         frequency_of_transactions = int(request.form['frequency_of_transactions'])
        
# #         features = [transaction_amount, average_transaction_amount, frequency_of_transactions]
# #         input_df = pd.DataFrame([features], columns=['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions'])
        
# #         prediction = ml_model.predict(input_df)
# #         is_anomaly = 1 if prediction == -1 else 0
        
# #         result = {
# #             'is_anomaly': is_anomaly,
# #             'message': "Anomaly detected" if is_anomaly else "No anomaly detected"
# #         }
# #         return jsonify(result)
# #     return jsonify({"error": "Invalid request"}), 400
# @app.route('/predict', methods=['POST'])
# def predict():
#     print("Predict endpoint hit")
#     if 'file' in request.files and request.files['file'].filename != '':
#         file = request.files['file']
        
#         # Save the uploaded PDF file in the current directory or create tmp directory if not exists
#         tmp_dir = os.path.join(os.getcwd(), 'tmp')
#         if not os.path.exists(tmp_dir):
#             os.makedirs(tmp_dir)
        
#         pdf_path = os.path.join(tmp_dir, file.filename)
#         file.save(pdf_path)
        
#         # Extract text from the PDF
#         text = extract_text_from_pdf(pdf_path)
        
#         # Parse transactions from the text
#         transactions = parse_transactions(text)
        
#         if not transactions:
#             return jsonify({"error": "No transactions found in the provided PDF"})
        
#         # Calculate features from transactions
#         features = calculate_features(transactions)
        
#     else:
#         try:
#             transaction_amount = float(request.form['transaction_amount'])
#             average_transaction_amount = float(request.form['average_transaction_amount'])
#             frequency_of_transactions = int(request.form['frequency_of_transactions'])
#             features = [transaction_amount, average_transaction_amount, frequency_of_transactions]
#         except ValueError:
#             return jsonify({"error": "Invalid input values"})
        
#     # Create a DataFrame from the features
#     input_df = pd.DataFrame([features], columns=relevant_features)
    
#     # Predict anomaly
#     prediction = ml_model.predict(input_df)
#     is_anomaly = 1 if prediction == -1 else 0
    
#     result = {
#         'is_anomaly': is_anomaly,
#         'message': "Anomaly detected" if is_anomaly else "No anomaly detected"
#     }
#     return jsonify(result)
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import pandas as pd
import joblib
import fitz  # PyMuPDF
import re
import os
import logging
import pickle
# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
app.config['SECRET_KEY'] = 'FLASK_SECRET_KEY'  # Replace with a secure key

# Initialize SocketIO and CORS
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Google Generative AI model
GOOGLE_API_KEY = 'AIzaSyD6nEa53aEKH_NTv77xWaYX9Fq-8kPJsa0'  # Replace with your actual API key
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Define the prompt template and output parser for AI assistance
prompt = ChatPromptTemplate.from_template(
    "if the user want to know .\n\nUser: {message}\nAI:"
)
output_parser = StrOutputParser()
general_chain = prompt | model | output_parser

logging.basicConfig(level=logging.DEBUG)
# Load the ML model for anomaly detection
model_path = r'D:\MaverickAI\backend\model\isolation_forest_model3.pkl'
scaler_path = r'D:\MaverickAI\backend\model\scaler.pkl'

model = joblib.load(model_path)
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

# # Define helper functions
# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file"""
#     document = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(document.page_count):
#         page = document.load_page(page_num)
#         text += page.get_text()
#     return text

# def parse_transactions(text):
#     """Parse transaction amounts from extracted text"""
#     transaction_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b')
#     transactions = [float(amount.replace(',', '')) for amount in transaction_pattern.findall(text)]
#     return transactions

# def calculate_features(transactions):
#     """Calculate features from transactions"""
#     transaction_amount = transactions[-1]  # Assuming the last transaction is the one to be analyzed
#     average_transaction_amount = sum(transactions) / len(transactions)
#     frequency_of_transactions = len(transactions)
#     return [transaction_amount, average_transaction_amount, frequency_of_transactions]

# Define Flask routes
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Backend is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Manually input features
        transaction_amount = float(request.form['transaction_amount'])
        average_transaction_amount = float(request.form['average_transaction_amount'])
        frequency_of_transactions = int(request.form['frequency_of_transactions'])
        features = [transaction_amount, average_transaction_amount, frequency_of_transactions]
        
        # Validate feature values
        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({"error": "Invalid feature values"})
        
        # Create a DataFrame from the features
        input_df = pd.DataFrame([features], columns=relevant_features)
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        logging.debug(f"Features: {features}")
        logging.debug(f"DataFrame for prediction: {input_df}")
        logging.debug(f"Scaled features: {input_scaled}")
        
        # Predict anomaly
        prediction = model.predict(input_scaled)
        is_anomaly = 1 if prediction[0] == -1 else 0
    except ValueError:
        return jsonify({"error": "Invalid input values"})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "Error making prediction"})
    
    result = {
        'is_anomaly': is_anomaly,
        'message': "Anomaly detected" if is_anomaly else "No anomaly detected"
    }
    return jsonify(result)
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    user_message = data['message']
    print(f'Received message: {user_message}')
    response = general_chain.invoke({"message": user_message})
    emit('response', {'message': response}, room=request.sid)

if __name__ == '__main__':
    socketio.run(app, debug=True)
