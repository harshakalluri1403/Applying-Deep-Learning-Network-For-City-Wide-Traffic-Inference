from flask import Flask, request, jsonify
from flask_cors import CORS
import a # Import your existing Python script

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/process', methods=['POST'])
def process():
    data = request.json  # Get JSON data from frontend
    result = your_python_code.process_data(data)  # Call a function from your Python script
    return jsonify({"output": result})  # Return response as JSON

if __name__ == '__main__':
    app.run(debug=True)