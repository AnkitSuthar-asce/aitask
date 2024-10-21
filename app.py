import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel
import torch
from langchain.document_loaders import WebBaseLoader
from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data from the URL
urls = ["https://brainlox.com/courses/category/technical"]
loader = WebBaseLoader(urls)
data = loader.load()

if data:
    print(data[0].page_content)
else:
    print("No data loaded. The loader could not process the content.")

# Load the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Store texts and embeddings
texts = [data[0].page_content]
embeddings = np.array([get_embeddings(texts[0])])  # Replace with actual embeddings

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

class Chatbot(Resource):
    def post(self):
        user_input = request.json.get('query')
        user_embedding = get_embeddings(user_input)

        # Calculate cosine similarity between user input and stored embeddings
        similarities = cosine_similarity([user_embedding], embeddings)
        best_match_index = np.argmax(similarities)

        # Get the best matching text
        response_text = texts[best_match_index]
        return jsonify({"response": response_text})

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Add resource to the API
api.add_resource(Chatbot, '/chat')

if __name__ == '__main__':
    app.run(debug=True)
