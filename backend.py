from flask import Flask, jsonify, request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the variables at the top
user_profiles, tfidf_matrix, news_df = {}, None, None

# Attempt to load precomputed data
try:
    with open('user_profiles.pkl', 'rb') as f:
        user_profiles = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('news_df.pkl', 'rb') as f:  # Assuming news details are saved in this file
        news_df = pickle.load(f)
except Exception as e:
    app.logger.error(f"Failed to load models: {e}")

@app.route('/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    if tfidf_matrix is None or user_profiles is None or news_df is None:
        app.logger.error("Model data not loaded or is incomplete.")
        return jsonify({"error": "Model data not loaded"}), 500

    if user_id not in user_profiles:
        return jsonify({"error": "User profile not found"}), 404

    try:
        user_vector = user_profiles[user_id]
        user_vector = np.asarray(user_vector)  # Convert to ndarray if it's not already
        if user_vector.ndim == 1:
            user_vector = user_vector.reshape(1, -1)

        similarities = cosine_similarity(user_vector, tfidf_matrix)
        recommended_indices = similarities.argsort()[0][-5:][::-1]

        # Fetch the details of the recommended articles
        recommended_articles = news_df.iloc[recommended_indices][['Title', 'Abstract']].to_dict(orient='records')
        return jsonify(recommended_articles)
    except Exception as e:
        app.logger.error(f"Error processing recommendations for {user_id}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
