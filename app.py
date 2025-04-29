# app.py
from flask import Flask, render_template, request, jsonify
from recommender import EnhancedMovieRecommender
import time
import os

app = Flask(__name__)
# Create a global variable for the recommender
recommender = None

@app.route('/')
def index():
    global recommender
    # Initialize recommender if not already done
    if recommender is None:
        start_time = time.time()
        recommender = EnhancedMovieRecommender()
        load_time = time.time() - start_time
        print(f"Model initialized in {load_time:.2f} seconds")
    
    # Get list of all movies for dropdown
    movies = recommender.get_all_movies()
    return render_template('index.html', movies=movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie')
    
    # Get recommendations
    start_time = time.time()
    recommendations = recommender.get_recommendations(movie_title)
    process_time = time.time() - start_time
    
    return render_template(
        'results.html', 
        selected_movie=movie_title,
        recommendations=recommendations,
        process_time=process_time
    )

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """API endpoint for recommendations"""
    movie_title = request.args.get('movie', '')
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    recommendations = recommender.get_recommendations(movie_title)
    return jsonify({
        "selected_movie": movie_title,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize the recommender before running the app
    recommender = EnhancedMovieRecommender()
    print("Model initialized successfully")
    
    app.run(debug=True, host='0.0.0.0', port=5000)