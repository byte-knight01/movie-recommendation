# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommender import EnhancedMovieRecommender
import time
import os
import threading

app = Flask(__name__)
# Create a global variable for the recommender
recommender = None
model_loading = False
model_ready = False

def initialize_model():
    """Initialize the model in a separate thread"""
    global recommender, model_loading, model_ready
    try:
        start_time = time.time()
        recommender = EnhancedMovieRecommender(use_movielens=True)
        load_time = time.time() - start_time
        print(f"Model initialized in {load_time:.2f} seconds")
        model_ready = True
    except Exception as e:
        print(f"Error initializing model: {e}")
    finally:
        model_loading = False

@app.route('/')
def index():
    global recommender, model_loading, model_ready
    
    # Check if model is ready
    if not model_ready and not model_loading:
        # Start loading in background if not already loading
        model_loading = True
        model_thread = threading.Thread(target=initialize_model)
        model_thread.daemon = True
        model_thread.start()
        return render_template('loading.html')
    elif model_loading:
        # Still loading
        return render_template('loading.html')
    
    # Get list of all movies for dropdown
    movies = recommender.get_all_movies()
    return render_template('index.html', movies=movies)

@app.route('/check_loading')
def check_loading():
    """AJAX endpoint to check if model is done loading"""
    global model_ready
    return jsonify({"ready": model_ready})

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie')
    num_recommendations = int(request.form.get('num_recommendations', 5))
    
    # Get recommendations
    start_time = time.time()
    recommendations = recommender.get_recommendations(movie_title, num_recommendations)
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
    num_recommendations = int(request.args.get('count', 5))
    
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    recommendations = recommender.get_recommendations(movie_title, num_recommendations)
    return jsonify({
        "selected_movie": movie_title,
        "recommendations": recommendations
    })

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by partial title"""
    query = request.args.get('q', '').lower()
    if not query or len(query) < 2:
        return jsonify([])
    
    # Get all movies that contain the query string
    matching_movies = [movie for movie in recommender.get_all_movies() 
                      if query in movie.lower()]
    
    # Limit to top 10 matches
    return jsonify(matching_movies[:10])

@app.route('/retrain', methods=['GET'])
def retrain_model():
    """Retrain the model"""
    if recommender:
        start_time = time.time()
        recommender.train_model()
        process_time = time.time() - start_time
        return jsonify({
            "status": "success",
            "message": f"Model retrained in {process_time:.2f} seconds"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Model not initialized"
        }), 500

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize the recommender in a background thread
    model_loading = True
    model_thread = threading.Thread(target=initialize_model)
    model_thread.daemon = True
    model_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)