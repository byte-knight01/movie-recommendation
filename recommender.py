import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommender:
    def __init__(self):
        # Load movie data
        data_file = os.path.join(os.path.dirname(__file__), 'data', 'movies.csv')
        
        if os.path.exists(data_file):
            self.movies_df = pd.read_csv(data_file)
        else:
            # Fallback to sample data if CSV doesn't exist
            self.movies_df = pd.DataFrame({
                'title': [
                    'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
                    'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'Inception',
                    'The Matrix', 'Goodfellas', 'The Silence of the Lambs',
                    'Interstellar', 'The Lion King', 'Titanic', 'Jurassic Park',
                    'Avatar', 'The Avengers', 'Star Wars', 'Toy Story',
                    'The Social Network', 'Parasite'
                ],
                'genres': [
                    'Drama', 'Crime,Drama', 'Action,Crime,Drama', 
                    'Crime,Drama', 'Drama,Thriller', 'Drama,Romance', 'Action,Adventure,Sci-Fi',
                    'Action,Sci-Fi', 'Biography,Crime,Drama', 'Crime,Drama,Thriller',
                    'Adventure,Drama,Sci-Fi', 'Animation,Adventure,Drama', 'Drama,Romance', 'Adventure,Sci-Fi,Thriller',
                    'Action,Adventure,Fantasy', 'Action,Adventure,Sci-Fi', 'Action,Adventure,Fantasy', 'Animation,Adventure,Comedy',
                    'Biography,Drama', 'Comedy,Drama,Thriller'
                ]
            })
            
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            # Save sample data to CSV
            self.movies_df.to_csv(data_file, index=False)
        
        # Create a CountVectorizer
        self.count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
        
        # Fit and transform the genres
        self.genre_matrix = self.count_vectorizer.fit_transform(self.movies_df['genres'])
        
        # Calculate the cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.genre_matrix, self.genre_matrix)
        
        # Create a Series of movie titles for easy lookup
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'])
    
    def get_all_movies(self):
        """Return all movie titles for the dropdown"""
        return sorted(self.movies_df['title'].tolist())
    
    def get_recommendations(self, title, num_recommendations=5):
        """Get movie recommendations based on genre similarity"""
        # Check if the movie is in our database
        if title not in self.indices:
            return []
        
        # Get the index of the movie
        idx = self.indices[title]
        
        # Get the similarity scores for all movies compared to this one
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies (excluding the input movie itself)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top movies
        recommended_movies = self.movies_df.iloc[movie_indices]
        
        # Format the results as a list of dictionaries
        recommendations = []
        for _, row in recommended_movies.iterrows():
            recommendations.append({
                'title': row['title'],
                'genres': row['genres'].replace(',', ', ')
            })
            
        return recommendations