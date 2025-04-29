# recommender.py - Enhanced with AI/ML concepts from the research paper
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class EnhancedMovieRecommender:
    def __init__(self):
        """Initialize the movie recommender with advanced NLP techniques"""
        # Download NLTK resources if not already present
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Load movie data
        data_file = os.path.join(os.path.dirname(__file__), 'data', 'movies.csv')
        
        if os.path.exists(data_file):
            self.movies_df = pd.read_csv(data_file)
        else:
            # Create sample data with more attributes as mentioned in the paper
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
                ],
                'director': [
                    'Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan',
                    'Quentin Tarantino', 'David Fincher', 'Robert Zemeckis', 'Christopher Nolan',
                    'Lana and Lilly Wachowski', 'Martin Scorsese', 'Jonathan Demme',
                    'Christopher Nolan', 'Roger Allers', 'James Cameron', 'Steven Spielberg',
                    'James Cameron', 'Joss Whedon', 'George Lucas', 'John Lasseter',
                    'David Fincher', 'Bong Joon-ho'
                ],
                'cast': [
                    'Tim Robbins, Morgan Freeman', 'Marlon Brando, Al Pacino', 'Christian Bale, Heath Ledger',
                    'John Travolta, Uma Thurman', 'Brad Pitt, Edward Norton', 'Tom Hanks, Robin Wright', 'Leonardo DiCaprio, Joseph Gordon-Levitt',
                    'Keanu Reeves, Laurence Fishburne', 'Robert De Niro, Ray Liotta', 'Jodie Foster, Anthony Hopkins',
                    'Matthew McConaughey, Anne Hathaway', 'Matthew Broderick, Jeremy Irons', 'Leonardo DiCaprio, Kate Winslet', 'Sam Neill, Laura Dern',
                    'Sam Worthington, Zoe Saldana', 'Robert Downey Jr., Chris Evans', 'Mark Hamill, Harrison Ford', 'Tom Hanks, Tim Allen',
                    'Jesse Eisenberg, Andrew Garfield', 'Song Kang-ho, Lee Sun-kyun'
                ],
                'overview': [
                    'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                    'The aging patriarch of an organized crime dynasty transfers control to his reluctant son.',
                    'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
                    'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
                    'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.',
                    'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75.',
                    'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
                    'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
                    'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito.',
                    'A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.',
                    'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
                    'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.',
                    'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
                    'A pragmatic paleontologist visiting an almost complete theme park is tasked with protecting a couple of kids after a power failure causes the park\'s cloned dinosaurs to run loose.',
                    'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.',
                    'Earth\'s mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity.',
                    'Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire\'s world-destroying battle station.',
                    'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
                    'As Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook, he is sued by the twins who claimed he stole their idea, and by the co-founder who was later squeezed out of the business.',
                    'Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.'
                ],
                'release_year': [
                    1994, 1972, 2008, 1994, 1999, 1994, 2010, 
                    1999, 1990, 1991, 2014, 1994, 1997, 1993, 
                    2009, 2012, 1977, 1995, 2010, 2019
                ]
            })
            
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            # Save sample data to CSV
            self.movies_df.to_csv(data_file, index=False)
        
        # Create combined tags as mentioned in Section IV.C of the paper
        self.prepare_movie_tags()
        
        # Generate similarity matrix using TF-IDF and cosine similarity
        self.compute_similarity()
    
    def preprocess_text(self, text):
        """Normalize text by removing special chars, lowercasing, and stemming"""
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Remove special characters and convert to lowercase
        text = re.sub('[^a-zA-Z]', ' ', text.lower())
        
        # Tokenize, remove stopwords, and stem
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def prepare_movie_tags(self):
        """Create a combined 'tags' field as mentioned in the research paper"""
        # Handle potentially missing columns
        required_columns = ['genres', 'director', 'cast', 'overview']
        for col in required_columns:
            if col not in self.movies_df.columns:
                self.movies_df[col] = ''
        
        # Create tags by combining multiple fields
        self.movies_df['tags'] = self.movies_df['genres'] + ' ' + \
                                 self.movies_df['director'] + ' ' + \
                                 self.movies_df['cast'] + ' ' + \
                                 self.movies_df['overview']
        
        # Apply preprocessing to tags
        self.movies_df['processed_tags'] = self.movies_df['tags'].apply(self.preprocess_text)
    
    def compute_similarity(self):
        """Create vector representation and compute similarity matrix using TF-IDF"""
        # Initialize TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(max_features=5000)
        
        # Fit and transform the processed tags
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df['processed_tags'])
        
        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create a Series of movie titles for easy lookup
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'])
    
    def get_all_movies(self):
        """Return all movie titles for the dropdown"""
        return sorted(self.movies_df['title'].tolist())
    
    def get_recommendations(self, title, num_recommendations=5):
        """Get movie recommendations based on content similarity"""
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
        
        # Return the top movies with additional info
        recommended_movies = self.movies_df.iloc[movie_indices]
        
        # Format the results as a list of dictionaries
        recommendations = []
        for _, row in recommended_movies.iterrows():
            recommendations.append({
                'title': row['title'],
                'genres': row['genres'].replace(',', ', '),
                'director': row['director'],
                'similarity': round(sim_scores[len(recommendations)][1] * 100, 2),  # Convert to percentage
                'year': row['release_year'] if 'release_year' in row else 'N/A'
            })
            
        return recommendations