import pandas as pd
from surprise import Reader, Dataset

# Load the data into a Pandas DataFrame
df = pd.read_csv('movie_ratings.csv')

# Use the Reader class from Surprise to parse the data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
from surprise import SVD
from surprise.model_selection import cross_validate

# Instantiate the SVD model
model = SVD()

# Perform cross-validation to evaluate the model
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# Train the model on the whole dataset
trainset = data.build_full_trainset()
model.fit(trainset)

# Function to get top N movie recommendations for a given user
def get_top_n_recommendations(user_id, n=10):
    # Get a list of all movie IDs
    all_movie_ids = df['movie_id'].unique()
    
    # Get a list of movies the user has already rated
    rated_movies = df[df['user_id'] == user_id]['movie_id']
    
    # Predict ratings for all movies the user hasn't rated yet
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]
    
    # Sort the predictions by estimated rating in descending order
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # Return the top N movie IDs
    return [pred.iid for pred in top_predictions]

# Get top 10 movie recommendations for user with ID 1
user_id = 1
top_n_recommendations = get_top_n_recommendations(user_id, n=10)
print("Top 10 movie recommendations for user {}: {}".format(user_id, top_n_recommendations))
