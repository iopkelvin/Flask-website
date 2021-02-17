"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.
This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Merged movies DF
merged_movies = pd.read_csv('models/merged_movies_tags.csv')
# TFIDF matrix
movie_df = pd.read_csv('models/movie_tfidf_matrix.csv')

def recommend_by_genre(movie):
    """
    What this function does is:
    It uses the matrix of TFIDF scores and it finds the cosine similarity for each score compared to the movie that is chosen.
    It then returns a list of the top 10 movies with the highest score.
    """
    requested_movie_id = merged_movies[merged_movies['key'] == movie].index
    requested_movie_values = (movie_df.iloc[requested_movie_id]
                                      .values
                                      .reshape((-1,)))
    num_recs = 10
    movie_scores = []

    for movie_id, movie_values in enumerate(movie_df.values):
        score = cosine_similarity([requested_movie_values],[movie_values])[0][0]
        title = merged_movies.loc[movie_id, 'key']
        movie_scores.append((title, score))

    return sorted(movie_scores, key = lambda x:x[1], reverse = True)[1:num_recs]

# Trying out npy instead of pickle
# with open('models/cosine.pkl', 'rb') as f:
#     cosine = pickle.load(f)
# with open('models/cosine.npy', 'rb') as f:
#     cosine = np.load(f)

# Sorted by Similarity and Rating
# titles = movies['key'] # switch to key for no year
# indices = pd.Series(movies.index, index=movies['key']) # switch to key for no year
#
# def recommend_by_genre(title):
#     idx = indices[title]
#     sim_scores = cosine[idx]
#     datas = pd.concat([pd.Series(sim_scores), movies['weighted_mean_rating']], axis=1)
#     datas.columns = ['similarity', 'weighted_mean_rating']
#     datas = datas.sort_values(by=["similarity", 'weighted_mean_rating'], ascending=False)
#     index = datas.iloc[1:11].index
#     result = titles.iloc[index]
#     return list(result.reset_index()['key'])

# This section checks that the prediction code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    from pprint import pprint
    print("These are the recommended movies:")
    results = recommend_by_genre()
    pprint(results)
