import itertools
import string
from typing import List, Optional
import numpy as np
import pandas as pd
from numpy.linalg import svd
from random import randrange
from nltk.corpus import stopwords
from pandas import DataFrame, Series
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import svd

np.seterr(divide='ignore', invalid='ignore')
stopwords = stopwords.words('english')

'''
Cleans list of unneccesary columns
    Input:
        ratings (csv): Input dataset file
    Param:
        topRating (int): Cut-off point for ratings considered for dataset
    Returns:
        selectedUsers (csv): New dataset with only top-users considered 
''' 
def cleanInteraction():
    # Reduce the interactions to only the top N users (at least 500 ratings)
    ratings = pd.io.parsers.read_csv('data/movieDiversity/ratings.csv', skiprows=1, nrows=1000000,
                                     names=['user_id', 'movie_id', 'rating', 'time'],
                                     engine='python', delimiter=',')
    topRating = 500
    ratings['freq'] = ratings.groupby('user_id')['user_id'].transform('count')
    ratings.sort_values(by=['freq'])
    selectedUsers = ratings[ratings['freq'] >= topRating]
    selected_ID = selectedUsers['movie_id']
    selected_ID = selected_ID.drop_duplicates()
    selected_ID.to_csv("data/movieDiversity/topMovie_ID.csv", index=False)
    selectedUsers.to_csv("data/movieDiversity/topUsers.csv", index=False)

'''
Calculates the similarity of the movies based on SVD calculations
    Input:
        ratings (Dataframe): Input data from only the top users
    Returns:
        Prints recommended movies based on random user
'''

def svdCalc():
    # Select a seed if necessary
    #my_seed = 0
    #random.seed(my_seed)
    #np.random.seed(my_seed)

    # Load rating rata
    ratings = pd.io.parsers.read_csv('data/movieDiversity/topUsers.csv',
                                       names=["user_id","movie_id","rating","time","freq"], skiprows=1,
                                       engine='python', delimiter=',')
    ratingsMatrix = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)

    matrix = ratingsMatrix.values
    # Normalization of matrix
    normalised_mat = matrix - np.asarray([(np.mean(matrix, 1))]).T
    A = normalised_mat.T / np.sqrt(matrix.shape[0] - 1)
    u, s, vh = np.linalg.svd(A, full_matrices=False)

    # Creation of similarities
    col_headings = ['col', 'similarity']
    similarity_df = pd.DataFrame(columns=col_headings)

    # For every column we calculate the similarity values
    for col in range(0, vh.shape[1]):
        # We use a random participant out of the top
        # Use specific number of latent vectors here (optimal k = 14)
        k = 14
        similarity = cosine_similarity(vh[:k, randrange(300)], vh[:, col])
        similarity_df = similarity_df.append({'col': col, 'similarity': similarity}, ignore_index=True)

    # Sort by similarity (Similar neighbors)
    sorted = similarity_df.sort_values(by='similarity', ascending=False)

    # Save IDs of ten most similar neighbors
    nearestNeighbors = []
    counter = 0
    innerCounter = 0
    for indexSorted, rowSorted in sorted.iterrows():
        if counter < 11:
            innerCounter = 0
            for indexRating, rowRating in ratingsMatrix.iterrows():
                if indexSorted == innerCounter:
                    nearestNeighbors.append(indexRating)
                innerCounter = innerCounter + 1
        counter = counter + 1
    movieRated = {}
    # Find all recipes of neighbors and use average rating to recommend
    for user in nearestNeighbors:
        for indexRating, rowRating in ratingsMatrix.iterrows():
            if indexRating == user:
                for indexRecipe, valueRating in rowRating.items():
                    if valueRating == 5.0:
                        if indexRecipe in movieRated.keys():
                            tempMovieRated = movieRated[indexRecipe]
                            tempMovieRated = (tempMovieRated + 1)
                            movieRated[indexRecipe] = tempMovieRated
                        else:
                            movieRated[indexRecipe] = 1

    # Only show recipes that are considered highly rated
    col_headings = ['movie_id', 'recommend']

    recommendedMovies_df = pd.DataFrame(columns=col_headings)
    for entry in movieRated:
        recommendedMovies_df = recommendedMovies_df.append({'movie_id': entry, 'recommend': movieRated[entry]},
                                                             ignore_index=True)
    recommendedMovies_df = recommendedMovies_df[recommendedMovies_df.recommend > 3]
    recommendedMovies_df = recommendedMovies_df.sort_values(by='recommend', ascending=False)
    print("Recommended movies:")
    print(recommendedMovies_df.head(7))
    print("------------------------")


'''
Calculates the similarity of the recipes based on ingredients and directions
    Param:
        allMovieIds (csv): Dataset of all movie IDs
        allMovies (csv): Dataset of all movies
    Returns:
        Creates csv with movie pair and calculated genre similarity
'''
def preCalculation():
    # Get all movies in Top N
    allMovieIds = read_movie_ids_from_csv("data/movieDiversity/lists_ID.csv")
    allMovies = get_movies_by_id(allMovieIds, "data/extracted_content_ml-latest/")
    column_names = ["movie1_ID", "movie2_ID", "Genre:JACC"]
    df_calculatedData = pd.DataFrame(columns=column_names)
    df_calculatedData.astype({'movie1_ID': 'int32'}).dtypes
    df_calculatedData.astype({'movie2_ID': 'int32'}).dtypes
    # Compare all movies in said directory and calculate the following similarity measures
    for movie1, movie2 in itertools.combinations(allMovies, 2):
        try:
            movie1_ID = int(movie1['movielens']['movieId'])
            movie1_genres = movie1['movielens']['genres']
        except:
            movie1_ID = 0
            movie1_genres = ""
        try:
            movie2_ID = int(movie2['movielens']['movieId'])
            movie2_genres = movie2['movielens']['genres']
        except:
            movie2_ID = 0
            movie2_genres = ""

        # Genre: JACC calculation
        try:
            intersection = len(list(set(movie1_genres).intersection(movie2_genres)))
            union = (len(set(movie1_genres)) + len(set(movie2_genres))) - intersection
            genre_similarity = float(intersection) / union
        except:
            genre_similarity = -1
        # Append dataframes
        if genre_similarity != -1:
            df_calculatedData = df_calculatedData.append(
                {"movie1_ID": int(movie1_ID), "movie2_ID": int(movie2_ID), "Genre:JACC": genre_similarity},
                ignore_index=True)
    # Save dataframe in csv
    df_calculatedData.to_csv("data/movieDiversity/calculatedSimilarities.csv", index=False)



def cleanString(text):
    # Removal of unwanted punctuation, stopwords, and making text lowercase
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stopwords)
    return text


def extract_keywords(tags_list):
    extractedKeywords = []
    for tag in tags_list:
        extractedKeywords.append(tag['name'])
    return extractedKeywords

def read_movie_ids_from_csv(path: str) -> List[int]:
    """
    Returns ids of all movies read from path
    param path: path were movie csv is
    return: corresponding series of movie ids
    """

    df: DataFrame = pd.read_csv(path, index_col=False, header=0)
    series: Series = df.iloc[:, 0]
    return series.tolist()

def get_movies_by_id(list_of_movies: List[int], path_to_movies: str) -> List[DataFrame]:
    """
    Return dataframe of movies whose ids were passed by list_of_movies
    param list_of_movies: List of movies to get
    param path_to_movies: path where json are
    """
    movies: List[DataFrame] = []
    for movie_id in list_of_movies:
        movies.append(get_movie_from_json_folder(movie_id, path_to_movies))
    return movies

def get_movie_from_json_folder(movie_id: int, path: str) -> DataFrame:
    """
    Return dataframe of movie reading the path
    param movie_id: id of movie
    param path: path where the JSON folder is
    """
    path: str = path + str(movie_id) + ".json"
    df_movie: Optional[DataFrame] = None
    try:
        df_movie = pd.read_json(path)
    except ValueError:
        print(f"File not found: \n\t{path}")
    return df_movie

def cosine_similarity(v, u):
    return (v @ u) / (np.linalg.norm(v) * np.linalg.norm(u))


def tuningHP(svd):
    """
    (Optional) Check for optimal amount of latent factors
    """
    param_grid = {'n_factors': [10], 'n_epochs': [20], 'lr_all': [0.005], 'reg_all': [0.02]}
    gs = GridSearchCV(svd, param_grid, measures=['rmse'], cv=3)
