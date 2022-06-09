import ast
import itertools
from random import randrange
import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.spatial.distance import cosine, hamming, euclidean
from sklearn.preprocessing import normalize

'''
Cleans dataset of recipes with only a few ratings
    Input:
        interactions (csv): Recipe dataset
    Param: 
        minRatings (int): Minimum number of ratings for a recipe to be considered
    Returns:
        topUsers (csv): Dataset after removal of recipes with only few ratings 
'''

def cleanInteraction():
    # Reduce the interactions to only the top 300 entries (at least 285 ratings)
    minRatings = 285
    interactions = pd.read_csv("data/recipeDiversity/raw-data_interaction.csv")
    # Add count of recipes
    interactions['freq'] = interactions.groupby('user_id')['user_id'].transform('count')
    interactions.sort_values(by=['freq'])
    # Remove recipes below the minimal rating count
    selectedUsers = interactions[interactions['freq'] >= minRatings]
    selectedUsers.to_csv("data/recipe/topUsers.csv", index=False)

'''
Cleans list of unneccesary columns
    Input:
        topList (csv): Input dataset file
    Returns:
        topList (csv): New dataset with clutter removed 
'''
def cleanList():
    # Removal of reviews and information that clutters files
    topList = pd.read_csv('data/recipeDiversity/collection.csv')
    topList = topList.drop(columns=['aver_rate', 'review_nums', 'reviews'])
    topList.to_csv("data/recipeDiversity/collection.csv", index=False)

'''
(Optional) For searching the raw-data file for specific recipes due to low memory
    Param: 
        rows (int): How many rows can be searched in one iteration
        list (int): Recipes that we want to find
    Returns:
        recommendationList (csv): Dataset with only the specific recipe data 
'''

def recommendationList():
    recommendationList = pd.DataFrame()
    rows = 3000
    for count in range(50):
        print(count, " of 50")
        skip = (rows*count-1)
        allRecipes = pd.read_csv("data/recipeDiversity/raw-data_recipe.csv", nrows=rows, skiprows=range(1, skip), low_memory=True)
        list = [236172, 7023, 218201, 22375, 17113, 228553]
        # Search for recipe in each subset of entries
        for recipe in list:
            searchedRecipe = allRecipes[allRecipes['recipe_id'] == recipe]
            if searchedRecipe.empty == False:
                recommendationList = recommendationList.append(searchedRecipe)
    recommendationList.to_csv("data/recipeDiversity/collection.csv", index=False)

'''
Creates the top-N-list of items
    Param: 
        rows (int): How many rows can be searched in one iteration
        top (int): Defines how many items should be in the top-list
    Returns:
        recipesTop (csv): Dataset with only the top-N recipes 
'''

def initialize():
    # Reduce the recipes to only recipes that have at least a minimum number of ratings
    recipes = pd.read_csv("data/recipeDiversity/raw-data_recipe.csv", nrows=6000)
    minReviews = 17
    top = 100
    recipes = recipes[recipes['review_nums'] > minReviews]
    recipes.to_csv("data/recipeDiversity/selection.csv", index=False)
    # Sort by aver_rate for top n recipes
    recipes.sort_values(by=['aver_rate'])
    recipesTop = recipes.iloc[:top, :]
    recipesTop.to_csv("data/recipeDiversity/top.csv", index=False)

'''
(Optional) Provides basic meta-data information
    Returns:
        Prints the average rate of the top recipes 
'''

def generalInformation():
    recipes100 = pd.read_csv("data/recipeDiversity/top100.csv")
    print(recipes100.aver_rate)


'''
Calculates the recommendation of recipes based on a set of users chosen from the dataset. 
This is done using SVD for the user-profiles. We repeat this step 10 times for our study.
    Returns:
        prints the sorted recommendations based on one random user  
'''

def svd_user():
    # SVD for user-profiles
    ratings = pd.read_csv('data/recipeDiversity/topUsers.csv',
                          names=['user_id', 'recipe_id', 'rating', 'dateLastModified', 'freq'],
                          engine='python', delimiter=',')
    ratingsMatrix = ratings.pivot(index="user_id", columns="recipe_id", values="rating").fillna(0)
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
        # We use a random participant
        similarity = cosine_similarity(vh[:, randrange(200)], vh[:, col])
        similarity_df = similarity_df.append({'col': col, 'similarity': similarity}, ignore_index=True)

    # Sort by similarity (Similar neighbors)
    sorted = similarity_df.sort_values(by='similarity', ascending=False)

    # Save IDs of similar neighbors
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
    recipeRated = {}
    # Find all recipes of neighbors and use average rating to recommend
    for user in nearestNeighbors:
        for indexRating, rowRating in ratingsMatrix.iterrows():
            if indexRating == user:
                for indexRecipe, valueRating in rowRating.items():
                    if valueRating == 5.0:
                        if indexRecipe in recipeRated.keys():
                            temp = recipeRated[indexRecipe]
                            temp = (temp + 1)
                            recipeRated[indexRecipe] = temp
                        else:
                            recipeRated[indexRecipe] = 1

    # Only show recipes with more than N-top-ratings ratings
    col_headings = ['recipe_id', 'recommend']

    recommendedRecipes_df = pd.DataFrame(columns=col_headings)

    for entry in recipeRated:
        recommendedRecipes_df = recommendedRecipes_df.append({'recipe_id': entry, 'recommend': recipeRated[entry]},
                                                             ignore_index=True)
    recommendedRecipes_df = recommendedRecipes_df[recommendedRecipes_df.recommend > 3]
    sortedrecommendedRecipes_df = recommendedRecipes_df.sort_values(by='recommend', ascending=False)
    print(sortedrecommendedRecipes_df.head(7))

'''
Calculate SVD information based on recipe data
    Returns:
        Similarity between recipes based on meta-data 
        Prints list-ILS
'''
def svd_recipe():
    # SVD for content-information
    #ratings = pd.read_csv('data/recipeDiversity/raw-data_interaction.csv',names=['user_id', 'recipe_id', 'rating', 'dateLastModified'],engine='python', delimiter=',')
    recipe = pd.read_csv('data/recipeDiversity/collection.csv')
    recipe = recipe.drop(columns=['image_url', 'cooking_directions', 'reviews'])
    recipe.aver_rate = recipe.aver_rate.astype(float)
    list_of_dict = []

    for row in recipe.nutritions:
        list_of_dict.append(ast.literal_eval(row))

    # Structure meta data
    calories_list = []
    fat_list = []
    carbohydrates_list = []
    protein_list = []
    cholesterol_list = []
    sodium_list = []
    fiber_list = []

    # Search for information provided
    for x in range(len(list_of_dict)):
        calories_list.append(list_of_dict[x]['calories']['percentDailyValue'])
        fat_list.append(list_of_dict[x]['fat']['percentDailyValue'])
        carbohydrates_list.append(list_of_dict[x]['carbohydrates']['percentDailyValue'])
        protein_list.append(list_of_dict[x]['protein']['percentDailyValue'])
        cholesterol_list.append(list_of_dict[x]['cholesterol']['percentDailyValue'])
        sodium_list.append(list_of_dict[x]['sodium']['percentDailyValue'])
        fiber_list.append(list_of_dict[x]['fiber']['percentDailyValue'])
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    data = {'calories': calories_list, 'fat': fat_list, 'carbohydrates': carbohydrates_list,
            'protein': protein_list, 'cholesterol': cholesterol_list, 'sodium': sodium_list,
            'fiber': fiber_list}

    df = pd.DataFrame(data)
    df.index = recipe['recipe_id']
    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].apply(text_cleaning)
    df = df.apply(pd.to_numeric)
    df_normalized = pd.DataFrame(normalize(df, axis=0))
    df_normalized.columns = df.columns
    df_normalized.index = df.index

    # Calculate nearest neighbour
    def knn_recommender(distance_method, recipe_id, N):
        allRecipes = pd.DataFrame(df_normalized.index)
        allRecipes = allRecipes[allRecipes.recipe_id != recipe_id]
        allRecipes["similarity"] = allRecipes["recipe_id"].apply(
            lambda x: distance_method(df_normalized.loc[recipe_id], df_normalized.loc[x]))
        TopNRecommendation = allRecipes.sort_values(["similarity"]).head(N).sort_values(by=['similarity', 'recipe_id'])

        recipe_df = recipe.set_index('recipe_id')
        recipe_id = [recipe_id]
        recipe_list = []
        for recipeid in TopNRecommendation.recipe_id:
            recipe_id.append(recipeid)  # list of recipe id of selected recipe and recommended recipe(s)
            recipe_list.append("{}  {}".format(recipeid, recipe_df.at[recipeid, 'recipe_name']))

        return df_normalized.loc[recipe_id, :]

    # Calculates the top similar items based on a recipe ID
    def svd(recipe_id, sort_order, N):

        # Calculate cosine distance
        allRecipes_cosine = pd.DataFrame(df_normalized.index)
        allRecipes_cosine = allRecipes_cosine[allRecipes_cosine.recipe_id != recipe_id]
        allRecipes_cosine["similarity"] = allRecipes_cosine["recipe_id"].apply(
            lambda x: 1 - cosine(df_normalized.loc[recipe_id], df_normalized.loc[x]))

        # Calculate euclidian distance
        allRecipes_euclidean = pd.DataFrame(df_normalized.index)
        allRecipes_euclidean = allRecipes_euclidean[allRecipes_euclidean.recipe_id != recipe_id]
        allRecipes_euclidean["similarity"] = allRecipes_euclidean["recipe_id"].apply(
            lambda x: 1 - euclidean(df_normalized.loc[recipe_id], df_normalized.loc[x]))

        # Calculate hamming distance
        allRecipes_hamming = pd.DataFrame(df_normalized.index)
        allRecipes_hamming = allRecipes_hamming[allRecipes_hamming.recipe_id != recipe_id]
        allRecipes_hamming["similarity"] = allRecipes_hamming["recipe_id"].apply(
            lambda x: 1 - hamming(df_normalized.loc[recipe_id], df_normalized.loc[x]))

        # If other SVD algorithms are needed change back (for now we focus on cosine)
        Top2Recommendation_cosine = allRecipes_cosine.sort_values(["similarity"]).head(N).sort_values(
            by=['similarity', 'recipe_id'])
        Top2Recommendation_euclidean = allRecipes_euclidean.sort_values(["similarity"]).head(0).sort_values(
            by=['similarity', 'recipe_id'])
        Top2Recommendation_hamming = allRecipes_hamming.sort_values(["similarity"]).head(0).sort_values(
            by=['similarity', 'recipe_id'])
        recipe_df = recipe.set_index('recipe_id')

        # Combine all lists
        hybrid_Top6Recommendation = pd.concat(
            [Top2Recommendation_cosine, Top2Recommendation_euclidean, Top2Recommendation_hamming])
        aver_rate_list = []
        review_nums_list = []

        # Return the recommendations of the lists
        for recipeid in hybrid_Top6Recommendation.recipe_id:
            aver_rate_list.append(recipe_df.at[recipeid, 'aver_rate'])
            review_nums_list.append(recipe_df.at[recipeid, 'review_nums'])
        hybrid_Top6Recommendation['aver_rate'] = aver_rate_list
        hybrid_Top6Recommendation['review_nums'] = review_nums_list
        TopNRecommendation = hybrid_Top6Recommendation.sort_values(by=sort_order, ascending=False)

        recipe_id = [recipe_id]
        recipe_list = []
        for recipeid in TopNRecommendation.recipe_id:
            recipe_id.append(recipeid)  # list of recipe id of selected recipe and recommended recipe(s)
            recipe_list.append("{}  {}".format(recipeid, recipe_df.at[recipeid, 'recipe_name']))

        # Return either all recipes or just the topN recommendations
        return df_normalized.loc[recipe_id, :], allRecipes_cosine
        #return df_normalized.loc[recipe_id, :], TopNRecommendation

    # List of recipes that need to be compared
    recipeList = []


    """
    (Optional) Calculates pecific similarity between items
    """

    #nutrition_ar, topN_ar = svd(56927, ['similarity'], 6)
    #print(topN_ar)

    for i in range(1):
        list = recipeList
        counter = 0
        listILS = 0
        # Calculates the list similarity at creation
        for recipe1, recipe2 in itertools.combinations(list, 2):
            nutrition_ar, topN_ar = svd(recipe1, ['aver_rate'], 6)
            recipeSimilarity = topN_ar.query("recipe_id == {}".format(recipe2))
            counter = counter + 1
            listILS = listILS + float(recipeSimilarity.similarity)
            print(recipe1, " + ", recipe2, " = ", float(recipeSimilarity.similarity))
        listILSCalculated = listILS / counter
        print(list)
        print('ListILS = ', float(listILSCalculated))



def text_cleaning(cols):
    if cols == '< 1':
        return 1
    else:
        return cols


def cosine_similarity(v, u):
    return (v @ u) / (np.linalg.norm(v) * np.linalg.norm(u))
