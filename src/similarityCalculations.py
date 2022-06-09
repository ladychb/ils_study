import itertools
import pandas as pd
import random

'''
(Optional) can be used to create list out of a selection of ids.
    Input:
        id_data (int): Array of IDs for the creation of lists
    Returns:
        df_combinations (as csv): List created with the IDs
'''

def createLists():
    # Can be used to create all list combinations
    id_data = [47, 260, 296, 364, 541, 589, 595, 858, 924, 1193, 1214, 2959, 4886, 4993, 5218, 5952, 6377, 7153, 8957, 8961, 45722, 54001, 72998, 79132, 88125, 89745, 91529, 95510,96079,109487,116797,122882,122900,134853]

    combinations = itertools.combinations(id_data, 7)
    df_combinations = pd.DataFrame(columns=['list'])
    for entry in list(combinations):
        df_combinations = df_combinations.append({'list': entry}, ignore_index=True)
    df_combinations.to_csv("data/combinations.csv", index=False)

'''
Calculates the similarity of the recipes based on ingredients and directions
    Input:
        similaritiesIngredients (csv): Pairs of recipes with the ingredients similarity value
        similaritiesDirections (csv): Pairs of recipes with the directions similarity value
    Returns:
        Prints every caluclated pair similarity in the list with the corresponding ILS for the list
'''

def calculateSimilarityRecipe():
    similaritiesIngredients = pd.io.parsers.read_csv('data/recipe/lda_ingredients.csv',
                                       names=['recipe1', 'recipe2', 'lda_ingredients'],
                                       engine='python', delimiter=',')
    similaritiesDirections = pd.io.parsers.read_csv('data/recipe/lda_directions.csv',
                                                     names=['recipe1', 'recipe2', 'lda_directions'],
                                                     engine='python', delimiter=',')

    list = ["67952", "58112", "213742", "214651", "17066", "8818", "26317"]
    for i in range(1):
        list = random.sample(list, 7)
        counter = 0
        listILS = 0
        print("Pairwise ILS of ", list, ":")
        # Important for ILS calculation: The similarity of every combination needs to be calculated (order should not matter in the end)
        for item1, item2 in itertools.combinations(list,2):
            # Look for pairs of recipes
            View1 = similaritiesDirections[similaritiesDirections['recipe1'] == item1]
            View2 = similaritiesIngredients[similaritiesIngredients['recipe1'] == item1]
            currentSimilarity1 = View1[View1['recipe2'] == item2]
            currentSimilarity2 = View2[View2['recipe2'] == item2]
            if currentSimilarity1.empty and currentSimilarity2.empty:
                similarity = None
            else:
                similarity = (float(currentSimilarity2["lda_ingredients"]) + float(
                    currentSimilarity1["lda_directions"])) / 2
                counter = counter + 1
                listILS = listILS + float(similarity)
            print("Recipe1: ", item1, " + ", "Recipe2: ", item2, " = ", float(similarity))

        listILSCalculated = listILS / counter
        print("-------------------------")
        print('ListILS = ', float(listILSCalculated))

'''
Calculates the similarity of the movies based on genre
    Input:
        similaritiesGenre (csv): Pairs of movies with the genre similarity value
    Returns:
        Prints every caluclated pair similarity in the list with the corresponding ILS for the list
'''

def calculateSimilarityMovie():
    similaritiesGenre = pd.io.parsers.read_csv('data/movieDiversity/calculatedSimilarities.csv', skiprows=1,
                                               names=['movie1', 'movie2', 'jacc_genre'],
                                               engine='python', delimiter=',')

    list = [1221, 4993, 47, 356, 457, 1258, 2571]
    for i in range(1):
        list = random.sample(list, 7)
        counter = 0
        listILS = 0
        print("Pairwise ILS of ", list, ":")
        # Important for ILS calculation: The similarity of every combination needs to be calculated (order should not matter in the end)
        for item1, item2 in itertools.combinations(list, 2):
            # Look for pairs of movies
            View1 = similaritiesGenre[similaritiesGenre['movie1'] == item1]
            View2 = similaritiesGenre[similaritiesGenre['movie2'] == item1]
            currentSimilarity1 = View1[View1['movie2'] == item2]
            currentSimilarity2 = View2[View2['movie1'] == item2]
            if currentSimilarity1.empty and currentSimilarity2.empty:
                similarity = None
            elif currentSimilarity1.empty:
                similarity = (float(currentSimilarity2["jacc_genre"]))
                counter = counter + 1
                listILS = listILS + float(similarity)
            else:
                similarity = (float(currentSimilarity1["jacc_genre"]))
                counter = counter + 1
                listILS = listILS + float(similarity)
            print("Movie1: ", item1, " + ", "Movie2: ", item2, " = ", float(similarity))

        listILSCalculated = listILS / counter
        print("-------------------------")
        print('ListILS = ', float(listILSCalculated))