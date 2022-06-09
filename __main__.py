import src.lda as lda
import src.movieAnalysis as mov
import src.similarityCalculations as brute
import src.recipeAnalysis as rec
import src.mixedModel as mix

'''
Calculations for similarity for the movie and recipe domain.
Uncomment/Comment the sections as needed. If all information is used as provided, no cleaning and precalculation is needed.
'''
if __name__ == "__main__":
    # Movie similarity calculation
    # -----------------------------------------------------------------------------------
    # Removes clutter and unnecessary information
    # mov.cleanInteraction()
    # Format dataset for calculations
    # mov.preCalculation()
    # Calculates similarity of all movie pairs
    # mov.svdCalc()
    # Final calculation of similarity for only the selected list
    brute.calculateSimilarityMovie()
    

    # Recipe similarity calculation
    # -----------------------------------------------------------------------------------
    # Removes clutter
    # rec.initialize()
    # Removes unnecessary information
    # rec.cleanList()
    # SVD calculation to find users close to random selection
    # rec.svd_user()
    # SVD calculation to find recipes based on similar users
    # rec.svd_recipe()
    # LDA calculations for recipes
    # lda.lda_calculation()
    # Final calculation of similarity for only the selected list
    brute.calculateSimilarityRecipe()
    # (Optional) find recipes in whole dataset
    # rec.recommendationList()

    '''
    Additional calculations for Borda count and the mixed models
    '''
    # Borda count
    # -----------------------------------------------------------------------------------
    # Run borda.py with corresponding dataset

    # Mixed models
    # -----------------------------------------------------------------------------------
    # Uncomment line and change dataset in mixedModel.py
    # mix.preprocess()