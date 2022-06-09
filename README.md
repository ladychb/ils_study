Abstract
=======
Given a dataset of movies or recipes we calculate the similarity in each of these domains and measure the corresponding ILS of all items in a list.
These lists can be created through SVD (latent-item-vectors) or manually picked by users. 
Furthermore, we provide all scripts for the analysis that is conducted in our paper, so they can be executed and tested.

All scripts for the calculation of ILS can be found in the **src/** folder.
These are as follows:
- **movieAnalysis.py**: Precalculations for the similarity of the movie dataset. 
- **recipeAnalysis.py**: Precalculations for the similarity of the recipe dataset.
- **lda.py**: Specifically used for LDA implementations with ingredients and directions for *recipes*.
- **similarityCalculations.py**: Final calculation of the ILS for both domains.

**borda.py** and **mixedModel.py** are used for further analysis and explained below in the corresponding section.

In folder **data/** there are six folders which we will explain in detail now.

Folder *data/movieDiversity* encompasses the following files:
- *calculatedSimilarities.csv*: Final file with all similarities for all the movie pairs.
- *lists_ID.csv*: Contains all necessary IDs of movies that are used in the list.
- *movies.dat*: Main file providing all meta-data for movies.
- *ratings.csv/.dat*: Main file(s) containing all rating events.
- *topMovie_ID.csv*: Contains the IDs of only the top movies.
- *topUsers.csv*: Contains the users with the highest number of ratings.
- *users.dat*: Main file containing all user-data.

Inside the folder *data/recipeDiversity* there are the following files:
- *lda_directions/ingredients.csv*: Final file(s) with all similarities for all the recipe pairs.
- *collection.csv*: Cleaned recipe data for faster access (memory).
- *raw-data_interactions.csv*: Main file containing all interaction events.
- *raw-data_recipe.csv*: Main file containing all recipe meta-data.
- *recommendation.csv*: Recommendation of recipes based on SVD (used to subset raw-data_recipe).
- *selection.csv*: Selection of recipes for the calculation of ILS (subset of raw-data_recipe).
- *top100.csv*: Top-100 recipes in the dataset based on ratings.
- *topUsers.csv*: Contains the users with the highest number of ratings.

*data/mixed* contains the different files for analysing the mixed model. The data is split by study-phase and domain.

*data/borda* also contains the data split domain for the use with **borda.py**.
 
 Finally, the collected data for the different study-phases can be found in the folder **data/study-data**.
 *Note:* This data is also used by the R scripts. 
 
 We provide the R scripts to our analysis in the folder **R code/**.
 These calculations were done in R Studio but should also run in any new R IDE.

Project setup
=======
Fill data folder
------
The folder 'data' needs to be filled with the **extracted_content_ml_latest** folder. This is a folder containing a set of jsons with movies information. Each file should be called **\<movieId\>.json**

Move study-data to directory
------
The content of study-data needs to be moved to the working directory of the R IDE that is used (e.g., R Studio).

Setup completed
------

Generate ILS and pairwise similarities
======
Once the setup is finished, you can run the file **__main__.py**

Based on the domain that you want to investigate you can uncomment/comment the sections as needed.
If all data is used as provided, all preprocessing and cleaning can be left commented (as it is in the initial state).
Only if you want to re-run these steps, the corresponding sections can be uncommented.

Running the corresponding lines of code for each domain provides the following results:
- All pairwise similarities for selected items in the list (Genre for movie domain and combination of ingredients and directions for recipe domain)
- ILS for the whole list 

If you want to calculate the ILS in the movie domain for another list (not inside **calculatedSimilarities.csv**) follow these steps:
- Change IDs in **similarityCalculations.py**
- Run precalculation with new IDs
- Rerun **__main__.py** with preCalculations uncommented

If you want to calculate the ILS in the recipe domain for another list (not inside **lda_ingredients.csv** or **lda_directions.csv**) follow these steps:
- Run precalculations with new IDs (create *collection.csv*)
- Change IDs in **similarityCalculations.py**
- Rerun **__main__.py** with lda uncommented

R calculations
=====
To run the R calculations open the **analysis.R** file in the R IDE of your choice.
All calculations work with the provided files in **study-data/** and sufficient comments and further instructions are provided in the R script.

Calculate Borda-count
=====
Additionally, it is possible to use **borda.py** to calculate the Borda-count as we implemented it.
Simply run **borda.py** and change the corresponding data in the file for any of the cleaned files in data/borda/.

Mixed-model calculations
=====
Furthermore, we added the Python implementation as well as the R implementation.
The R file is called **mixed effect regression.R** and outputs the same results as **mixedModel.py**