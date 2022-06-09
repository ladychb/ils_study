import string
import gensim
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

attribute_ingredients = False

'''
Calculates the similarity of the recipes based on the lda calculation
    Input:
        recipes (csv): Recipe metadata saved as csv
    Returns:
        Pairwise similarity based on lda in csv
        Prints coherence score for evaluation
'''

def lda_calculation():
    tokenizer = RegexpTokenizer(r'\w+')
    recipes = pd.read_csv('data/recipe/recommendation.csv')
    corpus = []

    # Choose relevant attribute for LDA calculation (ingredients or directions)
    if attribute_ingredients == True:
        for ingredients in recipes["ingredients"]:
            corpus.append(ingredients)
    else:
        for directions in recipes["cooking_directions"]:
            corpus.append(directions)
    # Apply preprocessing on the corpus
    # Stop loss words
    stop = set(stopwords.words('english'))

    # Punctuation
    exclude = set(string.punctuation)

    # lemmatization
    lemma = WordNetLemmatizer()

    # One function for all the steps:
    def clean(doc):
        # Convert text into lower case + split into words
        raw = doc.lower()
        tokens = tokenizer.tokenize(raw)
        stop_free = " ".join([i for i in tokens if i not in stop])
        # Remove any stop words present
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        # Remove punctuations + normalize the text
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

        return normalized

    # Clean data stored in a new list
    clean_corpus = [clean(doc).split() for doc in corpus]

    # Creating the term dictionary of our corpus that is of all the words (Sepcific to Genism syntax perspective),
    # where every unique term is assigned an index.

    dict_ = corpora.Dictionary(clean_corpus)

    # Converting list of documents (corpus) into Document Term Matrix using the dictionary
    doc_term_matrix = [dict_.doc2bow(i) for i in clean_corpus]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Training LDA model on the document term matrix.
    # Ingredients num_topics = 8; directions num_topics = 8
    ldamodel = Lda(doc_term_matrix, num_topics=8, id2word=dict_, passes=1, random_state=0, eval_every=None)


    if attribute_ingredients == True:
        # Create corresponding structure for output file
        col_headings = ['recipe1_id', 'recipe2_id', 'lda_ingredients']
        df_ldaIngredients = pd.DataFrame(columns=col_headings)
        counter1 = -1
        for vec_lda1 in ldamodel[doc_term_matrix]:
            counter1 = counter1 + 1
            counter2 = -1
            for vec_lda2 in ldamodel[doc_term_matrix]:
                counter2 = counter2 + 1
                # Calculate similarity based on vectors for all pairs
                sim = gensim.matutils.cossim(vec_lda1, vec_lda2)
                recipe1 = recipes.iloc[counter1]
                recipe2 = recipes.iloc[counter2]
                df_ldaIngredients = df_ldaIngredients.append({'recipe1_id': recipe1['recipe_id'], 'recipe2_id': recipe2['recipe_id'], 'lda_ingredients': sim},ignore_index=True)
        df_ldaIngredients.to_csv("data/recipe/lda_ingredients.csv", index=False)
    else:
        col_headings = ['recipe1_id', 'recipe2_id', 'lda_directions']
        df_ldaDirections = pd.DataFrame(columns=col_headings)
        counter1 = -1
        for vec_lda1 in ldamodel[doc_term_matrix]:
            counter1 = counter1 + 1
            counter2 = -1
            for vec_lda2 in ldamodel[doc_term_matrix]:
                counter2 = counter2 + 1
                # Calculate similarity based on vectors for all pairs
                sim = gensim.matutils.cossim(vec_lda1, vec_lda2)
                recipe1 = recipes.iloc[counter1]
                recipe2 = recipes.iloc[counter2]
                df_ldaDirections = df_ldaDirections.append(
                {'recipe1_id': recipe1['recipe_id'], 'recipe2_id': recipe2['recipe_id'], 'lda_directions': sim},
                ignore_index=True)
        df_ldaDirections.to_csv("data/recipe/lda_directions.csv", index=False)

    # Compute Coherence Score (needed for evaluation of lda model)
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=clean_corpus, dictionary=dict_, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
