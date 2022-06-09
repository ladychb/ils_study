import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

'''
Calculate the linear regression and mixed model with the provided dataset in preprocess()
    Parameters:
            df (dataframe):     Dataset used for the models
            predictor (String): Predictor that is used in the regression   
            dependent (String): Dependent that is used for the regression  
    Returns: 
            Printed results of the linear regression and mixed model
'''

def mixedModel(df):
    df = pd.read_csv("data/mixed/Study1a.tsv", sep="\t")
    # ILS used as a predictor
    predictor = "ILS"
    # diversity, variety, similarity, easiness, confidence
    dependent = "diversity"
    print("--------------------------------------------")
    print("Linear regression: ")
    reg = LinearRegression()  # create object for the class
    X = df[predictor].values.reshape(-1, 1)
    Y = df[dependent].values.reshape(-1, 1)
    # perform linear regression
    reg.fit(X, Y)
    print("Coefficient: " + str(reg.coef_) + ", Intercept:" + str(reg.intercept_))
    # try out just on train data
    y_predicted = reg.predict(X)
    print("R Squared: " + str(r2_score(Y, y_predicted)))
    print("--------------------------------------------")
    print("Mixed regression: ")

    model = smf.mixedlm(dependent + " ~ " + predictor, df, groups="id").fit()
    print(model.summary())
    y_mixed = model.predict()
    print("R2 (mixed): " + str(r2_score(Y, y_mixed)))

    het_white_res = het_white(model.resid, model.model.exog)
    labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    for key, val in dict(zip(labels, het_white_res)).items():
        print(key, val)

'''
Removal of participants that always choose the same option or provide incorrect assumptions. Usable on all tab-seperated datasets. 
    Input:
        df_removal (dataframe): Dataset used for preprocessing and cleaning
        criteria[1-3] (String): Criteria used for the removal (can be changed)
    Returns:
        Directly after preprocessing starts the mixed-model with the cleaned dataset.
'''

def preprocess():
    # Insert dataset which needs preprocessing
    df_removal = pd.read_csv("data/mixed/Study1a_removal.tsv", sep="\t")
    for index, row in df_removal.iterrows():
        identical = False
        unlogical = False

        criteria1 = ["diversity1", "variety1", "similarity1"]
        criteria2 = ["diversity2", "variety2", "similarity2"]
        criteria3 = ["diversity3", "variety3", "similarity3"]

        for count in range(3):
            # Remove participants that always choose the same value
            if row[criteria1[count]] == row[criteria2[count]] == row[criteria3[count]]:
                identical = True
                print(row[criteria1[count]], " ", row[criteria2[count]], " ", row[criteria3[count]])

        for count in range(1,4):
            # Remove participants that consider the sequel/collection to be very diverse
            treatment = "treatment{}".format(count)
            similarity = "similarity{}".format(count)
            if row[treatment] == "Collection" or row[treatment] == "Sequels":
                if row[similarity] < 3:
                    unlogical = True

        # Removal of rows that are considered for our analysis
        if not identical and not unlogical:
            print("Row is fine")
            df_removal.drop(index, inplace=True)
        else:
            print("Row should be removed later on")

    df = pd.read_csv("data/mixed/Study1a.tsv", sep="\t")
    # Removal of rows we do not consider in our analysis for results
    for index1, row1 in df_removal.iterrows():
        for index2, row2 in df.iterrows():
            if row1['id'] == row2['id']:
                df.drop(index2, inplace=True)
    mixedModel(df)
