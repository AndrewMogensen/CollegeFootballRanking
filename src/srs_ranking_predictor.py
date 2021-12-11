import pandas as pd
import numpy as np
import models
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from operator import itemgetter
mpl.use('TkAgg')


feature_names = ["OSRS","DSRS","SRS","Scoring Off","Scoring Def","Passing Off","Passing Def","Rushing Off","Rushing Def","Total Off","Total Def","Win Percentage"]

aggregate_rankings = {}
iterations = 25

def div_row(row):
    return float(row["W"]) / float(row["W"] + row["L"])


def prep_rankings(ratings):
    ratings["Win Percentage"] = ratings.apply(div_row, axis=1)
    prepped = ratings.replace(np.nan, 30)

    return prepped.drop(["Rk", "School", "Conf", "W", "L"], axis=1)


def train_models():
    ratings = pd.read_csv('srs_rankings.csv')
    rankings = prep_rankings(ratings)

    x = rankings[feature_names]
    y = rankings["AP Rank"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    linReg = models.linReg(x_train, x_test, y_train, y_test)
    decTree = models.decisionTree(x_train, x_test, y_train, y_test)
    lda = models.lda(x_train, x_test, y_train, y_test)

    return linReg, decTree, lda


def predict(lin_reg_model, dec_tree_model, lda_model):
    new_data = pd.read_csv('srs_new_data.csv')
    schools = new_data["School"]
    new_ratings = prep_rankings(new_data).drop(["AP Rank"], axis=1)
    new_X = new_ratings[feature_names]

    lin_predictions = lin_reg_model.predict(new_X)
    dec_predictions = dec_tree_model.predict(new_X)
    lda_predictions = lda_model.predict(new_X)

    return generate_final_prediction(schools, lin_predictions, dec_predictions, lda_predictions)


def generate_final_prediction(schools, prediction1, prediction2, prediction3):
    rankings = []
    for i in range(len(prediction1)):
        average_rank = float(prediction1[i] + prediction2[i] + prediction3[i]) / 3.0
        rankings.append(average_rank)

    final_rankings = zip(schools, rankings)
    final_rankings.sort(key=itemgetter(0))

    return final_rankings


def update_agregate_rankings(rankings):
    first_rank = (len(aggregate_rankings) == 0)

    for prediction in rankings:
        if first_rank:
            aggregate_rankings[prediction[0]] = prediction[1]
        else:
            aggregate_rankings[prediction[0]] += prediction[1]


def print_final_rankings():
    final_rankings = aggregate_rankings.items()
    final_rankings.sort(key=itemgetter(1))
    for n, prediction in enumerate(final_rankings):
        rank_line = "{}: {} ({})".format((n+1), prediction[0], (prediction[1] / iterations))
        print(rank_line)

for i in range(iterations):
    linReg, decTree, lda = train_models()
    rankings = predict(linReg, decTree, lda)
    update_agregate_rankings(rankings)

print_final_rankings()
