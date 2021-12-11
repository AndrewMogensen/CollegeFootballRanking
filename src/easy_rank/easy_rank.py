import pandas as pd

ratings = pd.read_csv('teams.csv', sep='\t')

max_margin = 38.2
min_margin = -31.4
dif_margin = max_margin - min_margin

ratings["adjW"] = ratings["W"] + ((130.5 - ratings["SOS"].astype(float)) / 32.5 - 2.0) + ((ratings["margin"] - min_margin) / dif_margin)

ratings["rating"] = ratings["adjW"] / (ratings["W"] + ratings["L"])

sorted = ratings.sort_values(by=['rating'], ascending=False)
print(sorted[["team", "rating"]].to_string(index=False))