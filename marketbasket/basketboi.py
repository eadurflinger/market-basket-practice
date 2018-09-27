import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


class Main:

    store_data = pd.read_csv("Transactions.csv")
    store_data.head()
    print(store_data)

    records = []
    for i in range(0, 4):
        records.append([str(store_data.values[i, j]) for j in range(0, 5)])

    print(records)

    association_rules = apriori(
                                records,
                                min_support=0.0045,
                                min_confidence=1.0,
                                min_length=3,
                                max_length=3)

    association_results = list(association_rules)

    print(association_results)
