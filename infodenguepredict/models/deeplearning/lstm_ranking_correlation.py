from infodengue import build_multicity_dataset
import numpy as np
import pandas as pd
import re


def rank_cities(state):
    mult = build_multicity_dataset(state)
    cols = list(filter(re.compile('casos_\d+').search, mult.columns))
    mult = mult[cols]

    codes = pd.read_excel('../data/codigos_rj.xlsx'.format(lower(state)),
                          names=['city', 'code'], header=None).set_index('code').T

    ints = pd.DataFrame(index=codes.values[0], columns=['integral'])
    for col in mult.columns:
        ints.loc[codes[int(re.sub('casos_', '', col))]] = [np.trapz(mult[col])]

    return ints.sort_values('integral', ascending=False)


if __name__=="__main_":
    ranK_df = rank_cities('RJ')
