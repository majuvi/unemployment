# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_timelines(fr, colors, start='2013-01-01', end='2017-12-01'):
    vvvvkk_all = pd.to_datetime(pd.period_range(start, end, freq='M').to_timestamp()).strftime('%Y-%m-%d')
    vvvvkk_observed = np.sort(fr['vvvvkk'].unique())
    print("Specs:", len(fr['henkilotunnus'].unique()), "x", len(vvvvkk_observed), "/", len(vvvvkk_all))

    # Create henkilotunnus x vvvvkk matrix
    tb = fr.pivot(index='henkilotunnus', columns='vvvvkk', values='voimolevatyollkoodi')
    tb = tb.reindex(columns=vvvvkk_all)
    tb = tb.fillna('10')

    # Sort by observation entry and exit
    id_order = tb.apply(lambda s: pd.Series({'first_observable': np.argmin((s == '10').values),
                                             'last_observable': len(s) - np.argmin((s == '10').values[::-1]),
                                             'random_order': np.random.randn()}), axis=1).reset_index()

    id_order.sort_values(by=['first_observable', 'last_observable', 'random_order'], inplace=True)
    ids = id_order['henkilotunnus'].values
    tb = tb.reindex(index=ids, columns=vvvvkk_all)

    # Create color matrix
    ar = tb.values
    n1, n2 = ar.shape
    arr = np.zeros([n1, n2, 3], dtype=int)
    for i in range(n1):
        for j in range(n2):
            arr[i, j, :] = colors[ar[i, j]]
    return (arr)
