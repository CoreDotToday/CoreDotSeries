"""Main module."""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def get_feature_point(x, window=2000, center=True, bins=100, ctop=3):
    """
    Get the break feature point

    Parameters
    ----------
    x : Pandas Series
        time series data.

    window : rolling window size
    center : rolling position
    bins : histogram bins count
    ctop : top count

    Returns
    -------
    int, int : start_break_point, end_break_point
    """
    target_x = x.rolling(window=window, center=center).mean()
    vcount, vrange = np.histogram(target_x.dropna(), bins=bins)

    # select candidates
    cands = [np.where(vcount == i)[0][0] for i in sorted(vcount, reverse=True)[:ctop]]

    # select max, min
    temp = [vrange[c] for c in cands]
    v_max = np.max(temp)
    v_min = np.min(temp)

    # start_break_point
    start_break_point = 0
    for i in target_x.dropna().items():
        v = i[1]
        if v_min < v < v_max:
            start_break_point = i[0]
            break

    # end_break_point
    end_break_point = 0
    for i in list(target_x.dropna().items())[::-1]:
        v = i[1]
        if v_min < v < v_max:
            end_break_point = i[0]
            break

    return start_break_point, end_break_point


