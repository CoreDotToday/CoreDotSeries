"""Main module."""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import operator
import plotly
import plotly.express as px
# plotly.offline.init_notebook_mode(connected=False)


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


def get_feature_point_v2(x, window=1000, center=True, bins=3):
    """Get the break feature point

    Parameters
    ----------
    x : pandas.Series
        time series data
    window : int, optional
        size of the moving window on pandas.Series.rolling(), by default 1000
    center : bool, optional
        set the labels at the center of the window on on pandas.Series.rolling(), by default True
    bins : int, optional
        the number of equal-width bins in the range of x on pandas.cut(), by default 10

    Returns
    -------
    int, int
        start_break_point, end_break_point
    """
    x_ma = x.rolling(window=window, center=center).mean().dropna()
    out, _ = pd.cut(x_ma, bins=bins, retbins=True)
    x_binning = out.apply(lambda x: x.mid).astype(np.float)

    # x_start(included), x_end(included), x_length, y_value
    period = []
    x_start = x_binning.index[0]
    value = x_binning.to_numpy()[0]
    for x, y in x_binning.items():
        if y != value:
            x_end = x - 1
            period.append((x_start, x_end, x_end - x_start + 1, value))
            x_start = x
            value = y
        if x == x_binning.index[-1]:
            period.append((x_start, x, x - x_start + 1, value))
    start_break_point, end_break_point, length, _ = sorted(period, key=operator.itemgetter(2), reverse=True)[0]
    return start_break_point, end_break_point, length


def get_feature_point_v3(x, bins=3):
    """Get the break feature point

    Parameters
    ----------
    x : pandas.Series
        time series data
    bins : int, optional
        the number of equal-width bins in the range of x on pandas.cut(), by default 10

    Returns
    -------
    int, int, int
        start_break_point, end_break_point, length
    """
    out, _ = pd.cut(x, bins=bins, retbins=True)
    x_binning = out.apply(lambda x: x.mid)  # .astype(np.float)

    # x_start(included), x_end(included), x_length, y_value
    period = []
    x_start = x_binning.index[0]
    value = x_binning.to_numpy()[0]
    for x, y in x_binning.items():
        if y != value:
            x_end = x - 1
            period.append((x_start, x_end, x_end - x_start + 1, value))
            x_start = x
            value = y
        if x == x_binning.index[-1]:
            period.append((x_start, x, x - x_start + 1, value))
    start_break_point, end_break_point, length, _ = sorted(period, key=operator.itemgetter(2), reverse=True)[0]
    return start_break_point, end_break_point, length


def find_break_points(path, version='v3', verbose=False, figure=False):
    df = pd.read_pickle(path)
    points = pd.DataFrame()

    for column in df.columns:
        x = df[column]
        if version == 'v3':  # 원래 시계열 사용
            start, end, length = get_feature_point_v3(x)
        elif version == 'v2':  # moving average 사용
            start, end, length = get_feature_point_v2(x, bins=10)

        point_dict = {
            'start': start,
            'end': end,
            'length': length,
        }
        points = points.append(point_dict, ignore_index=True)

    # 최대 길이를 갖는 start와 end 좌표 가져오기
    max_idx = points.index[points['length'] == points.length.max()][0]
    break_point = (int(points['start'].iloc[max_idx]), int(points['end'].iloc[max_idx]))

    if verbose:
        print(f"{df.columns[max_idx]}")
        print(f"valid length: {break_point[1] - break_point[0]}")
        print(f"break point: {break_point}")

    if figure:  # plotly 포함할 것
        fig = px.line(df, markers=True, width=1500, height=1000)
        fig.add_vline(x=break_point[0], line_width=1, line_dash="solid", line_color="red")
        fig.add_vline(x=break_point[1], line_width=1, line_dash="solid", line_color="red")
        fig.show()

    #     return break_point
    return df
