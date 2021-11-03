"""Main module."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from scipy.signal import find_peaks


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


def find_break_points(df, columns, version='v3', verbose=False, figure=False):
    """Find break points

    Parameters
    ----------
    df : pandas.DataFrame
    columns : list of column name
    version : select feature point function version
              v1, v2, v3
    verbose : print
    figure : draw figure

    Returns
    -------
    (int, int)
        start_break_point, end_break_point
    """
    points = pd.DataFrame()

    for column in columns:
        x = df[column]
        if version == 'v3':  # 원래 시계열 사용
            start, end, length = get_feature_point_v3(x)
        elif version == 'v2':  # moving average 사용
            start, end, length = get_feature_point_v2(x, bins=10, window=300)

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
        fig = px.line(df, markers=True, width=1000, height=1000)
        fig.add_vline(x=break_point[0], line_width=1, line_dash="solid", line_color="red")
        fig.add_vline(x=break_point[1], line_width=1, line_dash="solid", line_color="red")
        fig.show()

    return break_point


def find_local_peaks(df, column_name, order=50, verbose=False, number=500, idx_start=None, idx_end=None, cut_ep=False):
    """Find local peaks

    Parameter
    ---------
    df : pandas.DataFrame
    column_name : str
        column name
    order : int, optional
        How many points on each side to use for the comparison to consider comparator(n, n+x) to be True.
    """
    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df[column_name].values, np.less_equal, order=order)[0]][column_name]
    df['max'] = df.iloc[argrelextrema(df[column_name].values, np.greater_equal, order=order)[0]][column_name]

    # 평균 근처의 값 제거
    peak_high = df[[column_name]][df['max'] > df[column_name].mean() + df[column_name].std()]
    peak_low = df[[column_name]][df['min'] < df[column_name].mean() - df[column_name].std()]

    # 인접한 여러 인덱스가 Peak로 나타날 수 있어서 인덱스가 직전과 동일하면 직전 인덱스 삭제
    # 인접한 인덱스 중 마지막 인덱스만 남김
    for idx in peak_low.index:
        if idx-1 in peak_low.index:
            peak_low = peak_low.drop(idx-1)
    for idx in peak_high.index:
        if idx-1 in peak_high.index:
            peak_high = peak_high.drop(idx-1)

    # peak_low와 peak high를 연결하여 끝단부분 잘라내기
    if cut_ep:
        # peak_high와 peak_low의 간격이 넓은 문제를 해결하여 첫 번째 포인트 찾기
        while peak_high.index[0] - peak_low.index[0] > number:
            peak_low = peak_low.iloc[1:]
        while peak_low.index[0] - peak_high.index[0] > number:
            peak_high = peak_high.iloc[1:]

        # peak_high와 peak_low의 간격이 넓은 문제를 해결하여 마지막 포인트 찾기
        while peak_high.index[-1] - peak_low.index[-1] > number:
            peak_high = peak_high.iloc[:-1]
        while peak_low.index[-1] - peak_high.index[-1] > number:
            peak_low = peak_low.iloc[:-1]

        if idx_start is None:
            idx_start = df.index[0]
        if idx_end is None:
            idx_end = df.index[-1]
        while peak_low.index[0] - idx_start < number:
            peak_low = peak_low.iloc[1:]
        while peak_high.index[0] - idx_start < number:
            peak_high = peak_high.iloc[1:]
        while idx_end - peak_low.index[-1] < number:
            peak_low = peak_low.iloc[:-1]
        while idx_end - peak_high.index[-1] < number:
            peak_high = peak_high.iloc[:-1]

    # peak 개수 print
    if verbose:
        if len(peak_low)==len(peak_high):
            print('number of peaks:', len(peak_low))
        else:
            print('number of peak_low:', len(peak_low))
            print('number of peaks_high:', len(peak_high))
    return peak_low, peak_high


def get_peak_points(df, column_name=None):
    """get max, min points

    Parameter
    ---------
    df : pandas.DataFrame

    """
    if column_name is None:
        target_df = df
    else:
        target_df = df[column_name]
    length_df = len(target_df)

    peak_candidate_1, _ = find_peaks(target_df, distance=length_df)  # max
    peak_candidate_2, _ = find_peaks(-target_df, distance=length_df)  # min

    return target_df.index[peak_candidate_1[0]], target_df.index[peak_candidate_2[0]]


def get_peak_wave_interval(df, column_name=None, peak_x=None):
    """get peak wave interval

    Parameter
    ---------
    df : pandas.DataFrame

    Return
    ------
    (int, int), (int, int) : peak_x, interval_x
    """
    if column_name is None:
        target_df = df
    else:
        target_df = df[column_name]

    if peak_x is None:
        length_df = len(target_df)

        peak_candidate_1, _ = find_peaks(target_df, distance=length_df)
        peak_candidate_2, _ = find_peaks(-target_df, distance=length_df)

        peak_x_1 = target_df.index[peak_candidate_1[0]]
        peak_x_2 = target_df.index[peak_candidate_2[0]]
    else:
        peak_x_1, peak_x_2 = peak_x

    target_value = (target_df[peak_x_1] + target_df[peak_x_2]) / 2

    if target_df[peak_x_1] > target_df[peak_x_2]:
        case = 1  # sine style
    else:
        case = 2  # -sine style

    if case == 1:
        cand_df = target_df.loc[:peak_x_1]
        left_point = cand_df.loc[cand_df <= target_value].index.max()
        cand_df = target_df.loc[peak_x_2:]
        right_point = cand_df.loc[cand_df >= target_value].index.min()
    else:
        cand_df = target_df.loc[:peak_x_1]
        left_point = cand_df.loc[cand_df >= target_value].index.max()
        cand_df = target_df.loc[peak_x_2:]
        right_point = cand_df.loc[cand_df <= target_value].index.min()

    return (peak_x_1, peak_x_2), (left_point, right_point)


def find_longest_word(word_list):
    """find longest element in the list

    Parameters
    ----------
    word_list : list

    Returns
    -------
        longest element in the list
    """
    longest_word = ''
    for word in word_list:
        if len(word) > len(longest_word):
            longest_word = word
    return longest_word


def binning(srs, bins=2):
    """binning y-axis

    Parameters
    ----------
    srs : pandas.Series
    bins : int, optional
        number of bins, by default 2

    Returns
    -------
    pandas.Series
    """
    out, _ = pd.cut(srs, bins=bins, retbins=True)
    result = out.apply(lambda x: x.mid) 
    return result
