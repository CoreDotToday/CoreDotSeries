"""Main module."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import math
import warnings
# plotly.offline.init_notebook_mode(connected=False)


warnings.filterwarnings(action='ignore')


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


def find_break_points(df, channel=['D1Y', 'D2Y', 'D3Y', 'D4Y'], version='v3', verbose=False, figure=False):
    points = pd.DataFrame()

    for column in channel:
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


def find_local_peaks(df, tsp_channel='D4Y', order=50, verbose=False, number=500, idx_start=None, idx_end=None):
    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df[tsp_channel].values, np.less_equal, order=order)[0]][tsp_channel]
    df['max'] = df.iloc[argrelextrema(df[tsp_channel].values, np.greater_equal, order=order)[0]][tsp_channel]
    
    # 평균 근처의 값 제거
    peak_high = df[[tsp_channel]][df['max'] > df[tsp_channel].mean() + df[tsp_channel].std()]
    peak_low = df[[tsp_channel]][df['min'] < df[tsp_channel].mean() - df[tsp_channel].std()]
    
    # 인접한 여러 인덱스가 Peak로 나타날 수 있어서 인덱스가 직전과 동일하면 직전 인덱스 삭제
    # 인접한 인덱스 중 마지막 인덱스만 남김
    for idx in peak_low.index:
        if idx-1 in peak_low.index:
            peak_low = peak_low.drop(idx-1)
    for idx in peak_high.index:
        if idx-1 in peak_high.index:
            peak_high = peak_high.drop(idx-1)
    
    # low peak -> high peak 순서대로 나타나야하며 (low peak와 high peak의 좌표가 number 이상 차이나면 벗어나는 좌표 제거)
    ''' 예제 -> low에서 55753, high에서 6580 제거
    peak low Int64Index([ 8557, 10279, 12006, 13730, 15564, 16955, 18965, 20743, 22639,
                24441, 26047, 27975, 29252, 31375, 33253, 35255, 37283, 39544,
                42226, 44326, 46898, 49268, 51186, 53030, 55753],
               dtype='int64')
    peak high Int64Index([ 6580,  8598, 10332, 12038, 13780, 15616, 17034, 19029, 20781,
                22682, 24474, 26097, 28024, 29314, 31469, 33304, 35306, 37339,
                39617, 42310, 44411, 46965, 49337, 51257, 53090],
               dtype='int64')
    '''
    
    # peak_high와 peak_low의 간격이 넓은 문제를 해결하여 첫 번째 TSP 찾기
    while peak_high.index[0] - peak_low.index[0] > number:
        peak_low = peak_low.iloc[1:]
    while peak_low.index[0] - peak_high.index[0] > number:
        peak_high = peak_high.iloc[1:]
    
    # peak_high와 peak_low의 간격이 넓은 문제를 해결하여 마지막 TSP 찾기
    while peak_high.index[-1] - peak_low.index[-1] > number:
        peak_high = peak_high.iloc[:-1]
    while peak_low.index[-1] - peak_high.index[-1] > number:
        peak_low = peak_low.iloc[:-1]
    
    # TEI와 TEO에서 떨어진 것만 TSP 신호로 정의하기
    if idx_start == None:
        idx_start = df.index[0]
    if idx_end == None:
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


def find_tsp_x(peak_low, peak_high, where='center'):
    tsp_list = []
    # 인접한 peak_low와 peak_high의 중간값을 tsp로 정의하기
    for low, high in zip(peak_low.index, peak_high.index):
        if where == 'center':
            tsp_list.append(math.ceil((low + high) / 2))
        if where == 'low':
            tsp_list.append(low)
        if where == 'high':
            tsp_list.append(high)
        
    return tsp_list


def find_defect(tsp_x_list, tsp_num, distance, where='center', number=None, verbose=False):
    tsp_x = tsp_x_list[tsp_num - 1]
    distance_x = 4000 * distance
    result_x = int(tsp_x + distance_x)
    # 결함 위치 찾아가기 print
    
    if where == 'center':
        if verbose:
            print(f"TSP {tsp_num}(x: {int(tsp_x)}) + {distance}m(x: {int(distance_x)}) -> x: {int(tsp_x + distance_x)}")
        return result_x
    elif where == 'interval':
        if verbose:
            print(f"TSP {tsp_num}(x: {int(tsp_x)}) + {distance}m(x: {int(distance_x)}) -> x: {int(tsp_x + distance_x)} ± {number}")
            print(f"(start, end) : ({int(result_x - number)}, {int(result_x + number)})")
        return int(result_x - number), int(result_x + number)

    
def get_defect(df, tsp_num, distance, number=50, tsp_channel='D4Y', graph=True):
    idx_start, idx_end = find_break_points(df, channel=['D1Y', 'D2Y', 'D3Y', 'D4Y'])
    peak_low, peak_high = find_local_peaks(df[idx_start:idx_end], tsp_channel='D4Y', idx_start=idx_start, idx_end=idx_end)
    tsp_x_list = find_tsp_x(peak_low, peak_high, where='low')
    defect_start, defect_end = find_defect(tsp_x_list, tsp_num, distance, where='interval', number=number, verbose=True)
    
    if graph:
        fig = px.line(df, markers=True, width=1700, height=1000)
        fig.add_vline(x=idx_start, line_width=1, line_dash="solid", line_color="red")
        fig.add_vline(x=idx_end, line_width=1, line_dash="solid", line_color="red")
        fig.add_vline(x=defect_start, line_width=1, line_dash="solid", line_color="green")
        fig.add_vline(x=defect_end, line_width=1, line_dash="solid", line_color="green")

        for tsp_x in tsp_x_list:
            fig.add_vline(x=tsp_x, line_width=1, line_dash="solid", line_color="blue")
        fig.show()
    
    return defect_start, defect_end


def get_lissajous(df, start, end, channel ='D2'):

    axis_range = 1000
    size = 1000

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df[channel+'X'][start:end],
            y=df[channel+'Y'][start:end],
            hovertext=df.index[start:end],
            mode='markers+lines',
            marker=dict(size=5, line_width=0.5),
        )
    )
    fig.update_layout(
        width=size, height=size,
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    fig.show()
