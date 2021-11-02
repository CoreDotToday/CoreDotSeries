import plotly
import warnings


def ignore_warning(action='ignore'):
    warnings.filterwarnings(action=action)


def plotly_offline_mode(connected=False):
    plotly.offline.init_notebook_mode(connected=connected)

