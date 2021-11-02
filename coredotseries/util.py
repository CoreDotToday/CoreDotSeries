import plotly
import warnings


def ignore_warning(action='ignore'):
    """ignore warning message

    Parameter
    ---------
    action : str
        default ignore
    """
    warnings.filterwarnings(action=action)


def plotly_offline_mode(offline_mode=True):
    """plotly offline mode

    Parameter
    ---------
    connected : boolean
        default False
    """
    if offline_mode:
        plotly.offline.init_notebook_mode(connected=False)
    else:
        plotly.offline.init_notebook_mode(connected=True)
