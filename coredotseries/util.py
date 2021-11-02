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


def plotly_offline_mode(connected=False):
    """plotly offline mode

    Parameter
    ---------
    connected : boolean
        default False
    """
    plotly.offline.init_notebook_mode(connected=connected)

