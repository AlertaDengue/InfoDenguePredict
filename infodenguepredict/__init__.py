import pkg_resources
from .models.deeplearning import lstm


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'

models =
