"""soundalike: an open-source music recommender.

Find songs similar to the ones you like using content-based similarity on audio
features, optionally blended with Last.fm listening data and your live Spotify
taste. Built to work *without* Spotify's deprecated Recommendations / Audio
Features endpoints (locked for new apps as of 2024-11-27).
"""

from .dataset import Dataset, load_bundled_dataset
from .features import AUDIO_FEATURES, FeatureConfig
from .recommender import ContentBasedRecommender, Recommendation

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "load_bundled_dataset",
    "AUDIO_FEATURES",
    "FeatureConfig",
    "ContentBasedRecommender",
    "Recommendation",
    "__version__",
]
