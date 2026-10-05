"""Central configuration for the Spotify Audio Feature Analysis project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

SAMPLE_CSV = DATA_DIR / "sample_tracks.csv"
API_CSV = DATA_DIR / "api_tracks.csv"          # written by --mode api

# Audio metrics analysed in the project
FEATURES = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness"]
CORE_FEATURES = ["danceability", "energy", "tempo"]   # the three headline metrics

# Genres analysed in the offline sample (also a good set for real playlists)
GENRES = ["pop", "hip-hop", "rock", "electronic", "classical", "jazz", "country", "metal"]

RANDOM_STATE = 42
