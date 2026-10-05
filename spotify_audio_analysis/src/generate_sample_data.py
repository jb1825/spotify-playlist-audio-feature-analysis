"""
Generate a SYNTHETIC offline dataset so the project runs without API keys.

The per-genre means / spreads below are loosely based on well-known tendencies
of Spotify's audio features (e.g. classical = low energy & high acousticness,
metal = very high energy & loudness, hip-hop = high speechiness/danceability).
They are NOT real measurements -- for real results use --mode api or --mode csv.
"""
import numpy as np
import pandas as pd

from config import GENRES, RANDOM_STATE, SAMPLE_CSV

# genre: {feature: (mean, std)}
PROFILES = {
    "pop":        dict(danceability=(0.65, 0.12), energy=(0.65, 0.15), valence=(0.55, 0.20), acousticness=(0.20, 0.20), tempo=(118, 22), loudness=(-6.5, 2.0)),
    "hip-hop":    dict(danceability=(0.75, 0.10), energy=(0.65, 0.14), valence=(0.50, 0.20), acousticness=(0.15, 0.17), tempo=(120, 30), loudness=(-6.8, 2.2)),
    "rock":       dict(danceability=(0.50, 0.12), energy=(0.75, 0.15), valence=(0.50, 0.22), acousticness=(0.12, 0.17), tempo=(125, 25), loudness=(-6.5, 2.5)),
    "electronic": dict(danceability=(0.70, 0.12), energy=(0.78, 0.13), valence=(0.45, 0.22), acousticness=(0.07, 0.12), tempo=(125, 12), loudness=(-6.0, 2.0)),
    "classical":  dict(danceability=(0.30, 0.12), energy=(0.18, 0.15), valence=(0.25, 0.18), acousticness=(0.92, 0.10), tempo=(105, 32), loudness=(-22, 6.0)),
    "jazz":       dict(danceability=(0.50, 0.14), energy=(0.35, 0.18), valence=(0.50, 0.22), acousticness=(0.65, 0.25), tempo=(112, 30), loudness=(-14, 4.5)),
    "country":    dict(danceability=(0.57, 0.12), energy=(0.62, 0.16), valence=(0.58, 0.22), acousticness=(0.30, 0.25), tempo=(122, 26), loudness=(-7.5, 2.5)),
    "metal":      dict(danceability=(0.38, 0.11), energy=(0.90, 0.07), valence=(0.35, 0.20), acousticness=(0.02, 0.05), tempo=(133, 30), loudness=(-5.5, 2.0)),
}
BOUNDS = dict(danceability=(0, 1), energy=(0, 1), valence=(0, 1), acousticness=(0, 1),
              tempo=(50, 220), loudness=(-40, 0))


def generate(n_per_genre: int = 250, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for genre in GENRES:
        data = {"genre": genre}
        cols = {}
        for feat, (mu, sd) in PROFILES[genre].items():
            lo, hi = BOUNDS[feat]
            cols[feat] = np.clip(rng.normal(mu, sd, n_per_genre), lo, hi)
        # realistic coupling: louder tracks tend to be more energetic
        cols["loudness"] = np.clip(cols["loudness"] + 6 * (cols["energy"] - PROFILES[genre]["energy"][0]), -40, 0)
        df = pd.DataFrame(cols)
        df["genre"] = genre
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "track_id", [f"SAMPLE{i:05d}" for i in range(len(df))])
    df.insert(1, "track_name", [f"{g.title()} Track {i}" for i, g in enumerate(df["genre"])])
    df["artist"] = "Synthetic Artist"
    df["popularity"] = np.clip(rng.normal(50, 18, len(df)), 0, 100).round().astype(int)
    return df.round(4)


if __name__ == "__main__":
    df = generate()
    SAMPLE_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(SAMPLE_CSV, index=False)
    print(f"Wrote {len(df)} synthetic tracks -> {SAMPLE_CSV}")
