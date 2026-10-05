"""
Data loading for the project. Three modes:

  sample : synthetic offline dataset (no keys, no internet)
  csv    : any CSV with genre + audio-feature columns (e.g. the Kaggle
           "Spotify Tracks Dataset" which has a `track_genre` column)
  api    : live pull -- playlist tracks + artist genres from the Spotify Web API,
           audio features from Spotify (if your app still has access) or from the
           free ReccoBeats API (fallback).

NOTE: Spotify closed /v1/audio-features to apps created after 27 Nov 2024 (HTTP 403),
which is why the ReccoBeats fallback exists.
"""
import base64
import os
import re
import time

import pandas as pd
import requests

from config import API_CSV, FEATURES, SAMPLE_CSV

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"
RECCO_API = "https://api.reccobeats.com/v1"

FEATURE_COLS = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness"]


# --------------------------------------------------------------------- helpers
def _get(url, headers=None, params=None, retries=4):
    """GET with simple retry/back-off (handles HTTP 429 rate limits)."""
    for attempt in range(retries):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
            continue
        return r
    return r


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def parse_playlist_id(url_or_id: str) -> str:
    m = re.search(r"playlist/([A-Za-z0-9]+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()


# --------------------------------------------------------------------- spotify
def get_spotify_token() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    cid, secret = os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in a .env file "
                           "(see .env.example).")
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    r = requests.post(SPOTIFY_TOKEN_URL, data={"grant_type": "client_credentials"},
                      headers={"Authorization": f"Basic {auth}"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def fetch_playlist_tracks(playlist_id: str, token: str) -> pd.DataFrame:
    """Return track_id, track_name, artist, artist_id, popularity for a PUBLIC playlist."""
    headers = {"Authorization": f"Bearer {token}"}
    url, params, rows = f"{SPOTIFY_API}/playlists/{playlist_id}/tracks", {"limit": 100}, []
    while url:
        r = _get(url, headers, params)
        if r.status_code == 403:
            raise RuntimeError("403 from Spotify: Spotify-owned/editorial playlists are blocked for "
                               "new apps. Use a public playlist created by a normal user.")
        r.raise_for_status()
        data = r.json()
        for item in data["items"]:
            t = item.get("track")
            if not t or not t.get("id"):
                continue
            rows.append(dict(track_id=t["id"], track_name=t["name"],
                             artist=t["artists"][0]["name"], artist_id=t["artists"][0]["id"],
                             popularity=t.get("popularity")))
        url, params = data.get("next"), None
    return pd.DataFrame(rows).drop_duplicates("track_id")


def fetch_artist_genres(artist_ids, token: str) -> dict:
    """Map artist_id -> primary genre string (first genre Spotify lists, else 'unknown')."""
    headers, out = {"Authorization": f"Bearer {token}"}, {}
    for batch in _chunks(list(dict.fromkeys(artist_ids)), 50):
        r = _get(f"{SPOTIFY_API}/artists", headers, {"ids": ",".join(batch)})
        if r.status_code != 200:
            continue
        for a in r.json().get("artists", []):
            if a:
                out[a["id"]] = a["genres"][0] if a.get("genres") else "unknown"
    return out


def fetch_audio_features_spotify(track_ids, token: str):
    """Try Spotify's own endpoint. Returns DataFrame, or None if blocked (403)."""
    headers, rows = {"Authorization": f"Bearer {token}"}, []
    for batch in _chunks(list(track_ids), 100):
        r = _get(f"{SPOTIFY_API}/audio-features", headers, {"ids": ",".join(batch)})
        if r.status_code in (403, 404):
            return None
        r.raise_for_status()
        rows += [f for f in r.json().get("audio_features", []) if f]
    if not rows:
        return None
    df = pd.DataFrame(rows).rename(columns={"id": "track_id"})
    return df[["track_id"] + FEATURE_COLS]


def fetch_audio_features_reccobeats(track_ids) -> pd.DataFrame:
    """
    Free fallback: ReccoBeats returns Spotify-style audio features.
    Docs: https://reccobeats.com/docs  (endpoint accepts Spotify track IDs; each
    result carries an `href` containing the Spotify URL, which we use to map back).
    Coverage is not 100%, so missing tracks are simply dropped.
    """
    rows = []
    for batch in _chunks(list(track_ids), 40):
        r = _get(f"{RECCO_API}/audio-features", params={"ids": ",".join(batch)})
        if r.status_code != 200:
            print(f"  ReccoBeats returned HTTP {r.status_code} for a batch; skipping it.")
            continue
        for item in r.json().get("content", []):
            m = re.search(r"track/([A-Za-z0-9]+)", item.get("href", ""))
            if m:
                rec = {k: item.get(k) for k in FEATURE_COLS}
                rec["track_id"] = m.group(1)
                rows.append(rec)
        time.sleep(0.3)   # be polite to a free API
    return pd.DataFrame(rows)


def load_from_api(playlist_urls) -> pd.DataFrame:
    token = get_spotify_token()
    frames = []
    for url in playlist_urls:
        pid = parse_playlist_id(url)
        print(f"Fetching playlist {pid} ...")
        frames.append(fetch_playlist_tracks(pid, token))
    tracks = pd.concat(frames).drop_duplicates("track_id").reset_index(drop=True)
    print(f"  {len(tracks)} unique tracks")

    genres = fetch_artist_genres(tracks["artist_id"], token)
    tracks["genre"] = tracks["artist_id"].map(genres).fillna("unknown")

    feats = fetch_audio_features_spotify(tracks["track_id"], token)
    if feats is None:
        print("Spotify audio-features is blocked for this app (403) -> using ReccoBeats fallback")
        feats = fetch_audio_features_reccobeats(tracks["track_id"])
    if feats.empty:
        raise RuntimeError("No audio features could be retrieved. Try --mode csv instead.")

    df = tracks.merge(feats, on="track_id", how="inner")
    print(f"  {len(df)} tracks with audio features")
    df.to_csv(API_CSV, index=False)
    print(f"  saved -> {API_CSV}")
    return df


# ------------------------------------------------------------------------ csv
def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "genre" not in df.columns and "track_genre" in df.columns:   # Kaggle naming
        df = df.rename(columns={"track_genre": "genre"})
    if "track_name" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "track_name"})
    missing = [c for c in ["genre"] + FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def load_data(mode: str, playlists=None, csv_path=None) -> pd.DataFrame:
    if mode == "sample":
        if not SAMPLE_CSV.exists():
            from src.generate_sample_data import generate
            SAMPLE_CSV.parent.mkdir(exist_ok=True)
            generate().to_csv(SAMPLE_CSV, index=False)
        return load_csv(SAMPLE_CSV)
    if mode == "csv":
        if not csv_path:
            raise ValueError("--csv PATH is required with --mode csv")
        return load_csv(csv_path)
    if mode == "api":
        if not playlists:
            raise ValueError("--playlists URL [URL ...] is required with --mode api")
        return load_from_api(playlists)
    raise ValueError(f"Unknown mode: {mode}")
