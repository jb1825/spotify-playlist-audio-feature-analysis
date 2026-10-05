# Spotify Playlist Audio Feature Analysis

Data-science mini project: pull playlist tracks and audio features from music APIs, then graph and
statistically compare **danceability, energy and tempo** (plus valence, acousticness, loudness) across genres.

## Important: Spotify API change
Spotify's `/audio-features`, `/audio-analysis` and `/recommendations` endpoints return **HTTP 403 for apps created after
27 Nov 2024**, and Spotify-owned editorial playlists are blocked for new apps. This project therefore:
1. gets playlist tracks + artist genres from the **Spotify Web API** (still available),
2. tries Spotify's audio-features endpoint, and on 403 **falls back to the free ReccoBeats API**,
3. also supports **CSV mode** (e.g. the Kaggle "Spotify Tracks Dataset") and **sample mode** (offline, synthetic data).

> ReccoBeats is a third-party free API; coverage can be incomplete. If it fails, use CSV mode.
> The bundled sample data is synthetic, so results from `--mode sample` demonstrate the pipeline, not real music.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
# 1) Offline demo (no keys, no internet)
python main.py --mode sample

# 2) Any CSV with columns: genre (or track_genre), danceability, energy, valence, acousticness, tempo, loudness
python main.py --mode csv --csv data/dataset.csv

# 3) Live APIs  (copy .env.example to .env and add your Spotify keys first)
python main.py --mode api --playlists https://open.spotify.com/playlist/AAA https://open.spotify.com/playlist/BBB
```
Tip for API mode: use several **user-made public playlists** of different genres (e.g. "classical piano", "heavy metal",
"hip hop", "EDM"). Genre is taken from each track's first artist.

## Build the PDF report
Edit your name / roll number at the top of `report/build_report.py`, run a mode above, then:
```bash
python report/build_report.py
```
Output: `outputs/Spotify_Audio_Feature_Analysis_Report.pdf` (upload this to the PBL portal).
If you used API/CSV mode, also change `DATA_NOTE` in that file so the report describes your data correctly.

## Outputs
- `outputs/figures/` - 7 graphs (boxplots, heatmap, scatter, tempo violin, correlation, radar, PCA/K-Means)
- `outputs/genre_summary_stats.csv`, `anova_results.csv`, `cleaned_dataset.csv`, `results.json`

## Project structure
```
main.py  config.py  requirements.txt  .env.example
src/data_sources.py          Spotify + ReccoBeats clients, CSV loader
src/generate_sample_data.py  synthetic offline dataset
src/analysis.py              cleaning, ANOVA, graphs, PCA + K-Means
report/build_report.py       builds the report PDF
```
