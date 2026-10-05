"""
Spotify Playlist Audio Feature Analysis -- command line entry point.

Examples
  python main.py --mode sample
  python main.py --mode csv --csv data/dataset.csv
  python main.py --mode api --playlists https://open.spotify.com/playlist/XXXX https://open.spotify.com/playlist/YYYY
"""
import argparse

from src.analysis import run_all
from src.data_sources import load_data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sample", "csv", "api"], default="sample")
    p.add_argument("--csv", help="path to CSV (mode=csv)")
    p.add_argument("--playlists", nargs="+", help="Spotify playlist URLs/IDs (mode=api)")
    a = p.parse_args()

    df = load_data(a.mode, a.playlists, a.csv)
    print(f"Loaded {len(df)} rows from mode='{a.mode}'")
    res = run_all(df, label=a.mode)

    print("\n=== KEY RESULTS ===")
    for k, v in res.items():
        print(f"{k:28s}: {v}")
    print("\nFigures -> outputs/figures/   Tables -> outputs/*.csv   Summary -> outputs/results.json")


if __name__ == "__main__":
    main()
