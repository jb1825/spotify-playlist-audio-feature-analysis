"""Builds the mini-project report PDF from the files in outputs/. Run AFTER main.py.
Edit STUDENT_* below with your own details first."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# ----------------------------- EDIT THESE ------------------------------------
STUDENT_NAME = "Your Name"
ROLL_NO = "Roll No / Enrollment No"
COURSE = "Data Science (PBL)"
INSTITUTE = "Your Institute / Department"
GUIDE = "Faculty Guide Name"
DATA_NOTE = ("synthetic offline sample dataset (2,000 tracks, 8 genres)")   # change if you ran api/csv mode
# -----------------------------------------------------------------------------

OUT = ROOT / "outputs"
FIG = OUT / "figures"
res = json.loads((OUT / "results.json").read_text())
summ = pd.read_csv(OUT / "genre_summary_stats.csv")
anova = pd.read_csv(OUT / "anova_results.csv")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#1DB954"), spaceBefore=10, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#191414"), spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=ss["Title"], fontSize=22, leading=28, textColor=colors.HexColor("#191414"))
center = ParagraphStyle("center", parent=body, alignment=TA_CENTER)
cap = ParagraphStyle("cap", parent=body, fontSize=9, alignment=TA_CENTER, textColor=colors.grey, italic=True)
code = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=7.8, backColor=colors.HexColor("#F4F4F4"), borderPadding=3)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=2)

P = lambda t, s=body: Paragraph(t, s)
B = lambda t: Paragraph(t, bullet, bulletText="\u2022")


def fig(name, caption, width=16):
    img = Image(str(FIG / name))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width * cm, width * cm * ratio
    return [img, P(caption, cap), Spacer(1, 6)]


def table(data, col_widths=None, font=8.5, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#191414") if header else colors.white), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F9F4")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, f"{STUDENT_NAME} | {ROLL_NO} | Spotify Playlist Audio Feature Analysis")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def wrap_code(text, width=115):
    """Hard-wrap long source lines so they stay inside the page (PDF only; files are unchanged)."""
    out = []
    for line in text.split("\n"):
        indent = " " * (len(line) - len(line.lstrip()))
        while len(line) > width:
            cut = line.rfind(", ", 0, width)
            cut = cut + 1 if cut > len(indent) + 10 else width
            out.append(line[:cut])
            line = indent + "    " + line[cut:].lstrip()
        out.append(line)
    return "\n".join(out)


s = []
# ------------------------------------------------------------------ cover page
s += [Spacer(1, 3 * cm), P("MINI PROJECT REPORT", center), Spacer(1, 10),
      P("Spotify Playlist Audio Feature Analysis", title),
      P("Extracting and graphing danceability, energy and tempo across musical genres", center),
      Spacer(1, 2 * cm)]
s.append(table([["Student Name", STUDENT_NAME], ["Roll / Enrollment No.", ROLL_NO], ["Course", COURSE],
                ["Institute", INSTITUTE], ["Faculty Guide", GUIDE], ["Technology", "Python, pandas, matplotlib, seaborn, scikit-learn, REST APIs"]],
               col_widths=[5 * cm, 10.5 * cm], font=10, header=False))
s.append(PageBreak())

# ------------------------------------------------------------------- abstract
s += [P("Abstract", h1),
      P("This project builds a reproducible data-science pipeline that collects track-level audio metrics "
        "(danceability, energy, valence, acousticness, tempo and loudness) for tracks in Spotify playlists, groups the tracks by genre, "
        "and analyses how the metrics differ between genres using descriptive statistics, one-way ANOVA with effect sizes, "
        "visualisation, and an unsupervised (PCA + K-Means) check. The pipeline connects to the Spotify Web API for playlist and "
        "artist-genre data and to the free ReccoBeats API for audio features, because Spotify closed its own audio-features endpoint to "
        "newly created apps on 27 November 2024. An offline mode makes the project runnable without API keys."),
      P(f"<b>Data used for the results in this report:</b> {DATA_NOTE}. The sample data is generated by "
        "<font face='Courier'>src/generate_sample_data.py</font> from assumed genre profiles; it is <b>not</b> real Spotify data, so the "
        "numbers below demonstrate that the pipeline works rather than being findings about real music. "
        "Running <font face='Courier'>python main.py --mode api</font> or <font face='Courier'>--mode csv</font> regenerates every table "
        "and figure from real data."),
      P("Keywords: Spotify Web API, audio features, danceability, energy, tempo, genre analysis, ANOVA, PCA, K-Means.", body)]

# ----------------------------------------------------------------- introduction
s += [P("1. Introduction and Objectives", h1),
      P("Streaming platforms describe each track with numeric audio descriptors. Comparing these descriptors across genres helps "
        "understand what makes genres sonically distinct and supports tasks such as playlist curation and recommendation. "
        "The objectives of this project are:"),
      B("Connect to music APIs and extract playlist tracks, artist genres and audio features programmatically."),
      B("Clean and combine the data into one analysis-ready table."),
      B("Graph danceability, energy and tempo (plus supporting metrics) across genres."),
      B("Test statistically whether the differences between genres are significant and how large they are."),
      B("Check whether audio features alone can recover genre groupings (clustering)."),
      P("2. Audio Features Used", h1),
      table([["Feature", "Range", "Meaning"],
             ["danceability", "0 - 1", "Suitability for dancing (tempo stability, beat strength, regularity)"],
             ["energy", "0 - 1", "Perceived intensity and activity (fast, loud, noisy tracks score high)"],
             ["valence", "0 - 1", "Musical positiveness (high = happy/cheerful, low = sad/tense)"],
             ["acousticness", "0 - 1", "Confidence the track is acoustic"],
             ["tempo", "BPM", "Estimated beats per minute"],
             ["loudness", "dB", "Overall loudness (typically -60 to 0 dB)"]],
            col_widths=[3 * cm, 2.5 * cm, 10 * cm], font=9)]

# ----------------------------------------------------------------- methodology
s += [P("3. System Design and Methodology", h1),
      P("The project is organised as follows:"),
      Preformatted("spotify_audio_analysis/\n  main.py                  CLI entry point (--mode sample | csv | api)\n  config.py                paths, feature lists, genres\n"
                   "  requirements.txt  .env.example\n  src/data_sources.py      Spotify + ReccoBeats API clients, CSV loader\n"
                   "  src/generate_sample_data.py   offline synthetic dataset\n  src/analysis.py          cleaning, statistics, 7 graphs, clustering\n"
                   "  report/build_report.py   builds this PDF\n  data/  outputs/          inputs and results (tables, figures)", code),
      Spacer(1, 6),
      P("<b>Pipeline.</b> (1) Authenticate to Spotify with the Client Credentials flow. (2) Read every track of each public playlist, "
        "following pagination. (3) Look up the primary genre of each track's first artist (Spotify stores genres at artist level, not track level). "
        "(4) Request audio features: Spotify's endpoint is tried first; on HTTP 403 the code falls back to ReccoBeats. "
        "(5) Merge, drop missing values, duplicates and failed tempo detections (tempo = 0), and keep genres with at least 15 tracks. "
        "(6) Compute per-genre statistics, ANOVA and eta-squared, draw graphs, and run PCA + K-Means."),
      P("<b>Statistics.</b> For each feature a one-way ANOVA tests whether genre means differ. Because p-values become tiny with large samples, "
        "eta-squared (share of variance explained by genre) is reported as the effect size. K-Means (k = number of genres) is run on "
        "standardised features and scored by cluster purity against the true genre."),
      P("<b>API limitations (important).</b> Spotify's audio-features, audio-analysis and recommendations endpoints return HTTP 403 for apps "
        "created after 27 November 2024, and Spotify-owned editorial playlists are also restricted for new apps. "
        "The code therefore uses user-created public playlists and a documented fallback; ReccoBeats coverage is not guaranteed, so "
        "tracks it does not return are dropped and the final track count is reported.")]

# --------------------------------------------------------------------- results
s += [PageBreak(), P("4. Results and Outcome", h1),
      P(f"After cleaning, the analysis used <b>{res['n_tracks']:,}</b> tracks across <b>{res['n_genres']}</b> genres "
        f"({', '.join(res['genres'])}).")]

cols = ["genre", "n_tracks", "danceability_mean", "energy_mean", "tempo_mean", "valence_mean", "acousticness_mean", "loudness_mean"]
hdr = ["Genre", "N", "Dance.", "Energy", "Tempo", "Valence", "Acoust.", "Loud. (dB)"]
rows = [hdr] + [[r["genre"], int(r["n_tracks"])] + [f"{r[c]:.2f}" if c != "tempo_mean" else f"{r[c]:.0f}" for c in cols[2:]] for _, r in summ.iterrows()]
s += [P("4.1 Genre summary (means)", h2), table(rows, font=8.5), Spacer(1, 8)]

s += [P("4.2 Danceability, energy and tempo across genres", h2)]
s += fig("01_boxplots_core_features.png", "Figure 1. Distribution of the three headline metrics by genre (sorted by median energy).")
s += [P(f"<b>Energy</b> is highest for <b>{res['highest_energy']}</b> and lowest for <b>{res['lowest_energy']}</b>. "
        f"<b>Danceability</b> is highest for <b>{res['most_danceable']}</b> and lowest for <b>{res['least_danceable']}</b>. "
        f"Mean <b>tempo</b> is highest for <b>{res['fastest']}</b> and lowest for <b>{res['slowest']}</b>, but tempo distributions overlap heavily.")]
s += fig("04_tempo_distribution.png", "Figure 2. Tempo (BPM) distribution by genre (violin plot with quartiles).", 13)
s += fig("03_danceability_vs_energy.png", "Figure 3. Danceability versus energy; genres occupy different regions of the plane.", 13)
s += fig("02_genre_feature_heatmap.png", "Figure 4. Mean value of every feature by genre (colour = min-max scaled, text = real mean).", 14)
s += fig("06_radar_genre_profiles.png", "Figure 5. Radar chart of each genre's four 0-1 features.", 11)
s += fig("05_correlation_matrix.png", "Figure 6. Pearson correlations between features.", 11)

s += [P("4.3 Statistical tests", h2),
      table([["Feature", "F", "p-value", "eta-squared"]] +
            [[r["feature"], f"{r['F']:.1f}", ("< 1e-30" if r["p_value"] < 1e-30 else f"{r['p_value']:.1e}"), f"{r['eta_squared']:.3f}"] for _, r in anova.iterrows()],
            col_widths=[4 * cm, 3 * cm, 3.5 * cm, 3.5 * cm], font=9),
      Spacer(1, 6),
      P("All differences between genres are statistically significant (p &lt; 0.001). The effect sizes show that genre explains most of the variation "
        "in acousticness, loudness and energy, a large share of danceability, and comparatively little of tempo. In other words, tempo is the weakest "
        "single descriptor for separating genres.")]
s += [P("4.4 Correlations and clustering", h2),
      P(f"Energy correlates positively with loudness (r = {res['corr_energy_loudness']}) and negatively with acousticness (r = {res['corr_energy_acousticness']}). "
        f"Danceability and valence are only weakly related (r = {res['corr_danceability_valence']}). "
        f"K-Means on the standardised features reaches a cluster purity of {res['kmeans_purity']} against the true genres "
        f"(chance level for {res['n_genres']} equal genres is about 0.125), so audio features carry real genre information, "
        "but genres with similar profiles (e.g. pop, hip-hop, country, rock) remain hard to separate without extra features.")]
s += fig("07_pca_kmeans_clusters.png", "Figure 7. PCA projection coloured by genre (left) and by K-Means cluster (right).", 16)

# ------------------------------------------------------------------ conclusion
s += [P("5. Conclusion", h1),
      P("The project delivers a working, reproducible pipeline from API extraction to statistical analysis and graphs. On the data used here, "
        "energy, acousticness and loudness separate genres most strongly, danceability separates them moderately, and tempo least. "
        "Because this report's numbers come from a synthetic sample, the conclusions describe the expected behaviour of the method; the same code "
        "should be run on real playlists to draw conclusions about real music."),
      P("<b>Limitations.</b> Spotify's audio features are model estimates, not ground truth; genre is assigned per artist (an artist's first listed genre), "
        "so individual tracks can be mislabelled; ReccoBeats coverage and consistency with Spotify's original values are not guaranteed; "
        "playlists are a biased sample of music."),
      P("<b>Future work.</b> Add a supervised genre classifier, compare decades or markets, add time-series analysis of a listener's playlist, "
        "and build a Streamlit dashboard.")]

s += [P("References", h1),
      B("Spotify for Developers, Web API reference: developer.spotify.com/documentation/web-api"),
      B("Spotify for Developers blog, \"Changes to Web API\" (27 Nov 2024): developer.spotify.com/blog/2024-11-27-changes-to-the-web-api"),
      B("ReccoBeats API documentation: reccobeats.com/docs"),
      B("pandas, matplotlib, seaborn, SciPy and scikit-learn documentation.")]

# ------------------------------------------------------------------- appendix
s.append(PageBreak())
s.append(P("Appendix A: Source Code", h1))
for rel in ["main.py", "config.py", "src/data_sources.py", "src/generate_sample_data.py", "src/analysis.py"]:
    s.append(P(rel, h2))
    for chunk in (ROOT / rel).read_text().replace("\t", "    ").split("\n\n\n"):
        s.append(Preformatted(wrap_code(chunk), code))
        s.append(Spacer(1, 4))

out_pdf = OUT / "Spotify_Audio_Feature_Analysis_Report.pdf"
SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title="Spotify Playlist Audio Feature Analysis", author=STUDENT_NAME).build(s, onFirstPage=footer, onLaterPages=footer)
print("Report written ->", out_pdf)
