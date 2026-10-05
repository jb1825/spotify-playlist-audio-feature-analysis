import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Spotify Playlist Audio Feature Analysis",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Playlist Audio Feature Analysis")
st.write(
    "Analyze Spotify tracks using audio features such as "
    "danceability, energy, valence, acousticness, tempo and loudness."
)

# Load dataset
df = pd.read_csv("data/sample_tracks.csv")

st.success(f"Dataset loaded successfully: {len(df)} tracks")

# Sidebar
st.sidebar.header("Filters")

genres = st.sidebar.multiselect(
    "Select Genre",
    sorted(df["genre"].unique()),
    default=sorted(df["genre"].unique())
)

filtered_df = df[df["genre"].isin(genres)]

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tracks", len(filtered_df))
col2.metric("Genres", filtered_df["genre"].nunique())
col3.metric(
    "Average Energy",
    round(filtered_df["energy"].mean(), 2)
)
col4.metric(
    "Average Danceability",
    round(filtered_df["danceability"].mean(), 2)
)

st.divider()

# Summary
st.subheader("📊 Genre Summary")

summary = filtered_df.groupby("genre")[
    [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "tempo",
        "loudness"
    ]
].mean().round(2)

st.dataframe(summary, use_container_width=True)

# Energy
st.subheader("⚡ Energy by Genre")

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=filtered_df,
    x="genre",
    y="energy",
    ax=ax
)

ax.set_xlabel("Genre")
ax.set_ylabel("Average Energy")
plt.xticks(rotation=45)

st.pyplot(fig)

# Danceability
st.subheader("💃 Danceability by Genre")

fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(
    data=filtered_df,
    x="genre",
    y="danceability",
    ax=ax
)

ax.set_xlabel("Genre")
ax.set_ylabel("Danceability")
plt.xticks(rotation=45)

st.pyplot(fig)

# Tempo
st.subheader("🥁 Tempo Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

sns.violinplot(
    data=filtered_df,
    x="genre",
    y="tempo",
    ax=ax
)

ax.set_xlabel("Genre")
ax.set_ylabel("Tempo (BPM)")
plt.xticks(rotation=45)

st.pyplot(fig)

# Correlation
st.subheader("🔗 Audio Feature Correlation")

features = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "tempo",
    "loudness"
]

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    filtered_df[features].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# Dataset
with st.expander("📋 View Dataset"):
    st.dataframe(
        filtered_df,
        use_container_width=True
    )

st.divider()

st.caption(
    "Spotify Playlist Audio Feature Analysis | Data Science PBL Project"
)