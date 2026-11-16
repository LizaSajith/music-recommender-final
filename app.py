"""Music Similarity Engine - Streamlit front end for the hybrid recommender.

Run from the repo root:  python -m streamlit run app.py
"""
import html

import numpy as np
import pandas as pd
import streamlit as st

from config import K
from content_based.content_based import FEATURES
from hybrid import build_hybrid_recommender, recommend_hybrid

st.set_page_config(
    page_title="Music Similarity Engine",
    page_icon="🎵",
    layout="wide"
)

# ---------------------------------------------------------------- styling
CSS = """
<style>
.stApp {
  background: radial-gradient(1100px 520px at 12% -8%, #2b2263 0%, #0f1020 58%) fixed;
}
section[data-testid="stSidebar"] {
  background: #12132a;
  border-right: 1px solid #24264a;
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }

.hero h1 {
  font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #b6acff 0%, #ff8fa8 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: #a9a6d6; margin: 6px 0 22px 0; font-size: 1.02rem; }

.seed {
  display: flex; gap: 18px; align-items: center;
  background: linear-gradient(135deg, #241f55 0%, #181a33 100%);
  border: 1px solid #34307a; border-radius: 18px;
  padding: 18px 22px; margin-bottom: 18px;
}
.seed .disc {
  width: 64px; height: 64px; border-radius: 14px; flex: none;
  background: linear-gradient(135deg, #8b7bff, #ff6b8b);
  display: flex; align-items: center; justify-content: center; font-size: 30px;
}
.seed .label { color: #a9a6d6; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.seed .name { font-size: 1.45rem; font-weight: 700; color: #fff; line-height: 1.2; }
.seed .artist { color: #cfcdf5; }

.pill {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  background: #272a55; color: #cfcdf5; font-size: .78rem; margin-right: 6px;
  border: 1px solid #353a75;
}
.pill.accent { background: #3a2f86; border-color: #5a4fd0; color: #fff; }

.card {
  display: grid; grid-template-columns: 40px minmax(0, 1fr) 270px; gap: 18px;
  align-items: center; background: #181a33; border: 1px solid #262850;
  border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;
  transition: border-color .15s, transform .15s;
}
.card:hover { border-color: #6f61e6; transform: translateY(-1px); }
.card .rank { font-size: 1.3rem; font-weight: 800; color: #6f68b8; text-align: center; }
.card .title { font-weight: 700; color: #fff; font-size: 1.04rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card .artist { color: #b4b1e0; font-size: .92rem; margin-bottom: 6px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.bar { display: grid; grid-template-columns: 74px 1fr 40px; gap: 8px;
  align-items: center; font-size: .75rem; color: #a9a6d6; margin: 3px 0; }
.bar .track { height: 7px; background: #262850; border-radius: 99px; overflow: hidden; }
.bar .fill { height: 100%; border-radius: 99px; }
.bar .val { text-align: right; color: #eceaff; font-variant-numeric: tabular-nums; }
.f-content { background: #8b7bff; }
.f-collab  { background: #ffb454; }
.f-hybrid  { background: #ff6b8b; }
.f-seed    { background: #8b7bff; }
.f-rec     { background: #ff6b8b; }

.feat { display: grid; grid-template-columns: 130px 1fr; gap: 12px;
  align-items: center; margin: 10px 0; }
.feat .name { color: #cfcdf5; text-transform: capitalize; }
.feat .pair .bar { grid-template-columns: 1fr 40px; margin: 2px 0; }

.legend span { margin-right: 16px; font-size: .85rem; color: #cfcdf5; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 99px; margin-right: 6px; }

.hist { display: flex; justify-content: space-between; gap: 12px;
  background: #181a33; border: 1px solid #262850; border-radius: 12px;
  padding: 10px 16px; margin-bottom: 8px; }
.hist .stars { color: #ffb454; letter-spacing: 2px; white-space: nowrap; }

@media (max-width: 820px) {
  .card { grid-template-columns: 30px 1fr; }
  .card .scores { grid-column: 1 / -1; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

esc = html.escape


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading songs and training the models...")
def load_app_data():
    songs, _, interactions, matrix_df, _ = build_hybrid_recommender()
    song_list = (
        songs[["track_name", "artists", "track_genre"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return song_list, interactions, list(matrix_df.index)


@st.cache_data
def feature_ranges():
    songs = build_hybrid_recommender()[0]
    return songs[FEATURES].agg(["min", "max"])


try:
    songs, interactions, users = load_app_data()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.info(
        "Check that data/dataset.csv and data/interactions.csv exist. "
        "If interactions.csv is missing, run: python generate_interactions.py"
    )
    st.stop()


# ---------------------------------------------------------------- html helpers
def bar(label, value, css_class):
    pct = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="bar"><span>{esc(label)}</span>'
        f'<div class="track"><div class="fill {css_class}" style="width:{pct:.0f}%"></div></div>'
        f'<span class="val">{value:.2f}</span></div>'
    )


def result_card(rank, row, hybrid_mode):
    if hybrid_mode:
        scores = (
            bar("Content", row["content_score"], "f-content")
            + bar("Collab", row["collab_score"], "f-collab")
            + bar("Hybrid", row["hybrid_score"], "f-hybrid")
        )
    else:
        scores = bar("Similarity", row["content_score"], "f-content")
    return (
        f'<div class="card"><div class="rank">{rank}</div>'
        f'<div><div class="title">{esc(str(row["track_name"]))}</div>'
        f'<div class="artist">{esc(str(row["artists"]))}</div>'
        f'<span class="pill">{esc(str(row["track_genre"]))}</span></div>'
        f'<div class="scores">{scores}</div></div>'
    )


def feature_rows(seed_row, rec_row, ranges):
    out = ""
    for f in FEATURES:
        lo, hi = ranges.loc["min", f], ranges.loc["max", f]
        span = (hi - lo) or 1.0
        a = (seed_row[f] - lo) / span
        b = (rec_row[f] - lo) / span
        out += (
            f'<div class="feat"><div class="name">{esc(f)}</div><div class="pair">'
            f'{bar_single(a, "f-seed")}{bar_single(b, "f-rec")}</div></div>'
        )
    return out


def bar_single(value, css_class):
    pct = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="bar"><div class="track"><div class="fill {css_class}" '
        f'style="width:{pct:.0f}%"></div></div><span class="val">{value:.2f}</span></div>'
    )


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### 🎛️ Settings")

song_options = sorted(songs["track_name"].dropna().astype(str).unique().tolist())

selected_song = st.sidebar.selectbox(
    "Choose a song",
    song_options,
    index=song_options.index("Comedy") if "Comedy" in song_options else 0
)

selected_user = st.sidebar.selectbox(
    "User profile",
    ["Content-only"] + users,
    help="Pick a user to add collaborative filtering. Content-only ranks "
         "all songs by audio similarity alone."
)
hybrid_mode = selected_user != "Content-only"

alpha = st.sidebar.slider(
    "Content weight (alpha)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    disabled=not hybrid_mode,
    help="1 = content-only; 0 = collaborative-only."
)

top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

st.sidebar.caption(
    f"hybrid = α · content + (1 − α) · collab  \nSVD rank k = {K}"
)

# ---------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><h1>🎵 Music Similarity Engine</h1>'
    '<p>Find songs that sound alike, then personalise them with what '
    'similar listeners enjoy.</p></div>',
    unsafe_allow_html=True
)

user_id = selected_user if hybrid_mode else None

try:
    results = recommend_hybrid(
        song_name=selected_song,
        user_id=user_id,
        n=top_n,
        alpha=alpha
    )
except Exception as exc:
    st.error(f"Recommendation failed: {exc}")
    st.stop()

seed_info = songs[songs["track_name"] == selected_song].iloc[0]
mode_pill = (
    f'<span class="pill accent">Hybrid · user {esc(str(selected_user))} · α = {alpha:.1f}</span>'
    if hybrid_mode else '<span class="pill accent">Content-only</span>'
)
st.markdown(
    f'<div class="seed"><div class="disc">🎧</div><div>'
    f'<div class="label">Because you picked</div>'
    f'<div class="name">{esc(str(seed_info["track_name"]))}</div>'
    f'<div class="artist">{esc(str(seed_info["artists"]))}</div>'
    f'<div style="margin-top:8px"><span class="pill">{esc(str(seed_info["track_genre"]))}</span>'
    f'{mode_pill}</div></div></div>',
    unsafe_allow_html=True
)

tab_rec, tab_why, tab_hist = st.tabs(
    ["Recommendations", "Why this song?", "Listening history"]
)

# ---------------------------------------------------------------- tab 1
with tab_rec:
    if results.empty:
        st.warning("No recommendations were found.")
    else:
        legend = (
            '<div class="legend"><span><i class="dot f-content"></i>Content (audio similarity)</span>'
            + (
                '<span><i class="dot f-collab"></i>Collaborative (similar users)</span>'
                '<span><i class="dot f-hybrid"></i>Hybrid (final rank)</span>'
                if hybrid_mode else ""
            )
            + '</div>'
        )
        st.markdown(legend, unsafe_allow_html=True)
        st.write("")
        st.markdown(
            "".join(
                result_card(i + 1, row, hybrid_mode)
                for i, row in results.iterrows()
            ),
            unsafe_allow_html=True
        )

        with st.expander("Show as table / download"):
            display = results.drop(columns=["track_id"])
            st.dataframe(
                display,
                hide_index=True,
                column_config={
                    "content_score": st.column_config.NumberColumn("Content score", format="%.3f"),
                    "collab_score": st.column_config.NumberColumn("Collaborative score", format="%.3f"),
                    "hybrid_score": st.column_config.NumberColumn("Hybrid score", format="%.3f"),
                }
            )
            st.download_button(
                "Download recommendations as CSV",
                data=results.to_csv(index=False),
                file_name="hybrid_recommendations.csv",
                mime="text/csv"
            )

# ---------------------------------------------------------------- tab 2
with tab_why:
    if results.empty:
        st.info("Nothing to compare yet.")
    else:
        full, X_unit, _, _, _ = build_hybrid_recommender()
        ranges = feature_ranges()

        choice = st.selectbox(
            "Compare the selected song with",
            list(range(len(results))),
            format_func=lambda i: f"{i + 1}. {results.loc[i, 'track_name']} — {results.loc[i, 'artists']}"
        )
        rec = results.loc[choice]

        q_idx = int(
            full.index[full["track_name"].str.casefold() == selected_song.casefold()][0]
        )
        r_idx = int(full.index[full["track_id"] == rec["track_id"]][0])
        cosine = float(X_unit[q_idx] @ X_unit[r_idx])

        c1, c2, c3 = st.columns(3)
        c1.metric("Cosine similarity (audio)", f"{cosine:.3f}")
        c2.metric("Content score (0-1)", f"{rec['content_score']:.3f}")
        c3.metric("Hybrid score", f"{rec['hybrid_score']:.3f}")

        if hybrid_mode:
            st.caption(
                f"Hybrid = {alpha:.1f} × {rec['content_score']:.3f} (content) + "
                f"{1 - alpha:.1f} × {rec['collab_score']:.3f} (collaborative) "
                f"= {rec['hybrid_score']:.3f}"
            )

        st.markdown(
            '<div class="legend" style="margin-top:12px">'
            f'<span><i class="dot f-seed"></i>{esc(selected_song)}</span>'
            f'<span><i class="dot f-rec"></i>{esc(str(rec["track_name"]))}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            feature_rows(full.loc[q_idx], full.loc[r_idx], ranges),
            unsafe_allow_html=True
        )
        st.caption(
            "Bars are each feature rescaled to 0-1 over the whole dataset so "
            "features with different units can be compared. Cosine similarity is "
            "computed on the z-scored feature vectors (angle between them); the "
            "content score is that cosine rescaled to 0-1 across the songs being ranked."
        )

# ---------------------------------------------------------------- tab 3
with tab_hist:
    if not hybrid_mode:
        st.info(
            "Pick a user profile in the sidebar to see what they rated. "
            "Collaborative filtering learns from these ratings."
        )
    else:
        full = build_hybrid_recommender()[0]
        mine = interactions[interactions["user_id"] == selected_user].merge(
            full[["track_id", "track_name", "artists", "track_genre"]],
            on="track_id", how="left"
        )
        loved = mine[mine["rating"] >= 4]

        m1, m2, m3 = st.columns(3)
        m1.metric("Songs rated", len(mine))
        m2.metric("Average rating", f"{mine['rating'].mean():.2f}")
        m3.metric("Rated 4-5 stars", len(loved))

        top_genres = loved["track_genre"].value_counts().head(5)
        if len(top_genres):
            st.markdown(
                "**Favourite genres:** " + "".join(
                    f'<span class="pill accent">{esc(str(g))} · {n}</span>'
                    for g, n in top_genres.items()
                ),
                unsafe_allow_html=True
            )
        st.write("")
        st.markdown("**Highest rated songs**")
        top = mine.sort_values("rating", ascending=False).head(10)
        st.markdown(
            "".join(
                f'<div class="hist"><div><b>{esc(str(r["track_name"]))}</b> '
                f'<span style="color:#b4b1e0">— {esc(str(r["artists"]))}</span> '
                f'<span class="pill">{esc(str(r["track_genre"]))}</span></div>'
                f'<div class="stars">{"★" * int(r["rating"])}{"☆" * (5 - int(r["rating"]))}</div></div>'
                for _, r in top.iterrows()
            ),
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------- about
with st.expander("About this project"):
    st.markdown(f"""
- **Content-based filtering:** each song is a vector of 8 z-scored audio features;
  songs are ranked by cosine similarity to the selected song.
- **Collaborative filtering:** a user x song rating matrix, unrated cells filled with
  each user's mean, then truncated SVD (rank k = {K}) to predict ratings.
- **Hybrid ranking:** both scores are scaled to 0-1 and blended with the chosen weight.

The ratings are synthetically generated (20 users, a 500-song catalog), so the
collaborative side demonstrates the method rather than real listening behaviour.
With a user selected, only catalog songs can be recommended.
""")
