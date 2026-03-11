import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# ==================================================
# PAGE SETUP
# ==================================================
st.set_page_config(
    page_title="Barbie Style Super Squad",
    page_icon="💗",
    layout="wide"
)

st.markdown("""
<style>
    /* Global typography + background */
    html, body, [class*="st-"] {
        font-family: "Didot", "Georgia", "Times New Roman", serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #ffe2f2 0, #fff7fb 40%, #ffffff 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffeaf4 0%, #ffe0f0 40%, #ffeaf4 100%);
        border-right: 1px solid #f4c7da;
    }

    /* Headings */
    h1, h2, h3 {
        color: #a43d6c;
        font-family: "Didot", "Georgia", "Times New Roman", serif;
        letter-spacing: 0.03em;
    }

    /* App title + subtitle */
    .app-title {
        font-size: 2.6rem;
        color: #a43d6c;
        margin-bottom: 0.2rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .app-subtitle {
        color: #7d5a6b;
        margin-bottom: 1.4rem;
        font-size: 1rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.2rem;
        color: #a43d6c;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Item cards */
    .item-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #f0c9da;
        border-radius: 16px;
        padding: 12px 12px 10px 12px;
        box-shadow: 0 10px 24px rgba(164, 61, 108, 0.08);
        transition: transform 0.15s ease-out, box-shadow 0.15s ease-out, border-color 0.15s ease-out;
    }

    .item-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 32px rgba(164, 61, 108, 0.16);
        border-color: #e69abb;
    }

    .item-name {
        color: #8e315f;
        font-weight: 700;
        font-size: 0.98rem;
        margin-top: 0.55rem;
        margin-bottom: 0.3rem;
        line-height: 1.25;
    }

    .item-meta {
        color: #6f5762;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .score-pill {
        display: inline-block;
        margin-top: 0.45rem;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        background: #ffe3ef;
        color: #8e315f;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .small-note {
        color: #6f5762;
        font-size: 0.93rem;
        margin-bottom: 0.6rem;
    }

    /* Primary button */
    div.stButton > button {
        background: linear-gradient(90deg, #e277a7, #d35f96);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.3rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        box-shadow: 0 6px 14px rgba(190, 76, 123, 0.35);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #d35f96, #c04784);
        color: white;
        box-shadow: 0 10px 22px rgba(190, 76, 123, 0.4);
    }

    /* Select widgets */
    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADING
# ==================================================
CSV_FILE = "fashion.csv"


@st.cache_data
def load_data(csv_file):
    df_raw = pd.read_csv(csv_file)

    # Clean columns
    df_raw.columns = [str(col).strip() for col in df_raw.columns]
    df_raw = df_raw.dropna(axis=1, how="all").copy()

    bad_cols = [col for col in df_raw.columns if col.lower().startswith("unnamed")]
    if bad_cols:
        df_raw = df_raw.drop(columns=bad_cols)

    # Rename image columns if present
    rename_map = {}
    for col in df_raw.columns:
        c = col.lower().strip()
        if c == "filename":
            rename_map[col] = "image_id"
        elif c == "link":
            rename_map[col] = "image_url"

    df_raw = df_raw.rename(columns=rename_map)

    if "image_id" not in df_raw.columns and "id" in df_raw.columns:
        df_raw["image_id"] = df_raw["id"].astype(str) + ".jpg"

    if "image_url" not in df_raw.columns:
        df_raw["image_url"] = np.nan

    wanted_cols = [
        "id",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "year",
        "usage",
        "productDisplayName",
        "image_id",
        "image_url"
    ]

    existing_cols = [col for col in wanted_cols if col in df_raw.columns]
    df = df_raw[existing_cols].copy()

    required_cols = [
        "id",
        "gender",
        "articleType",
        "baseColour",
        "season",
        "usage",
        "productDisplayName"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols).copy()

    for col in df.columns:
        if col != "image_url":
            df[col] = df[col].astype(str).str.strip()

    df["image_url"] = df["image_url"].astype(str).str.strip()
    df.loc[df["image_url"].isin(["", "nan", "None"]), "image_url"] = np.nan

    df = df.drop_duplicates().reset_index(drop=True)

    def keep_top_k(dataframe, col, k):
        top_vals = dataframe[col].value_counts().head(k).index
        return dataframe[dataframe[col].isin(top_vals)].copy()

    # Keep common categories to reduce noise
    df = keep_top_k(df, "gender", 5)
    df = keep_top_k(df, "season", 4)
    df = keep_top_k(df, "usage", 10)
    df = keep_top_k(df, "articleType", 20)
    df = keep_top_k(df, "baseColour", 15)

    df = df.reset_index(drop=True)
    return df


df = load_data(CSV_FILE)

# ==================================================
# PROBABILITY FUNCTIONS
# ==================================================
def conditional_prob(dataframe, target_col, target_val, given_cols, given_vals, alpha=1.0):
    """
    Estimate:
    P(target_col = target_val | given_cols = given_vals)
    using counts + Laplace smoothing.
    """
    filtered = dataframe.copy()

    for col, val in zip(given_cols, given_vals):
        filtered = filtered[filtered[col] == val]

    total_count = len(filtered)
    num_categories = dataframe[target_col].nunique()
    match_count = len(filtered[filtered[target_col] == target_val])

    prob = (match_count + alpha) / (total_count + alpha * num_categories)
    return prob


def build_preference_profile(selected_rows):
    article_counts = Counter(selected_rows["articleType"])
    color_counts = Counter(selected_rows["baseColour"])

    total_articles = sum(article_counts.values())
    total_colors = sum(color_counts.values())

    article_pref = {}
    color_pref = {}

    if total_articles > 0:
        for k, v in article_counts.items():
            article_pref[k] = v / total_articles

    if total_colors > 0:
        for k, v in color_counts.items():
            color_pref[k] = v / total_colors

    return article_pref, color_pref


def score_item(row, dataframe, client_gender, client_season, client_usage, article_pref, color_pref, alpha=1.0):
    article = row["articleType"]
    color = row["baseColour"]

    p_article = conditional_prob(
        dataframe=dataframe,
        target_col="articleType",
        target_val=article,
        given_cols=["gender", "season", "usage"],
        given_vals=[client_gender, client_season, client_usage],
        alpha=alpha
    )

    p_color = conditional_prob(
        dataframe=dataframe,
        target_col="baseColour",
        target_val=color,
        given_cols=["articleType", "season"],
        given_vals=[article, client_season],
        alpha=alpha
    )

    # Small fallback so unseen preferences do not zero everything out
    article_bonus = article_pref.get(article, 0.05)
    color_bonus = color_pref.get(color, 0.05)

    return p_article * p_color * article_bonus * color_bonus


def get_candidate_pool(dataframe, gender, season, usage, n=30):
    pool = dataframe[
        (dataframe["gender"] == gender) &
        (dataframe["season"] == season) &
        (dataframe["usage"] == usage)
    ].copy()

    if len(pool) < 6:
        pool = dataframe[dataframe["gender"] == gender].copy()

    if len(pool) == 0:
        pool = dataframe.copy()

    if len(pool) > n:
        pool = pool.sample(n=n, random_state=42)

    return pool.reset_index(drop=True)


def recommend_items(
    dataframe,
    client_gender,
    client_season,
    client_usage,
    selected_example_rows,
    top_n=12,
    alpha=1.0
):
    article_pref, color_pref = build_preference_profile(selected_example_rows)

    candidate_df = dataframe.copy()

    # Prefer same gender if available
    same_gender = candidate_df[candidate_df["gender"] == client_gender].copy()
    if len(same_gender) > 0:
        candidate_df = same_gender

    # Remove already selected items
    selected_ids = set(selected_example_rows["id"].astype(str).tolist())
    candidate_df = candidate_df[~candidate_df["id"].astype(str).isin(selected_ids)].copy()

    candidate_df["score"] = candidate_df.apply(
        lambda row: score_item(
            row=row,
            dataframe=dataframe,
            client_gender=client_gender,
            client_season=client_season,
            client_usage=client_usage,
            article_pref=article_pref,
            color_pref=color_pref,
            alpha=alpha
        ),
        axis=1
    )

    candidate_df = candidate_df.sort_values("score", ascending=False)
    candidate_df = candidate_df.drop_duplicates(subset=["productDisplayName"])

    return candidate_df.head(top_n).copy()

# ==================================================
# UI HELPERS
# ==================================================
def render_item_card(row, show_score=False, checkbox_key=None):
    st.markdown('<div class="item-card">', unsafe_allow_html=True)

    if pd.notna(row.get("image_url", np.nan)):
        st.image(row["image_url"], use_container_width=True)
    else:
        st.markdown("No image")

    st.markdown(f'<div class="item-name">{row["productDisplayName"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="item-meta">
            <b>ID:</b> {row["id"]}<br>
            <b>Type:</b> {row["articleType"]}<br>
            <b>Color:</b> {row["baseColour"]}<br>
            <b>Season:</b> {row["season"]}<br>
            <b>Usage:</b> {row["usage"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    if show_score and "score" in row:
        st.markdown(
            f'<div class="score-pill">Score: {row["score"]:.5f}</div>',
            unsafe_allow_html=True
        )

    selected = False
    if checkbox_key is not None:
        selected = st.checkbox("Select", key=checkbox_key)

    st.markdown('</div>', unsafe_allow_html=True)
    return selected


def render_selectable_grid(items_df, prefix="pick"):
    selected_ids = []
    if items_df.empty:
        st.info("No items available.")
        return selected_ids

    n_cols = 3
    rows = [items_df.iloc[i:i+n_cols] for i in range(0, len(items_df), n_cols)]

    for r_idx, chunk in enumerate(rows):
        cols = st.columns(n_cols)
        for c_idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[c_idx]:
                checked = render_item_card(row, show_score=False, checkbox_key=f"{prefix}_{row['id']}_{r_idx}_{c_idx}")
                if checked:
                    selected_ids.append(str(row["id"]))
    return selected_ids


def render_grid(items_df, show_score=False):
    if items_df.empty:
        st.info("No items to display.")
        return

    n_cols = 3
    rows = [items_df.iloc[i:i+n_cols] for i in range(0, len(items_df), n_cols)]

    for chunk in rows:
        cols = st.columns(n_cols)
        for idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[idx]:
                render_item_card(row, show_score=show_score)

# ==================================================
# HEADER
# ==================================================
st.markdown('<div class="app-title">Barbie Style Super Squad</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Pick a few items you like, then get similar recommendations.</div>',
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("Options")

gender_options = sorted(df["gender"].unique())
season_options = sorted(df["season"].unique())
usage_options = sorted(df["usage"].unique())

client_gender = st.sidebar.selectbox("Gender", gender_options)
client_season = st.sidebar.selectbox("Season", season_options)
client_usage = st.sidebar.selectbox("Usage", usage_options)

num_pool = st.sidebar.slider("Items to browse", 12, 36, 18, step=6)
num_recs = st.sidebar.slider("Recommendations", 3, 12, 6)

# ==================================================
# STEP 1: CANDIDATE POOL
# ==================================================
pool = get_candidate_pool(df, client_gender, client_season, client_usage, n=num_pool)

st.markdown('<div class="section-title">Browse items</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small-note">Select at least 3 items so the model can learn your preferences.</div>',
    unsafe_allow_html=True
)

selected_ids = render_selectable_grid(pool, prefix="pool")

selected_rows = df[df["id"].astype(str).isin(selected_ids)].copy()

# ==================================================
# STEP 2: SHOW SELECTED
# ==================================================
if len(selected_rows) > 0:
    st.markdown('<div class="section-title">Selected items</div>', unsafe_allow_html=True)
    render_grid(selected_rows, show_score=False)

# ==================================================
# STEP 3: RECOMMENDATIONS
# ==================================================
if st.button("Get recommendations"):
    if len(selected_rows) < 3:
        st.warning("Please select at least 3 items.")
    else:
        recs = recommend_items(
            dataframe=df,
            client_gender=client_gender,
            client_season=client_season,
            client_usage=client_usage,
            selected_example_rows=selected_rows,
            top_n=num_recs,
            alpha=1.0
        )

        st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="small-note">Higher scores indicate stronger matches to both context and your selected preferences.</div>',
            unsafe_allow_html=True
        )
        render_grid(recs, show_score=True)

        article_pref, color_pref = build_preference_profile(selected_rows)

        st.markdown('<div class="section-title">Learned preferences</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.write("Article type")
            if article_pref:
                article_df = pd.DataFrame(
                    [{"articleType": k, "probability": v} for k, v in article_pref.items()]
                ).sort_values("probability", ascending=False)
                st.dataframe(article_df, use_container_width=True, hide_index=True)
            else:
                st.write("No data")

        with c2:
            st.write("Color")
            if color_pref:
                color_df = pd.DataFrame(
                    [{"baseColour": k, "probability": v} for k, v in color_pref.items()]
                ).sort_values("probability", ascending=False)
                st.dataframe(color_df, use_container_width=True, hide_index=True)
            else:
                st.write("No data")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    """
    <div class="small-note">
    Score used:
    <br><br>
    <code>P(articleType | gender, season, usage) × P(color | articleType, season) × user article preference × user color preference</code>
    </div>
    """,
    unsafe_allow_html=True
)