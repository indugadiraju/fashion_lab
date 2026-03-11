import pandas as pd
import numpy as np
from collections import Counter
from IPython.display import display, HTML

CSV_FILE = "fashion.csv"

df_raw = pd.read_csv(CSV_FILE)
print("Raw shape:", df_raw.shape)
print("Raw columns:")
print(df_raw.columns.tolist())

df_raw.columns = [str(col).strip() for col in df_raw.columns]

df_raw = df_raw.dropna(axis=1, how="all").copy()

bad_cols = [col for col in df_raw.columns if col.lower().startswith("unnamed")]
if bad_cols:
    df_raw = df_raw.drop(columns=bad_cols)

print("\nColumns after basic cleanup:")
print(df_raw.columns.tolist())

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

df = keep_top_k(df, "gender", 5)
df = keep_top_k(df, "season", 4)
df = keep_top_k(df, "usage", 10)
df = keep_top_k(df, "articleType", 20)
df = keep_top_k(df, "baseColour", 15)

df = df.reset_index(drop=True)

def show_items(items_df, n=12, title="Items"):
    items_df = items_df.head(n).copy()

    html = f"<h2>{title}</h2>"
    html += "<div style='display:flex; flex-wrap:wrap; gap:16px;'>"

    for _, row in items_df.iterrows():
        image_html = ""
        if pd.notna(row.get("image_url", np.nan)):
            image_html = f"""
            <img src="{row['image_url']}"
                 style="width:180px; height:220px; object-fit:cover;
                        border-radius:8px; border:1px solid #ddd;">
            """
        else:
            image_html = """
            <div style="width:180px; height:220px; display:flex;
                        align-items:center; justify-content:center;
                        background:#f2f2f2; border:1px solid #ddd;
                        border-radius:8px;">
                No Image
            </div>
            """

        html += f"""
        <div style="width:200px; padding:10px; border:1px solid #ccc;
                    border-radius:10px; background:white;">
            {image_html}
            <div style="margin-top:8px; font-size:14px;">
                <b>{row['productDisplayName']}</b><br>
                <b>ID:</b> {row['id']}<br>
                <b>Gender:</b> {row['gender']}<br>
                <b>Season:</b> {row['season']}<br>
                <b>Usage:</b> {row['usage']}<br>
                <b>Type:</b> {row['articleType']}<br>
                <b>Color:</b> {row['baseColour']}<br>
            </div>
        </div>
        """

    html += "</div>"
    display(HTML(html))


def conditional_prob(dataframe, target_col, target_val, given_cols, given_vals, alpha=1.0):
    """
    Estimate:
    P(target_col = target_val | given_cols = given_vals) using frequency counts and Laplace smoothing.
    """
    filtered = dataframe.copy()

    for col, val in zip(given_cols, given_vals):
        filtered = filtered[filtered[col] == val]

    total_count = len(filtered)
    num_categories = dataframe[target_col].nunique()
    match_count = len(filtered[filtered[target_col] == target_val])
 
    prob = (match_count + alpha) / (total_count + alpha * num_categories)
    return prob

def pmf_table(dataframe, col):
    out = dataframe[col].value_counts(normalize=True).reset_index()
    out.columns = [col, "probability"]
    return out

def conditional_pmf_article(dataframe, gender, season, usage, alpha=1.0):
    article_types = sorted(dataframe["articleType"].unique())
    rows = []

    for article in article_types:
        p = conditional_prob(
            dataframe=dataframe,
            target_col="articleType",
            target_val=article,
            given_cols=["gender", "season", "usage"],
            given_vals=[gender, season, usage],
            alpha=alpha
        )
        rows.append((article, p))

    result = pd.DataFrame(rows, columns=["articleType", "probability"])
    return result.sort_values("probability", ascending=False).reset_index(drop=True)

def build_preference_profile(selected_rows):
    article_counts = Counter(selected_rows["articleType"])
    color_counts = Counter(selected_rows["baseColour"])

    total_articles = sum(article_counts.values())
    total_colors = sum(color_counts.values())

    article_pref = {k: v / total_articles for k, v in article_counts.items()}
    color_pref = {k: v / total_colors for k, v in color_counts.items()}

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

    same_gender = candidate_df[candidate_df["gender"] == client_gender].copy()
    if len(same_gender) > 0:
        candidate_df = same_gender

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


def run_bayesian_barbie(dataframe):
    print("===================================")
    print("Welcome Barbie's Style Super Squad :)")
    print("===================================\n")

    print("Available genders:")
    print(sorted(dataframe["gender"].unique()))
    client_gender = input("\nEnter gender exactly as shown: ").strip()

    print("\nAvailable seasons:")
    print(sorted(dataframe["season"].unique()))
    client_season = input("\nEnter season exactly as shown: ").strip()

    print("\nAvailable usages:")
    print(sorted(dataframe["usage"].unique()))
    client_usage = input("\nEnter usage exactly as shown: ").strip()

    pool = get_candidate_pool(dataframe, client_gender, client_season, client_usage, n=30)

    print("\nHere are some example clothes to choose from:")
    show_items(pool, n=min(15, len(pool)), title="Choose 3 example outfits you like")

    print("\nType the IDs of 3 items you like from the images above.")
    id1 = input("First liked item ID: ").strip()
    id2 = input("Second liked item ID: ").strip()
    id3 = input("Third liked item ID: ").strip()

    selected_rows = dataframe[dataframe["id"].astype(str).isin([id1, id2, id3])].copy()

    # fallback if IDs are bad
    if len(selected_rows) == 0:
        print("\nNo valid IDs found. Using the first 3 items from the pool.")
        selected_rows = pool.head(3).copy()
    elif len(selected_rows) < 3:
        print("\nFewer than 3 valid IDs found. Filling the rest from the pool.")
        already = set(selected_rows["id"].astype(str).tolist())
        filler = pool[~pool["id"].astype(str).isin(already)].head(3 - len(selected_rows))
        selected_rows = pd.concat([selected_rows, filler], ignore_index=True)

    show_items(selected_rows, n=3, title="Your selected example outfits")

    recs = recommend_items(
        dataframe=dataframe,
        client_gender=client_gender,
        client_season=client_season,
        client_usage=client_usage,
        selected_example_rows=selected_rows,
        top_n=6,
        alpha=1.0
    )

    show_items(recs, n=min(6, len(recs)), title="Barbie's Style Super Squad Recommendations")

    return recs, selected_rows, pool

recommendations, selected_examples, candidate_pool = run_bayesian_barbie(df)