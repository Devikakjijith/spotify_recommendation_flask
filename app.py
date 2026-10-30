import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from io import BytesIO
import base64
import time

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from train_model import train_and_save_model

app = Flask(__name__)

# --------------------------------------------------
# 1. LOAD PERSISTED MODEL & METADATA
# --------------------------------------------------

MODEL_DIR = "models"
MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, "spotify_model.pkl")
ROOT_MODEL_PATH = "model.pkl"

def get_or_load_model():
    """Load pre-trained and saved model bundle from disk.
    If no saved model exists, train and save it first."""
    if os.path.exists(MODEL_BUNDLE_PATH):
        load_path = MODEL_BUNDLE_PATH
    elif os.path.exists(ROOT_MODEL_PATH):
        load_path = ROOT_MODEL_PATH
    else:
        print("No saved model found on disk. Training and saving model now...")
        train_and_save_model(
            dataset_path="SpotifyFeatures.csv",
            models_dir=MODEL_DIR,
            root_model_path=ROOT_MODEL_PATH
        )
        load_path = MODEL_BUNDLE_PATH

    print(f"Loading pre-trained model bundle from '{load_path}'...")
    start_time = time.time()
    bundle = joblib.load(load_path)
    print(f"Model successfully loaded in {time.time() - start_time:.3f} seconds.")
    return bundle, load_path

# Load the saved model and artifacts
model_bundle, model_source_path = get_or_load_model()

scaler = model_bundle["scaler"]
pca_model = model_bundle["pca_model"]
kmeans = model_bundle.get("kmeans")

features = model_bundle["features"]
feature_medians = model_bundle["feature_medians"]
n_components = model_bundle["n_components"]
cumulative_variance = model_bundle["cumulative_variance"]
variance_retained = model_bundle["variance_retained"]
reconstruction_mse = model_bundle["reconstruction_mse"]
original_features = model_bundle["original_features"]
reduced_features = model_bundle["reduced_features"]
dimensionality_reduction = model_bundle["dimensionality_reduction"]
silhouette = model_bundle.get("silhouette")
reduced_data = model_bundle["reduced_data"]
songs = model_bundle["compact_songs"]
base_song_options = model_bundle["base_song_options"]
sample_table_html = model_bundle["sample_table_html"]
column_table_html = model_bundle["column_table_html"]
summary_table_html = model_bundle["summary_table_html"]
audio_sample = model_bundle["audio_sample"]
correlation_matrix = model_bundle["correlation_matrix"]
total_rows = model_bundle["total_rows"]
total_columns = model_bundle["total_columns"]
duplicate_count = model_bundle["duplicate_count"]
model_trained_at = model_bundle.get("trained_at", "Pre-trained")

# In-memory chart cache so charts are rendered once and served instantaneously
_CHART_CACHE = {}

# --------------------------------------------------
# 2. HELPER FUNCTION: CREATE CHARTS
# --------------------------------------------------

def make_chart():
    """Convert the current Matplotlib figure into a Base64 image."""
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    plt.close("all")
    return image_data


def get_chart(chart_type):
    """Generate and cache charts used by the EDA and PCA pages using loaded model artifacts."""
    if chart_type in _CHART_CACHE:
        return _CHART_CACHE[chart_type]

    plt.close("all")

    if chart_type == "distribution":
        plot_data = audio_sample
        plot_data.hist(figsize=(14, 9), bins=30, edgecolor="black")
        plt.suptitle("Distribution of Spotify Audio Features", fontsize=16)
        img = make_chart()
        _CHART_CACHE["distribution"] = img
        return img

    elif chart_type == "correlation":
        plt.figure(figsize=(11, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Between Audio Features")
        img = make_chart()
        _CHART_CACHE["correlation"] = img
        return img

    elif chart_type == "boxplot":
        plot_data = audio_sample
        if len(plot_data) > 5000:
            plot_data = plot_data.sample(5000, random_state=42)
        plt.figure(figsize=(13, 7))
        sns.boxplot(data=plot_data, orient="h")
        plt.title("Boxplot of Spotify Audio Features (Sample of 5,000 Tracks)")
        plt.xlabel("Feature Value")
        img = make_chart()
        _CHART_CACHE["boxplot"] = img
        return img

    elif chart_type == "popularity":
        popularity = songs["display_popularity"].dropna()
        if popularity.empty or (popularity == 0).all():
            return None
        if len(popularity) > 20000:
            popularity = popularity.sample(20000, random_state=42)
        plt.figure(figsize=(10, 5))
        plt.hist(popularity, bins=25, edgecolor="black")
        plt.title("Distribution of Song Popularity")
        plt.xlabel("Popularity")
        plt.ylabel("Number of Songs")
        img = make_chart()
        _CHART_CACHE["popularity"] = img
        return img

    elif chart_type == "scatter":
        if not {"danceability", "energy"}.issubset(audio_sample.columns):
            return None
        plot_data = audio_sample[["danceability", "energy"]]
        if len(plot_data) > 5000:
            plot_data = plot_data.sample(5000, random_state=42)
        plt.figure(figsize=(9, 6))
        plt.scatter(plot_data["danceability"], plot_data["energy"], alpha=0.4)
        plt.title("Danceability vs Energy")
        plt.xlabel("Danceability")
        plt.ylabel("Energy")
        img = make_chart()
        _CHART_CACHE["scatter"] = img
        return img

    elif chart_type == "pca":
        if reduced_data.shape[1] < 2:
            return None
        rng = np.random.default_rng(42)
        indices = np.arange(len(reduced_data))
        if len(indices) > 5000:
            indices = rng.choice(indices, size=5000, replace=False)
        plot_data = reduced_data[indices]

        plt.figure(figsize=(10, 6))
        plt.scatter(plot_data[:, 0], plot_data[:, 1], alpha=0.5)
        plt.title("Songs Projected onto the First Two PCA Components (from Saved Model)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        img = make_chart()
        _CHART_CACHE["pca"] = img
        return img

    elif chart_type == "variance":
        components = np.arange(1, len(cumulative_variance) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(components, cumulative_variance * 100, marker="o")
        plt.axhline(y=95, linestyle="--", label="95% variance threshold")
        plt.axvline(x=n_components, linestyle="--", label=f"Saved components: {n_components}")
        plt.title("Cumulative Explained Variance (Saved PCA Model)")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.legend()
        img = make_chart()
        _CHART_CACHE["variance"] = img
        return img

    return None


# --------------------------------------------------
# 3. PAGE 1: DATASET INFORMATION
# --------------------------------------------------

@app.route("/")
@app.route("/dataset")
def dataset_page():
    return render_template(
        "dataset.html",
        total_rows=total_rows,
        total_columns=total_columns,
        total_features=len(features),
        duplicates=duplicate_count,
        sample_table=sample_table_html,
        column_table=column_table_html,
        summary_table=summary_table_html
    )


# --------------------------------------------------
# 4. PAGE 2: EDA AND FEATURE ANALYSIS
# --------------------------------------------------

@app.route("/eda")
def eda_page():
    return render_template(
        "eda.html",
        distribution_chart=get_chart("distribution"),
        correlation_chart=get_chart("correlation"),
        boxplot_chart=get_chart("boxplot"),
        scatter_chart=get_chart("scatter"),
        popularity_chart=get_chart("popularity"),
        feature_summary=summary_table_html
    )


# --------------------------------------------------
# 5. PAGE 3: SONG RECOMMENDATIONS USING SAVED MODEL
# --------------------------------------------------

@app.route("/recommendations", methods=["GET", "POST"])
def recommend_page():
    selected_song = ""
    selected_number = 5
    rec_mode = "library"  # "library" or "custom"
    recommendations = None
    input_summary = None
    pca_projections = None

    # Default custom feature values based on feature medians
    custom_defaults = {
        "acousticness": round(float(feature_medians.get("acousticness", 0.35)), 2),
        "danceability": round(float(feature_medians.get("danceability", 0.55)), 2),
        "energy": round(float(feature_medians.get("energy", 0.57)), 2),
        "instrumentalness": round(float(feature_medians.get("instrumentalness", 0.15)), 2),
        "liveness": round(float(feature_medians.get("liveness", 0.20)), 2),
        "loudness": round(float(feature_medians.get("loudness", -9.5)), 1),
        "speechiness": round(float(feature_medians.get("speechiness", 0.12)), 2),
        "tempo": round(float(feature_medians.get("tempo", 118.0)), 1),
        "valence": round(float(feature_medians.get("valence", 0.45)), 2),
    }
    custom_values = dict(custom_defaults)

    song_options = list(base_song_options)

    if request.method == "POST":
        rec_mode = request.form.get("rec_mode", "library")

        try:
            selected_number = int(request.form.get("number", "5"))
        except (TypeError, ValueError):
            selected_number = 5
        selected_number = max(1, min(selected_number, 20))

        # MODE 1: Select song from library
        if rec_mode == "library":
            selected_song = request.form.get("song", "").strip()
            try:
                song_index = int(selected_song)
            except (TypeError, ValueError):
                song_index = -1

            if 0 <= song_index < len(songs):
                # Ensure the selected song is present in dropdown
                if not any(opt["value"] == str(song_index) for opt in song_options):
                    song_options.insert(1, {
                        "value": str(song_index),
                        "label": (
                            f"{songs.at[song_index, 'display_track']} — "
                            f"{songs.at[song_index, 'display_artist']} "
                            f"({songs.at[song_index, 'display_genre']})"
                        )
                    })

                target_track = songs.at[song_index, 'display_track']
                target_artist = songs.at[song_index, 'display_artist']
                target_genre = songs.at[song_index, 'display_genre']
                input_summary = f"Selected Song: {target_track} by {target_artist} ({target_genre})"

                # Get PCA coordinates directly from the saved model's reduced_data
                query_pca_vector = reduced_data[song_index].reshape(1, -1)

                pca_projections = [
                    {"component": f"PC{i + 1}", "value": f"{val:.3f}"}
                    for i, val in enumerate(query_pca_vector[0])
                ]

                # Compute cosine similarity across library
                similarity_scores = cosine_similarity(query_pca_vector, reduced_data)[0]

                # Rank by similarity (highest first), excluding the query song itself
                similar_indices = np.argsort(similarity_scores)[::-1]
                similar_indices = [i for i in similar_indices if i != song_index]
                top_indices = similar_indices[:selected_number]

                result = songs.iloc[top_indices][
                    ["display_track", "display_artist", "display_genre", "display_popularity"]
                ].copy()

                result["Similarity Score"] = [
                    f"{round(float(similarity_scores[i]) * 100, 1)}%"
                    for i in top_indices
                ]

                result = result.rename(
                    columns={
                        "display_track": "Song",
                        "display_artist": "Artist",
                        "display_genre": "Genre",
                        "display_popularity": "Popularity",
                        "Similarity Score": "Match Similarity"
                    }
                )

                result = result[["Song", "Artist", "Genre", "Match Similarity", "Popularity"]]
                recommendations = result.to_html(index=False, classes="data-table", border=0)

        # MODE 2: Custom audio features (direct inference using saved StandardScaler & PCA)
        elif rec_mode == "custom":
            for f in features:
                raw_val = request.form.get(f, str(custom_defaults.get(f, 0.5)))
                try:
                    custom_values[f] = float(raw_val)
                except (ValueError, TypeError):
                    custom_values[f] = custom_defaults.get(f, 0.5)

            input_df = pd.DataFrame([[custom_values[f] for f in features]], columns=features)

            # 1. Transform with saved StandardScaler
            scaled_vector = scaler.transform(input_df)

            # 2. Transform with saved PCA model
            pca_vector = pca_model.transform(scaled_vector)

            input_summary = (
                f"Custom Audio Profile (Tempo: {custom_values['tempo']} BPM, "
                f"Energy: {custom_values['energy']}, Danceability: {custom_values['danceability']})"
            )

            pca_projections = [
                {"component": f"PC{i + 1}", "value": f"{val:.3f}"}
                for i, val in enumerate(pca_vector[0])
            ]

            # 3. Compute cosine similarity with reduced library
            similarity_scores = cosine_similarity(pca_vector, reduced_data)[0]
            top_indices = np.argsort(similarity_scores)[::-1][:selected_number]

            result = songs.iloc[top_indices][
                ["display_track", "display_artist", "display_genre", "display_popularity"]
            ].copy()

            result["Similarity Score"] = [
                f"{round(float(similarity_scores[i]) * 100, 1)}%"
                for i in top_indices
            ]

            result = result.rename(
                columns={
                    "display_track": "Song",
                    "display_artist": "Artist",
                    "display_genre": "Genre",
                    "display_popularity": "Popularity",
                    "Similarity Score": "Match Similarity"
                }
            )

            result = result[["Song", "Artist", "Genre", "Match Similarity", "Popularity"]]
            recommendations = result.to_html(index=False, classes="data-table", border=0)

    return render_template(
        "recommendations.html",
        song_options=song_options,
        selected_song=selected_song,
        selected_number=selected_number,
        rec_mode=rec_mode,
        custom_values=custom_values,
        recommendations=recommendations,
        input_summary=input_summary,
        pca_projections=pca_projections,
        model_source=model_source_path,
        n_components=n_components
    )


# --------------------------------------------------
# 6. PAGE 4: PCA ANALYSIS
# --------------------------------------------------

@app.route("/pca")
def pca_page():
    component_table = pd.DataFrame({
        "Principal Component": [
            f"PC{i + 1}" for i in range(len(pca_model.explained_variance_ratio_))
        ],
        "Explained Variance (%)": [
            round(float(value * 100), 2) for value in pca_model.explained_variance_ratio_
        ]
    }).to_html(index=False, classes="data-table", border=0)

    return render_template(
        "pca.html",
        original_features=original_features,
        reduced_features=reduced_features,
        variance_retained=round(variance_retained, 2),
        reconstruction_mse=round(float(reconstruction_mse), 5),
        dimensionality_reduction=round(float(dimensionality_reduction), 2),
        silhouette_score=(
            round(silhouette, 4) if silhouette is not None else "N/A"
        ),
        pca_chart=get_chart("pca"),
        variance_chart=get_chart("variance"),
        component_table=component_table,
        model_source=model_source_path,
        model_trained_at=model_trained_at
    )


# --------------------------------------------------
# 7. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
