"""
Training and Model Persistence Script for Spotify Unsupervised Learning (SoundSpace)
-----------------------------------------------------------------------------------
This script:
1. Loads and preprocesses SpotifyFeatures.csv
2. Fits StandardScaler and PCA (retaining >= 95% variance)
3. Evaluates KMeans clustering and silhouette score
4. Precomputes reduced feature representations & metadata
5. Saves the trained models and artifacts to disk:
   - models/scaler.pkl
   - models/pca_model.pkl
   - models/kmeans_model.pkl
   - models/spotify_model.pkl (full bundle)
   - model.pkl (root bundle)
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans


def train_and_save_model(
    dataset_path="SpotifyFeatures.csv",
    models_dir="models",
    root_model_path="model.pkl"
):
    start_time = time.time()
    print("=" * 60)
    print("  Spotify Music Unsupervised Model Training & Persistence")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    os.makedirs(models_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. LOAD & CLEAN DATASET
    # --------------------------------------------------
    print(f"\n[1/5] Loading dataset from '{dataset_path}'...")
    raw_df = pd.read_csv(dataset_path)
    total_raw_rows = len(raw_df)
    total_columns = len(raw_df.columns)

    duplicate_count = int(raw_df.duplicated().sum())
    df = raw_df.drop_duplicates().reset_index(drop=True)

    audio_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence"
    ]

    features = [col for col in audio_features if col in df.columns]
    if len(features) < 2:
        raise ValueError(
            "The dataset must contain at least two of these audio features: "
            + ", ".join(audio_features)
        )

    audio_data = df[features].copy()
    for col in features:
        audio_data[col] = pd.to_numeric(audio_data[col], errors="coerce")

    audio_data = audio_data.replace([np.inf, -np.inf], np.nan)
    valid_rows = audio_data.notna().any(axis=1)

    df = df.loc[valid_rows].reset_index(drop=True)
    audio_data = audio_data.loc[valid_rows].reset_index(drop=True)

    non_constant_features = [
        col for col in features
        if audio_data[col].nunique(dropna=True) > 1
    ]
    audio_data = audio_data[non_constant_features]
    features = non_constant_features

    feature_medians = audio_data.median().to_dict()
    audio_data = audio_data.fillna(feature_medians)

    print(f"      Valid tracks: {len(df):,} | Features: {len(features)} | Duplicates removed: {duplicate_count:,}")

    # --------------------------------------------------
    # 2. TRAIN STANDARDIZATION & PCA MODEL
    # --------------------------------------------------
    print("\n[2/5] Training StandardScaler and PCA model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(audio_data)

    pca_all = PCA()
    pca_all.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_variance >= 0.95) + 1)

    pca_model = PCA(n_components=n_components)
    reduced_data = pca_model.fit_transform(X_scaled)

    reconstructed_data = pca_model.inverse_transform(reduced_data)
    reconstruction_mse = float(mean_squared_error(X_scaled, reconstructed_data))
    variance_retained = float(np.sum(pca_model.explained_variance_ratio_) * 100)

    original_features = len(features)
    reduced_features = n_components
    dimensionality_reduction = float(((original_features - reduced_features) / original_features) * 100)

    print(f"      Optimal PCA components (>=95% variance): {n_components}")
    print(f"      Variance retained: {variance_retained:.2f}% | MSE: {reconstruction_mse:.5f}")
    print(f"      Dimensionality reduction: {dimensionality_reduction:.2f}%")

    # --------------------------------------------------
    # 3. FIT KMEANS & SILHOUETTE EVALUATION
    # --------------------------------------------------
    print("\n[3/5] Evaluating KMeans clustering on reduced latent space...")
    cluster_count = min(3, max(2, len(reduced_data) - 1))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_data)

    silhouette = None
    try:
        if len(np.unique(cluster_labels)) > 1:
            silhouette = float(
                silhouette_score(
                    reduced_data,
                    cluster_labels,
                    sample_size=min(5000, len(reduced_data)),
                    random_state=42
                )
            )
            print(f"      KMeans silhouette score: {silhouette:.4f}")
    except (ValueError, FloatingPointError) as e:
        print(f"      Silhouette score calculation skipped: {e}")

    # --------------------------------------------------
    # 4. PREPARE TRACK METADATA & UI DATA
    # --------------------------------------------------
    print("\n[4/5] Precomputing track metadata and dropdown options...")
    songs = df.reset_index(drop=True).copy()

    track_col = next((c for c in ["track_name", "track", "name"] if c in songs.columns), None)
    artist_col = next((c for c in ["artist_name", "artist", "artists"] if c in songs.columns), None)
    genre_col = next((c for c in ["genre", "track_genre"] if c in songs.columns), None)
    pop_col = next((c for c in ["popularity", "track_popularity"] if c in songs.columns), None)

    if track_col is None:
        songs["track_name"] = [f"Song {i + 1}" for i in range(len(songs))]
        track_col = "track_name"

    songs["display_track"] = songs[track_col].fillna("Unknown Song").astype(str)
    songs["display_artist"] = songs[artist_col].fillna("Unknown Artist").astype(str) if artist_col else "Unknown Artist"
    songs["display_genre"] = songs[genre_col].fillna("Unknown").astype(str) if genre_col else "Unknown"
    songs["display_popularity"] = (
        pd.to_numeric(songs[pop_col], errors="coerce").fillna(0).astype(int)
        if pop_col else 0
    )

    # Precompute top curated songs for dropdown
    genre_top_tracks = (
        songs.sort_values(by=["display_popularity", "display_track"], ascending=[False, True])
        .groupby("display_genre", as_index=False)
        .head(30)
    )
    overall_top_tracks = (
        songs.sort_values(by=["display_popularity", "display_track"], ascending=[False, True])
        .head(400)
    )
    curated_indices = (
        pd.concat([genre_top_tracks, overall_top_tracks])
        .drop_duplicates(subset=["display_track", "display_artist"])
        .index.tolist()
    )

    curated_selection = songs.loc[curated_indices].copy()
    curated_selection["orig_index"] = curated_selection.index
    curated_selection = curated_selection.sort_values(
        by=["display_track", "display_artist"],
        key=lambda col: col.str.lower()
    )

    base_song_options = [{"value": "", "label": "-- Select a song --"}]
    for orig_idx, track, artist, genre in zip(
        curated_selection["orig_index"],
        curated_selection["display_track"],
        curated_selection["display_artist"],
        curated_selection["display_genre"]
    ):
        base_song_options.append({
            "value": str(orig_idx),
            "label": f"{track} — {artist} ({genre})"
        })

    # Precompute static tables
    sample_table_html = songs.head(10)[
        ["display_track", "display_artist", "display_genre"]
    ].rename(
        columns={"display_track": "Track", "display_artist": "Artist", "display_genre": "Genre"}
    ).to_html(index=False, classes="data-table", border=0)

    column_table_html = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values": [int(df[col].isna().sum()) for col in df.columns]
    }).to_html(index=False, classes="data-table", border=0)

    summary_table_html = audio_data.describe().round(2).to_html(classes="data-table", border=0)

    # Compact track representation (for fast UI recommendations)
    compact_songs = songs[["display_track", "display_artist", "display_genre", "display_popularity"]].copy()

    # Audio sample for instant chart generation in EDA
    audio_sample = audio_data.sample(min(20000, len(audio_data)), random_state=42)
    correlation_matrix = audio_data.corr()

    # --------------------------------------------------
    # 5. SAVE MODELS TO DISK
    # --------------------------------------------------
    print("\n[5/5] Saving trained models and artifacts to disk...")

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    pca_path = os.path.join(models_dir, "pca_model.pkl")
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
    bundle_path = os.path.join(models_dir, "spotify_model.pkl")

    # 1. Save individual model files
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca_model, pca_path)
    joblib.dump(kmeans, kmeans_path)

    # 2. Save complete bundle for the web interface
    bundle = {
        "scaler": scaler,
        "pca_model": pca_model,
        "kmeans": kmeans,
        "features": features,
        "feature_medians": feature_medians,
        "n_components": n_components,
        "cumulative_variance": cumulative_variance,
        "variance_retained": variance_retained,
        "reconstruction_mse": reconstruction_mse,
        "original_features": original_features,
        "reduced_features": reduced_features,
        "dimensionality_reduction": dimensionality_reduction,
        "silhouette": silhouette,
        "reduced_data": reduced_data,
        "compact_songs": compact_songs,
        "base_song_options": base_song_options,
        "sample_table_html": sample_table_html,
        "column_table_html": column_table_html,
        "summary_table_html": summary_table_html,
        "audio_sample": audio_sample,
        "correlation_matrix": correlation_matrix,
        "total_rows": len(df),
        "total_columns": total_columns,
        "duplicate_count": duplicate_count,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(bundle, bundle_path, compress=3)
    joblib.dump(bundle, root_model_path, compress=3)

    print(f"      -> Saved StandardScaler to '{scaler_path}' ({os.path.getsize(scaler_path):,} bytes)")
    print(f"      -> Saved PCA model to       '{pca_path}' ({os.path.getsize(pca_path):,} bytes)")
    print(f"      -> Saved KMeans model to    '{kmeans_path}' ({os.path.getsize(kmeans_path):,} bytes)")
    print(f"      -> Saved Model Bundle to    '{bundle_path}' ({os.path.getsize(bundle_path):,} bytes)")
    print(f"      -> Saved Root Model to      '{root_model_path}' ({os.path.getsize(root_model_path):,} bytes)")

    elapsed = time.time() - start_time
    print(f"\nAll models successfully trained and saved in {elapsed:.2f} seconds!")
    print("=" * 60)

    return bundle


if __name__ == "__main__":
    train_and_save_model()

