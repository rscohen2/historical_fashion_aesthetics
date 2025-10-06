from classifier import classifier_preprocess
from classifier_preprocess import *
if __name__ == "__main__":

    import pyarrow.parquet as pq

    # fashion_pq = pd.read_parquet('fashion_mentions.parquet')
    # character_pq = pd.read_parquet('characters.parquet')
    fashion_pq = pq.ParquetFile("fashion_mentions.parquet")
    # characters_pq = pq.ParquetFile("characters.parquet")

    import pandas as pd

    characters_pq = pq.ParquetFile("characters.parquet")
    all_character_chunks = []

    for i in range(characters_pq.num_row_groups):
        df = characters_pq.read_row_group(i).to_pandas()
        # Only keep relevant columns
        df = df[['character_id', 'gender']]
        all_character_chunks.append(df)

    # Combine and deduplicate by character_id (keeping first)
    character_df = pd.concat(all_character_chunks, ignore_index=True)
    character_df = character_df.drop_duplicates(subset='character_id')

    # Save slimmed, deduplicated data
    character_df.to_parquet("slim_characters.parquet", index=False)

    slim_characters = pq.ParquetFile("slim_characters.parquet")

    # Save slimmed, deduplicated data
    character_df.to_parquet("slim_characters.parquet", index=False)

    # Load slim_characters once outside the loop
    slim_characters = pq.ParquetFile("slim_characters.parquet")

    # Read all row groups from slim_characters and concatenate into one DataFrame
    slim_df = pd.concat([
        slim_characters.read_row_group(j).to_pandas()
        for j in range(slim_characters.num_row_groups)
    ])

    merged_chunks = []  # If you want to accumulate merged data for all fashion chunks

    for i in range(fashion_pq.num_row_groups):
        fashion_df = fashion_pq.read_row_group(i).to_pandas()

        # Merge fashion_df with slim character data
        fashion_df['character_id'] = fashion_df['character_id'].astype(str)
        slim_df['character_id'] = slim_df['character_id'].astype(str)

        merged_df = fashion_df.merge(slim_df, on='character_id', how='left')
        merged_df = merged_df[merged_df['gender'] != 'they/them/their']

        merged_df = merged_df.dropna()

        # Filter out unwanted genders after merge (or before, your choice)
        # merged_df = merged_df[merged_df['gender'] != 'they/them/theirs']

        merged_chunks.append(merged_df)

    # Combine all merged chunks (if needed)
    full_merged_df = pd.concat(merged_chunks, ignore_index=True)

    # Now run the full pipeline on the combined data
    df_transformed, mlb, bow_matrix = full_pipeline(
        full_merged_df,
        min_frequency=1  # Lower for small example
    )


    # Prepare data for classification
    if bow_matrix is not None:
        # Using sparse matrix
        X = bow_matrix
        y = df_transformed['gender']
        print(f"\nSparse feature matrix (X) shape: {X.shape}")
    else:
        # Using dense DataFrame
        feature_cols = mlb.classes_
        X = df_transformed[feature_cols]
        y = df_transformed['gender']
        print(f"\nFeature matrix (X) shape: {X.shape}")

    print(f"Target variable (y) shape: {y.shape}")
    # Run classification
    model, feature_importance = full_classification_pipeline(
        df_transformed,
        mlb,
        bow_matrix,
        model_type='logistic',  # or 'random_forest'
        test_size=0.2
    )