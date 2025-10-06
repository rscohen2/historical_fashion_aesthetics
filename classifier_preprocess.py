import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter




def prepare_fashion_data(fashion_df, character_df):
    """
    Merge fashion mentions with character gender data.

    Parameters:
    fashion_df: DataFrame with columns ['sentence', 'term', 'adjectives', 'character_id']
                fashion_term and adjective can be lists/arrays or single values
    character_df: DataFrame with columns ['character_id', 'gender']

    Returns:
    Merged DataFrame with gender information added to fashion mentions
    """

    # Create copies to avoid modifying originals
    fashion_df = fashion_df.copy()
    character_df = character_df.copy()

    # Convert character_id to string in both dataframes to ensure consistency
    fashion_df['character_id'] = fashion_df['character_id'].astype(str)
    character_df['character_id'] = character_df['character_id'].astype(str)

    # Merge fashion data with character gender
    merged_df = fashion_df.merge(character_df, on='character_id', how='left')

    # Check for any unmatched records
    missing_gender = merged_df['gender'].isna().sum()
    if missing_gender > 0:
        print(f"Warning: {missing_gender} fashion mentions have no matching character gender")
        print("These rows will be dropped.")
        merged_df = merged_df.dropna(subset=['gender'])

    print(f"Successfully merged {len(merged_df)} fashion mentions with character data")

    return merged_df


def normalize_to_list(value):
    """Convert various formats to a list."""
    if isinstance(value, (list, np.ndarray)):
        return [str(v) for v in value if pd.notna(v)]
    elif pd.isna(value):
        return []
    else:
        return [str(value)]


def create_bow_features_efficient(df, min_frequency=2):
    """
    Convert fashion terms and adjectives to bag-of-words features efficiently.
    Handles multiple terms/adjectives per row without exploding.

    Parameters:
    df: DataFrame with columns ['sentence', 'fashion_term', 'adjective', 'character_id', 'gender']
    min_frequency: Minimum number of occurrences to include a feature (reduces memory)

    Returns:
    DataFrame with binary features for each term/adjective + original columns
    """

    # Create a copy
    df_bow = df.copy()

    # Normalize columns to lists
    print("Normalizing terms and adjectives to lists...")
    df_bow['fashion_term_list'] = df_bow['term'].apply(normalize_to_list)
    df_bow['adjective_list'] = df_bow['adjectives'].apply(normalize_to_list)

    # Count frequencies to filter rare terms (saves memory)
    print("Counting term frequencies...")
    all_terms = []
    all_adjs = []
    for terms in df_bow['fashion_term_list']:
        all_terms.extend([f'term_{t}' for t in terms])
    for adjs in df_bow['adjective_list']:
        all_adjs.extend([f'adj_{a}' for a in adjs])

    term_counts = Counter(all_terms)
    adj_counts = Counter(all_adjs)

    # Keep only frequent features
    frequent_features = set(
        [k for k, v in term_counts.items() if v >= min_frequency] +
        [k for k, v in adj_counts.items() if v >= min_frequency]
    )

    print(f"Total unique features before filtering: {len(term_counts) + len(adj_counts)}")
    print(f"Features after filtering (min_frequency={min_frequency}): {len(frequent_features)}")

    # Create combined feature lists with prefixes, filtered
    df_bow['all_features'] = df_bow.apply(
        lambda row: [f'term_{t}' for t in row['fashion_term_list']] +
                    [f'adj_{a}' for a in row['adjective_list']],
        axis=1
    )

    # Filter to only frequent features
    df_bow['all_features'] = df_bow['all_features'].apply(
        lambda features: [f for f in features if f in frequent_features]
    )

    # Use MultiLabelBinarizer to create binary features
    print("Creating binary feature matrix...")
    mlb = MultiLabelBinarizer(sparse_output=True)  # Use sparse matrix to save memory
    bow_matrix = mlb.fit_transform(df_bow['all_features'])

    print(f"Feature matrix shape: {bow_matrix.shape}")
    print(f"Memory usage (sparse): ~{bow_matrix.data.nbytes / 1024 ** 2:.2f} MB")

    # Convert sparse matrix to dense DataFrame (only if small enough)
    if bow_matrix.shape[1] < 1000:  # If less than 1000 features, safe to convert
        bow_df = pd.DataFrame(bow_matrix.toarray(), columns=mlb.classes_, index=df_bow.index)
        result_df = pd.concat([
            df_bow[['sentence', 'character_id', 'gender']],
            bow_df
        ], axis=1)
        return result_df, mlb, None
    else:
        # Keep as sparse matrix
        print("Keeping as sparse matrix due to large feature count")
        result_df = df_bow[['sentence', 'character_id', 'gender']].copy()
        return result_df, mlb, bow_matrix


def full_pipeline(merged_df, min_frequency=2):
    """
    Complete pipeline: merge data and create bag-of-words features.

    Parameters:
    fashion_df: DataFrame with fashion mentions

    character_df: DataFrame with character gender
    min_frequency: Minimum occurrences to include a feature

    Returns:
    Transformed DataFrame (or metadata dict if sparse), MultiLabelBinarizer, optional sparse matrix
    """

    # # Step 1: Merge the data
    # print("Step 1: Merging data...")
    # merged_df = prepare_fashion_data(fashion_df, character_df)
    # # filter they/them/their out for now
    # merged_df = merged_df[merged_df['gender'] != 'they/them/their']

    # Step 2: Create bag-of-words features
    print("\nStep 2: Creating bag-of-words features...")
    result = create_bow_features_efficient(merged_df, min_frequency=min_frequency)

    if len(result) == 3:
        transformed_df, mlb, bow_matrix = result
        if bow_matrix is not None:
            print(f"\nReturning sparse matrix format")
            print(f"Use bow_matrix for features, transformed_df for metadata")
        else:
            print(f"\nFinal dataset shape: {transformed_df.shape}")
    else:
        transformed_df, mlb = result
        bow_matrix = None
        print(f"\nFinal dataset shape: {transformed_df.shape}")

    print(f"Number of unique features: {len(mlb.classes_)}")
    print(f"\nGender distribution:")
    print(merged_df['gender'].value_counts())

    return transformed_df, mlb, bow_matrix


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_classification_data(df_transformed, mlb, bow_matrix=None):
    """
    Prepare features and target for classification.

    Parameters:
    df_transformed: DataFrame with metadata (gender, character_id, etc.)
    mlb: MultiLabelBinarizer with feature names
    bow_matrix: Optional sparse matrix if using sparse format

    Returns:
    X, y, feature_names
    """

    # Get features
    if bow_matrix is not None:
        # Using sparse matrix
        X = bow_matrix
        feature_names = mlb.classes_
    else:
        # Using dense DataFrame
        feature_names = mlb.classes_
        X = df_transformed[feature_names]

    # Get target variable
    y = df_transformed['gender']

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X, y, feature_names


def train_gender_classifier(X, y, test_size=0.2, random_state=42):
    """
    Train a classifier to predict gender from fashion terms.

    Parameters:
    X: Feature matrix (can be sparse or dense)
    y: Target variable (gender)
    test_size: Proportion of data for testing
    random_state: Random seed for reproducibility

    Returns:
    model, X_train, X_test, y_train, y_test
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Test set size: {X_test.shape[0]}")

    # print(f"Test set size: {len(X_test)}")
    print(f"\nTraining set gender distribution:\n{y_train.value_counts()}")

    # Train Logistic Regression (works well with BOW features)
    print("\nTraining Logistic Regression classifier...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'  # Handle imbalanced classes
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"\nTraining Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

    return model, X_train, X_test, y_train, y_test


def evaluate_classifier(model, X_test, y_test, feature_names=None):
    """
    Detailed evaluation of the classifier.
    """

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return y_pred, y_pred_proba


def get_most_important_features(model, feature_names, top_n=20):
    """
    Get the most important features for each gender.
    Works with Logistic Regression coefficients.
    """

    if not hasattr(model, 'coef_'):
        print("Model doesn't have feature importance")
        return

    # Get coefficients
    coef = model.coef_[0] if len(model.coef_.shape) == 2 and model.coef_.shape[0] == 1 else model.coef_

    if len(coef.shape) == 1:
        # Binary classification
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef
        })
        feature_importance['abs_coef'] = np.abs(feature_importance['coefficient'])
        feature_importance = feature_importance.sort_values('abs_coef', ascending=False)

        print("\n" + "=" * 50)
        print(f"TOP {top_n} MOST PREDICTIVE FEATURES")
        print("=" * 50)
        print(f"Positive coefficients → {model.classes_[1]}")
        print(f"Negative coefficients → {model.classes_[0]}")
        print("\n", feature_importance.head(top_n))

        # Plot
        top_features = feature_importance.head(top_n)
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient (← Female | Male →)' if model.classes_[0] == 'female' else 'Coefficient')
        plt.title(f'Top {top_n} Most Predictive Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:
        # Multi-class
        for i, gender in enumerate(model.classes_):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coef[i]
            })
            feature_importance = feature_importance.sort_values('coefficient', ascending=False)
            print(f"\n\nTop features for {gender}:")
            print(feature_importance.head(top_n))

    return feature_importance


def train_random_forest(X, y, test_size=0.2, random_state=42):
    """
    Alternative: Train Random Forest classifier.
    Often works better for complex patterns.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"\nTraining Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

    return model, X_train, X_test, y_train, y_test


# Complete pipeline
def full_classification_pipeline(df_transformed, mlb, bow_matrix=None,
                                 model_type='logistic', test_size=0.2):
    """
    Complete pipeline from BOW data to trained classifier.

    Parameters:
    df_transformed: DataFrame with gender and other metadata
    mlb: MultiLabelBinarizer with feature names
    bow_matrix: Optional sparse matrix
    model_type: 'logistic' or 'random_forest'
    test_size: Proportion for test set
    """

    # Prepare data
    X, y, feature_names = prepare_classification_data(df_transformed, mlb, bow_matrix)

    # Train model
    if model_type == 'logistic':
        model, X_train, X_test, y_train, y_test = train_gender_classifier(
            X, y, test_size=test_size
        )
    else:
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, test_size=test_size
        )

    # Evaluate
    y_pred, y_pred_proba = evaluate_classifier(model, X_test, y_test, feature_names)

    # Feature importance
    if model_type == 'logistic':
        feature_importance = get_most_important_features(model, feature_names, top_n=20)
    else:
        # Random Forest feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n\nTop 20 Most Important Features (Random Forest):")
        print(feature_importance.head(20))

    return model, feature_importance


# Example usage
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

    for i in range(fashion_pq.num_row_groups):
        fashion_df = fashion_pq.read_row_group(i).to_pandas()

        # Filter fashion_df before merge
        fashion_df = fashion_df[fashion_df['gender'] != 'they/them/theirs']

        # Now merge with slim character data in a streaming-safe way
        # Step 1: Load slim_characters into a DataFrame (should be small enough now)
        slim_df = pd.concat([
            slim_characters.read_row_group(j).to_pandas()
            for j in range(slim_characters.num_row_groups)
        ])

        # Step 2: Merge
        merged_df = fashion_df.merge(slim_df, on='character_id', how='left')


    df_transformed, mlb, bow_matrix = full_pipeline(
       merged_df,
        min_frequency=1  # Lower for small example
    )

    print("\n" + "=" * 50)

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

    # Save the model
    import joblib

    joblib.dump(model, 'gender_classifier.pkl')
    joblib.dump(mlb, 'multilabel_binarizer.pkl')
    print("\nModel saved!")

    # To use the model later:
    # model = joblib.load('gender_classifier.pkl')
    # mlb = joblib.load('multilabel_binarizer.pkl')

