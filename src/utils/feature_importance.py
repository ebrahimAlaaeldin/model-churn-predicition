from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def get_X_y(df, target_col='Churn'):
    """
    Splits a preprocessed DataFrame into features (X) and target (y).

    Parameters
    ----------
    df : pandas.DataFrame
        The processed DataFrame.
    target_col : str
        Name of the target column.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    """

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y




def get_top_feature_importances(X, y, top_n=10, random_state=42):
    """
    Train a Random Forest model and return the top N most important features.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    top_n : int
        Number of top features to return.
    random_state : int
        Random seed.

    Returns
    -------
    important_features : pandas.Series
        Top N feature importances sorted in descending order.
    model : RandomForestClassifier
        Trained model (useful for further evaluation if needed).
    """
    
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    important_features = importances.sort_values(ascending=False).head(top_n)

    print(f"\nTop {top_n} Important Features:")
    print(important_features)

    return important_features, model


import matplotlib.pyplot as plt

def plot_feature_importances(feature_series, title="Top Important Features"):
    """
    Plot the feature importances as a horizontal bar chart.

    Parameters
    ----------
    feature_series : pandas.Series
        A Series where index = feature names, values = importance scores.
    title : str
        Plot title.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 5))
    feature_series.sort_values().plot(kind='barh', color='skyblue')

    plt.title(title)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


