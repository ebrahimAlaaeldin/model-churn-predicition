import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from typing import List

def chi_square_test(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform Chi-square tests for categorical/binary features against a target variable.
    """
    results = []

    for feature in features:
        # Skip if feature or target has NaN
        if df[feature].isnull().any() or df[target].isnull().any():
            continue

        contingency_table = pd.crosstab(df[feature], df[target])

        # Chi-square test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Cramér's V
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        # Interpret strength
        if cramers_v <= 0.10:
            strength = "Very Weak"
        elif cramers_v <= 0.30:
            strength = "Weak"
        elif cramers_v <= 0.50:
            strength = "Moderate"
        else:
            strength = "Strong"

        reject_h0 = "Yes" if p_value <= alpha else "No"

        results.append({
            'Feature': feature,
            'Test': 'Chi-Square',
            'p-value': round(p_value, 4),
            'Reject_H0': reject_h0,
            'Cramer_V': round(cramers_v, 4),
            'Relationship_Strength': strength
        })

    return pd.DataFrame(results)

def t_test_analysis(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform independent t-tests for continuous features vs. binary target.
    """
    results = []

    churned = df[df[target] == 1]
    not_churned = df[df[target] == 0]

    for feature in features:
        # Perform Welch’s t-test (unequal variances)
        t_stat, p_value = ttest_ind(
            churned[feature],
            not_churned[feature],
            equal_var=False,
            nan_policy='omit'
        )

        # Hypothesis testing interpretation
        reject_h0 = "Yes" if p_value <= alpha else "No"

        results.append({
            'Feature': feature,
            'Test': 'T-Test',
            'p-value': round(p_value, 4),
            'Reject_H0': reject_h0,
            'Relationship_Strength': (
                "Significant" if p_value <= alpha else "Not Significant"
            )
        })

    return pd.DataFrame(results)
