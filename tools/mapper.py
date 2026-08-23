from typing import Dict, List, Optional, Union
import pandas as pd

# Default SNOMED-CT equivalent class mappings
DEFAULT_EQUIVALENT_MAP: Dict[str, str] = {
    "713427006": "59118001",
    "284470004": "63593006",
    "427172004": "17338001",
}


def process_class_labels(
    labels: Union[pd.DataFrame, pd.Series, List[str]],
    df_unscored: pd.DataFrame,
    equivalent_map: Optional[Dict[str, str]] = DEFAULT_EQUIVALENT_MAP,
    undefined_label: str = "undefined class",
) -> pd.DataFrame:
    """Vectorized label transformation and unscored class replacement.

    Replaces equivalent class codes and flags unscored targets efficiently.
    """
    # 1. Normalize input to DataFrame of string types
    df_labels = pd.DataFrame(labels).astype(str)

    # 2. Vectorized hash-map replacement for equivalent classes
    if equivalent_map:
        df_labels.replace(equivalent_map, inplace=True)

    # 3. Vectorized mask replacement for unscored class codes
    unscored_codes = set(df_unscored.iloc[:, 1].astype(str).unique())
    df_labels = df_labels.mask(df_labels.isin(unscored_codes), undefined_label)

    return df_labels


# --- Verification Example ---
if __name__ == "__main__":
    sample_labels = [["713427006"], ["284470004"], ["999999999"], ["123456789"]]
    sample_unscored = pd.DataFrame({"idx": [0, 1], "code": ["999999999", "888888888"]})

    result = process_class_labels(sample_labels, sample_unscored)
    print(result)
