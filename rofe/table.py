import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def table_encoder(table: pd.DataFrame, categorical: list[str], one_hot: bool) -> ColumnTransformer:
    numeric = [column for column in table if column not in categorical]
    if one_hot:
        # dense: numbers median-filled and standardized, categories one-hot
        steps = [("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                  numeric),
                 ("categorical", make_pipeline(
                     SimpleImputer(strategy="constant", fill_value="missing"),
                     OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical)]
    else:
        # PFNs: numbers as they are with NaN kept, categories as ordinal codes, unseen -1
        steps = [("numeric", "passthrough", numeric),
                 ("categorical", make_pipeline(
                     SimpleImputer(strategy="constant", fill_value="missing"),
                     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                  categorical)]
    return ColumnTransformer(steps)


def encoded(table: pd.DataFrame, categorical: list[str], fit_rows: np.ndarray,
            one_hot: bool) -> np.ndarray:
    frame = table.copy()
    for column in frame:
        if column in categorical:
            frame[column] = frame[column].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
        else:
            frame[column] = frame[column].map(
                lambda v: np.nan if pd.isna(v) else float(v)).astype(np.float64)
    encoder = table_encoder(frame, categorical, one_hot).fit(frame.iloc[fit_rows])
    # (N, encoded columns), statistics of fit_rows; numbers first, then categories
    return np.asarray(encoder.transform(frame), dtype=np.float64)


def categorical_first(codes: np.ndarray, numeric: int) -> tuple[np.ndarray, np.ndarray]:
    # ordinal codes reordered to [categorical, numeric], and the categorical indices
    return (np.hstack([codes[:, numeric:], codes[:, :numeric]]),
            np.arange(codes.shape[1] - numeric))
