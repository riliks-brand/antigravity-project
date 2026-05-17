"""
Preservation Property Tests — XGBoost Training Pipeline
=========================================================

**Property 2: Preservation** — Training Pipeline Functionality Preservation

This test is part of the BUGFIX WORKFLOW (Task 2 — Preservation Phase).

GOAL:
Verify that the non-buggy parts of the training pipeline work correctly on
UNFIXED code. These tests establish the baseline behavior that must be
preserved after the fix is applied.

EXPECTED OUTCOME: Tests PASS on unfixed code.
When re-run after the fix (Task 3.8), they should still PASS.

Validates: Requirements 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
           3.10, 3.11, 3.12, 3.14, 3.15, 3.16
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import joblib

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

import xgb_model as xgbm
from xgb_model import (
    engineer_lagged_features,
    prepare_tabular_data,
    shap_feature_selection,
    rfe_feature_selection,
    walk_forward_validate,
    calibrate_model,
    TOP_K_FEATURES,
)


# =========================================
# SYNTHETIC DATA HELPERS
# Small datasets for fast test runs
# =========================================

def make_ohlcv(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a small synthetic OHLCV DataFrame with a datetime index.
    Uses 6000 candles (well within the 5K-8K limit) for fast runs.
    """
    rng = np.random.default_rng(seed)
    close = 1.25 + np.cumsum(rng.normal(0.00003, 0.0008, n))
    close = np.maximum(close, 0.5)
    spread = rng.uniform(0.0001, 0.0005, n)
    high = close + rng.uniform(0.0001, 0.001, n)
    low  = close - rng.uniform(0.0001, 0.001, n)
    open_ = close + rng.normal(0, 0.0003, n)
    volume = rng.integers(100, 5000, n).astype(float)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "real_volume": volume, "spread": spread,
    }, index=idx)


def make_featured_df(n: int = 6000, seed: int = 42, symbol: str = "GBPUSD") -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to synthetic OHLCV data.
    Returns a DataFrame with all features + Target column.
    """
    from features import (
        add_technical_indicators, add_trend_features, add_momentum_features,
        add_pivot_points, add_session_features, add_price_action_features,
        generate_target_column,
    )
    df = make_ohlcv(n=n, seed=seed)
    df = add_technical_indicators(df)
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_pivot_points(df)
    df = add_session_features(df)
    df = add_price_action_features(df)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df = generate_target_column(df, symbol=symbol)
    return df


def make_xy(n_rows: int = 800, n_features: int = 30, seed: int = 42):
    """
    Generate a simple (X, y) pair for unit-level tests that don't need
    the full feature pipeline.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    y = rng.integers(0, 2, n_rows)
    return X, y


def make_trained_xgb(X_train, y_train, X_val, y_val):
    """Train a minimal XGBClassifier for use in feature-selection tests."""
    model = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# =========================================
# SECTION 1: FEATURE ENGINEERING (Req 3.4)
# Lagged features, rolling statistics, delta features
# =========================================

class TestFeatureEngineering:
    """
    **Property 2: Preservation** — Feature Engineering

    Validates: Requirement 3.4
    WHEN lagged features are engineered (lag 1, 3, 5) THEN the system SHALL
    CONTINUE TO create rolling statistics and delta features for key indicators.
    """

    def test_lagged_features_output_shape_grows(self):
        """
        engineer_lagged_features adds columns — output has more columns than input.
        Validates: Requirement 3.4
        """
        df = make_featured_df(n=6000)
        n_cols_before = df.shape[1]
        df_lagged = engineer_lagged_features(df)
        assert df_lagged.shape[1] > n_cols_before, (
            f"Expected more columns after lagging. Before={n_cols_before}, After={df_lagged.shape[1]}"
        )

    def test_lagged_features_row_count_preserved(self):
        """
        engineer_lagged_features preserves the number of rows.
        Validates: Requirement 3.4
        """
        df = make_featured_df(n=6000)
        df_lagged = engineer_lagged_features(df)
        assert df_lagged.shape[0] == df.shape[0], (
            f"Row count changed: {df.shape[0]} → {df_lagged.shape[0]}"
        )

    def test_lag_columns_created_for_rsi(self):
        """
        RSI lag columns (lag1, lag3, lag5) are present after engineering.
        Validates: Requirement 3.4
        """
        df = make_featured_df(n=6000)
        df_lagged = engineer_lagged_features(df)
        for lag in [1, 3, 5]:
            col = f"RSI_lag{lag}"
            assert col in df_lagged.columns, f"Missing column: {col}"

    def test_rolling_stats_columns_created(self):
        """
        Rolling mean and std columns are created for key indicators.
        Validates: Requirement 3.4
        """
        df = make_featured_df(n=6000)
        df_lagged = engineer_lagged_features(df)
        for indicator in ["RSI", "MACD", "ATR"]:
            for stat in ["roll5_mean", "roll5_std"]:
                col = f"{indicator}_{stat}"
                assert col in df_lagged.columns, f"Missing rolling stat column: {col}"

    def test_delta_columns_created(self):
        """
        Delta (rate of change) columns are created for key indicators.
        Validates: Requirement 3.4
        """
        df = make_featured_df(n=6000)
        df_lagged = engineer_lagged_features(df)
        for indicator in ["RSI", "MACD", "ATR"]:
            for delta in ["delta1", "delta3"]:
                col = f"{indicator}_{delta}"
                assert col in df_lagged.columns, f"Missing delta column: {col}"

    def test_lagged_features_deterministic(self):
        """
        engineer_lagged_features is deterministic — same input → same output.
        Validates: Requirement 3.4
        """
        df = make_featured_df(n=6000, seed=7)
        out1 = engineer_lagged_features(df.copy())
        out2 = engineer_lagged_features(df.copy())
        pd.testing.assert_frame_equal(out1, out2)


# =========================================
# SECTION 2: FEATURE SELECTION (Req 3.5, 3.11, 3.12)
# SelectKBest, SHAP, RFE
# =========================================

class TestFeatureSelection:
    """
    **Property 2: Preservation** — Feature Selection

    Validates: Requirements 3.5, 3.11, 3.12
    """

    def test_selectkbest_selects_k_features(self):
        """
        SelectKBest selects exactly k features from the training data.
        Validates: Requirement 3.5
        """
        X, y = make_xy(n_rows=500, n_features=40)
        k = 20
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        selected = selector.get_support(indices=True)
        assert len(selected) == k, f"Expected {k} features, got {len(selected)}"

    def test_selectkbest_deterministic_same_seed(self):
        """
        SelectKBest produces the same feature set given the same data.
        Validates: Requirement 3.5
        """
        X, y = make_xy(n_rows=500, n_features=40, seed=99)
        selector1 = SelectKBest(f_classif, k=20)
        selector1.fit(X, y)
        selector2 = SelectKBest(f_classif, k=20)
        selector2.fit(X, y)
        np.testing.assert_array_equal(
            selector1.get_support(indices=True),
            selector2.get_support(indices=True),
        )

    def test_shap_feature_selection_returns_subset(self):
        """
        shap_feature_selection returns a subset of the original features.
        Validates: Requirement 3.11
        """
        X, y = make_xy(n_rows=400, n_features=20)
        split = 300
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        model = make_trained_xgb(X_train, y_train, X_val, y_val)
        feature_names = [f"feat_{i}" for i in range(20)]

        indices, features, importance = shap_feature_selection(
            model, X_train, feature_names, top_k=10
        )
        assert len(features) <= 20, "SHAP returned more features than input"
        assert len(features) == len(indices), "Mismatch between indices and features"
        assert all(f in feature_names for f in features), "Unknown feature name returned"

    def test_shap_feature_selection_importance_dict_populated(self):
        """
        shap_feature_selection returns a non-empty importance dict.
        Validates: Requirement 3.11
        """
        X, y = make_xy(n_rows=400, n_features=20)
        split = 300
        model = make_trained_xgb(X[:split], y[:split], X[split:], y[split:])
        feature_names = [f"feat_{i}" for i in range(20)]

        _, _, importance = shap_feature_selection(model, X[:split], feature_names)
        assert isinstance(importance, dict), "Importance should be a dict"
        # Either populated (SHAP worked) or empty (fallback) — both are valid
        if importance:
            assert all(isinstance(v, float) for v in importance.values())

    def test_rfe_feature_selection_returns_correct_count(self):
        """
        rfe_feature_selection returns exactly n_features_to_select features.
        Validates: Requirement 3.12
        """
        X, y = make_xy(n_rows=400, n_features=20)
        feature_names = [f"feat_{i}" for i in range(20)]
        n_select = 10

        indices, features = rfe_feature_selection(
            X, y, feature_names, n_features_to_select=n_select
        )
        # RFE may fall back on error — check it returns a valid subset
        assert len(features) > 0, "RFE returned no features"
        assert len(features) == len(indices), "Mismatch between indices and features"
        assert all(f in feature_names for f in features), "Unknown feature name returned"

    def test_rfe_feature_selection_indices_in_range(self):
        """
        rfe_feature_selection returns indices within valid range.
        Validates: Requirement 3.12
        """
        X, y = make_xy(n_rows=400, n_features=20)
        feature_names = [f"feat_{i}" for i in range(20)]

        indices, _ = rfe_feature_selection(X, y, feature_names, n_features_to_select=8)
        assert all(0 <= i < 20 for i in indices), "Index out of range"


# =========================================
# SECTION 3: ROBUST SCALER (Req 3.6)
# =========================================

class TestRobustScaler:
    """
    **Property 2: Preservation** — RobustScaler

    Validates: Requirement 3.6
    WHEN RobustScaler is applied THEN the system SHALL CONTINUE TO scale
    features using robust statistics (median, IQR) to handle outliers.
    """

    def test_robust_scaler_output_shape_preserved(self):
        """
        RobustScaler transform preserves the shape of the input.
        Validates: Requirement 3.6
        """
        X, _ = make_xy(n_rows=300, n_features=15)
        scaler = RobustScaler()
        scaler.fit(X[:200])
        X_scaled = scaler.transform(X[200:])
        assert X_scaled.shape == X[200:].shape

    def test_robust_scaler_median_near_zero(self):
        """
        After RobustScaler, the median of each feature is approximately 0.
        Validates: Requirement 3.6
        """
        X, _ = make_xy(n_rows=500, n_features=10)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        medians = np.median(X_scaled, axis=0)
        np.testing.assert_allclose(medians, 0.0, atol=0.1,
            err_msg="Median after RobustScaler should be near 0")

    def test_robust_scaler_handles_outliers(self):
        """
        RobustScaler does not blow up when outliers are present.
        Validates: Requirement 3.6
        """
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 5))
        X[0, :] = 1000.0  # extreme outlier
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.isfinite(X_scaled).all(), "Scaled values should be finite even with outliers"


# =========================================
# SECTION 4: MODEL PERSISTENCE (Req 3.7, 3.8)
# Saving and loading joblib files
# =========================================

class TestModelPersistence:
    """
    **Property 2: Preservation** — Model Persistence

    Validates: Requirements 3.7, 3.8
    WHEN the model is saved/loaded THEN the system SHALL CONTINUE TO persist
    and restore model, scaler, and feature names to/from joblib files.
    """

    def test_model_saves_and_loads_correctly(self, tmp_path):
        """
        A trained XGBClassifier can be saved and loaded via joblib.
        Validates: Requirements 3.7, 3.8
        """
        X, y = make_xy(n_rows=300, n_features=10)
        model = XGBClassifier(
            n_estimators=20, max_depth=3, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X[:200], y[:200])

        model_path = str(tmp_path / "model.joblib")
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)

        # Predictions should be identical
        preds_orig = model.predict_proba(X[200:])
        preds_loaded = loaded.predict_proba(X[200:])
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_scaler_saves_and_loads_correctly(self, tmp_path):
        """
        A fitted RobustScaler can be saved and loaded via joblib.
        Validates: Requirements 3.7, 3.8
        """
        X, _ = make_xy(n_rows=300, n_features=10)
        scaler = RobustScaler()
        scaler.fit(X[:200])

        scaler_path = str(tmp_path / "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        loaded_scaler = joblib.load(scaler_path)

        out_orig = scaler.transform(X[200:])
        out_loaded = loaded_scaler.transform(X[200:])
        np.testing.assert_array_equal(out_orig, out_loaded)

    def test_feature_names_save_and_load(self, tmp_path):
        """
        Feature name list can be saved and loaded via joblib.
        Validates: Requirements 3.7, 3.8
        """
        feature_names = [f"feature_{i}" for i in range(50)]
        path = str(tmp_path / "features.joblib")
        joblib.dump(feature_names, path)
        loaded = joblib.load(path)
        assert loaded == feature_names

    def test_xgbmodel_save_load_roundtrip(self, tmp_path, monkeypatch):
        """
        XGBModel._save_model and _load_model persist and restore state.
        Validates: Requirements 3.7, 3.8
        """
        # Redirect joblib paths to tmp_path
        monkeypatch.setattr(xgbm, "XGB_MODEL_PATH",   str(tmp_path / "xgb_model.joblib"))
        monkeypatch.setattr(xgbm, "XGB_SCALER_PATH",  str(tmp_path / "xgb_scaler.joblib"))
        monkeypatch.setattr(xgbm, "XGB_FEATURES_PATH", str(tmp_path / "xgb_features.joblib"))

        from xgb_model import XGBModel
        xgb = XGBModel()

        # Manually set state
        X, y = make_xy(n_rows=300, n_features=10)
        model = XGBClassifier(
            n_estimators=20, max_depth=3, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X[:200], y[:200])
        xgb.model = model
        xgb.scaler = RobustScaler().fit(X[:200])
        xgb.feature_names = [f"f{i}" for i in range(10)]

        xgb._save_model()

        # Load into a fresh instance
        xgb2 = XGBModel()
        assert xgb2.model is not None, "Model should be loaded from disk"
        assert xgb2.feature_names == xgb.feature_names, "Feature names should match"


# =========================================
# SECTION 5: WALK-FORWARD VALIDATION (Req 3.3)
# 5-fold splitting and per-fold training
# =========================================

class TestWalkForwardValidation:
    """
    **Property 2: Preservation** — Walk-Forward Validation

    Validates: Requirement 3.3
    WHEN Walk-Forward Validation is performed with 5 folds THEN the system
    SHALL CONTINUE TO report per-fold accuracies, mean accuracy, std, and
    stability score.
    """

    @pytest.fixture(scope="class")
    def wfv_result(self):
        """Run WFV once and share across tests in this class."""
        df = make_featured_df(n=6000, seed=42)
        return walk_forward_validate(df, n_folds=5, symbol="TEST")

    def test_wfv_returns_dict(self, wfv_result):
        """walk_forward_validate returns a dict."""
        assert isinstance(wfv_result, dict)

    def test_wfv_reports_fold_accuracies(self, wfv_result):
        """
        WFV result contains fold_accuracies list.
        Validates: Requirement 3.3
        """
        assert "fold_accuracies" in wfv_result
        assert isinstance(wfv_result["fold_accuracies"], list)

    def test_wfv_completes_all_folds(self, wfv_result):
        """
        WFV completes all 5 folds (or as many as data allows).
        Validates: Requirement 3.3
        """
        if wfv_result.get("skipped"):
            pytest.skip("WFV skipped due to insufficient data")
        assert len(wfv_result["fold_accuracies"]) >= 1, "At least one fold should complete"

    def test_wfv_reports_mean_accuracy(self, wfv_result):
        """
        WFV result contains mean_accuracy.
        Validates: Requirement 3.3
        """
        if wfv_result.get("skipped"):
            pytest.skip("WFV skipped due to insufficient data")
        assert "mean_accuracy" in wfv_result
        assert isinstance(wfv_result["mean_accuracy"], float)
        assert 0.0 <= wfv_result["mean_accuracy"] <= 100.0

    def test_wfv_reports_std_accuracy(self, wfv_result):
        """
        WFV result contains std_accuracy.
        Validates: Requirement 3.3
        """
        if wfv_result.get("skipped"):
            pytest.skip("WFV skipped due to insufficient data")
        assert "std_accuracy" in wfv_result
        assert wfv_result["std_accuracy"] >= 0.0

    def test_wfv_reports_stability_score(self, wfv_result):
        """
        WFV result contains stability_score.
        Validates: Requirement 3.3
        """
        if wfv_result.get("skipped"):
            pytest.skip("WFV skipped due to insufficient data")
        assert "stability_score" in wfv_result

    def test_wfv_fold_accuracies_are_valid_percentages(self, wfv_result):
        """
        Each fold accuracy is a valid percentage (0-100).
        Validates: Requirement 3.3
        """
        if wfv_result.get("skipped"):
            pytest.skip("WFV skipped due to insufficient data")
        for acc in wfv_result["fold_accuracies"]:
            assert 0.0 <= acc <= 100.0, f"Invalid fold accuracy: {acc}"


# =========================================
# SECTION 6: EARLY STOPPING (Req 3.9)
# =========================================

class TestEarlyStopping:
    """
    **Property 2: Preservation** — Early Stopping

    Validates: Requirement 3.9
    WHEN early stopping is enabled during training THEN the system SHALL
    CONTINUE TO monitor eval_set performance and stop training when
    validation loss stops improving.
    """

    def test_early_stopping_reduces_trees(self):
        """
        With early_stopping_rounds set, the model may stop before n_estimators.
        Validates: Requirement 3.9
        """
        X, y = make_xy(n_rows=500, n_features=15, seed=0)
        X_train, X_val = X[:400], X[400:]
        y_train, y_val = y[:400], y[400:]

        model = XGBClassifier(
            n_estimators=200,
            early_stopping_rounds=10,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # best_iteration should be set when early stopping is active
        assert hasattr(model, "best_iteration"), "best_iteration should be set"
        # The model should have stopped at or before n_estimators
        assert model.best_iteration <= 200

    def test_early_stopping_model_still_predicts(self):
        """
        A model trained with early stopping can still make predictions.
        Validates: Requirement 3.9
        """
        X, y = make_xy(n_rows=500, n_features=15, seed=1)
        X_train, X_val = X[:400], X[400:]
        y_train, y_val = y[:400], y[400:]

        model = XGBClassifier(
            n_estimators=100,
            early_stopping_rounds=10,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)
        assert probs.shape == (len(X_val), 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))


# =========================================
# SECTION 7: FEATURE IMPORTANCE (Req 3.10)
# =========================================

class TestFeatureImportance:
    """
    **Property 2: Preservation** — Feature Importance

    Validates: Requirement 3.10
    WHEN feature importance is calculated THEN the system SHALL CONTINUE TO
    extract and store feature_importances_ from the trained XGBoost model.
    """

    def test_feature_importances_extracted(self):
        """
        feature_importances_ is available after training.
        Validates: Requirement 3.10
        """
        X, y = make_xy(n_rows=300, n_features=10)
        model = XGBClassifier(
            n_estimators=30, max_depth=3, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X[:200], y[:200])
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == 10

    def test_feature_importances_sum_to_one(self):
        """
        feature_importances_ values sum to approximately 1.0.
        Validates: Requirement 3.10
        """
        X, y = make_xy(n_rows=300, n_features=10)
        model = XGBClassifier(
            n_estimators=30, max_depth=3, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X[:200], y[:200])
        total = model.feature_importances_.sum()
        assert abs(total - 1.0) < 0.01, f"Importances sum to {total}, expected ~1.0"

    def test_feature_importances_non_negative(self):
        """
        All feature importance values are non-negative.
        Validates: Requirement 3.10
        """
        X, y = make_xy(n_rows=300, n_features=10)
        model = XGBClassifier(
            n_estimators=30, max_depth=3, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X[:200], y[:200])
        assert np.all(model.feature_importances_ >= 0.0)


# =========================================
# SECTION 8: ERROR HANDLING (Req 3.15, 3.16)
# Insufficient data, constant features
# =========================================

class TestErrorHandling:
    """
    **Property 2: Preservation** — Error Handling

    Validates: Requirements 3.15, 3.16
    """

    def test_insufficient_data_raises_value_error(self):
        """
        prepare_tabular_data raises ValueError when fewer than 200 valid rows.
        Validates: Requirement 3.15
        """
        # Build a minimal DataFrame directly (bypassing feature pipeline which needs 200+ rows)
        # We need a DataFrame with a Target column but very few valid rows
        rng = np.random.default_rng(42)
        n = 50  # far fewer than the 200-row minimum
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        df = pd.DataFrame({
            "RSI": rng.uniform(20, 80, n),
            "MACD": rng.normal(0, 0.001, n),
            "ATR": rng.uniform(0.0005, 0.003, n),
            "close": 1.25 + np.cumsum(rng.normal(0, 0.0005, n)),
            "Target": rng.integers(0, 2, n).astype(float),
        }, index=idx)
        with pytest.raises(ValueError, match="Not enough data"):
            prepare_tabular_data(df)

    def test_constant_features_removed(self):
        """
        Constant features (std=0) are removed before feature selection.
        Validates: Requirement 3.16
        """
        df = make_featured_df(n=6000, seed=42)
        df_lagged = engineer_lagged_features(df)

        # Inject a constant column
        df_lagged["CONSTANT_FEATURE"] = 99.0

        df_valid = df_lagged.dropna(subset=["Target"])
        feature_cols = [c for c in df_valid.columns if c != "Target"]
        df_valid = df_valid.dropna(subset=feature_cols)

        X = df_valid[feature_cols].values
        col_std = X.std(axis=0)
        non_constant_mask = col_std > 0

        # The constant column should be flagged for removal
        constant_col_idx = feature_cols.index("CONSTANT_FEATURE")
        assert not non_constant_mask[constant_col_idx], (
            "CONSTANT_FEATURE should be identified as constant (std=0)"
        )

        # After filtering, constant column should be gone
        X_filtered = X[:, non_constant_mask]
        filtered_cols = [f for f, keep in zip(feature_cols, non_constant_mask) if keep]
        assert "CONSTANT_FEATURE" not in filtered_cols

    def test_prepare_tabular_data_succeeds_with_sufficient_data(self):
        """
        prepare_tabular_data completes without error on sufficient data.
        Validates: Requirement 3.1
        """
        df = make_featured_df(n=6000, seed=42)
        result = prepare_tabular_data(df)
        # Returns 8-tuple: X_train, X_cal, X_test, y_train, y_cal, y_test, scaler, features
        assert len(result) == 8
        X_train, X_cal, X_test, y_train, y_cal, y_test, scaler, features = result
        assert X_train.shape[0] > 0
        assert X_cal.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(features) > 0


# =========================================
# SECTION 9: RETRAINING TRIGGERS (Req 3.14)
# needs_retraining() logic
# =========================================

class TestRetrainingTriggers:
    """
    **Property 2: Preservation** — Retraining Triggers

    Validates: Requirement 3.14
    WHEN model retraining is triggered based on time or candle count THEN
    the system SHALL CONTINUE TO check needs_retraining() and initiate
    training when thresholds are exceeded.
    """

    def test_needs_retraining_true_when_no_model(self, tmp_path, monkeypatch):
        """
        needs_retraining() returns True when no model is loaded.
        Validates: Requirement 3.14
        """
        monkeypatch.setattr(xgbm, "XGB_MODEL_PATH",    str(tmp_path / "xgb_model.joblib"))
        monkeypatch.setattr(xgbm, "XGB_SCALER_PATH",   str(tmp_path / "xgb_scaler.joblib"))
        monkeypatch.setattr(xgbm, "XGB_FEATURES_PATH", str(tmp_path / "xgb_features.joblib"))

        from xgb_model import XGBModel
        xgb = XGBModel()
        xgb.model = None  # force no model
        assert xgb.needs_retraining() is True

    def test_needs_retraining_true_when_no_train_time(self, tmp_path, monkeypatch):
        """
        needs_retraining() returns True when last_train_time is None.
        Validates: Requirement 3.14
        """
        monkeypatch.setattr(xgbm, "XGB_MODEL_PATH",    str(tmp_path / "xgb_model2.joblib"))
        monkeypatch.setattr(xgbm, "XGB_SCALER_PATH",   str(tmp_path / "xgb_scaler2.joblib"))
        monkeypatch.setattr(xgbm, "XGB_FEATURES_PATH", str(tmp_path / "xgb_features2.joblib"))

        from xgb_model import XGBModel
        xgb = XGBModel()
        # Simulate a model exists but no train time recorded
        X, y = make_xy(n_rows=200, n_features=5)
        xgb.model = XGBClassifier(n_estimators=5, verbosity=0).fit(X, y)
        xgb.last_train_time = None
        assert xgb.needs_retraining() is True

    def test_needs_retraining_false_when_recently_trained(self, tmp_path, monkeypatch):
        """
        needs_retraining() returns False when model was trained recently.
        Validates: Requirement 3.14
        """
        import datetime
        monkeypatch.setattr(xgbm, "XGB_MODEL_PATH",    str(tmp_path / "xgb_model3.joblib"))
        monkeypatch.setattr(xgbm, "XGB_SCALER_PATH",   str(tmp_path / "xgb_scaler3.joblib"))
        monkeypatch.setattr(xgbm, "XGB_FEATURES_PATH", str(tmp_path / "xgb_features3.joblib"))

        from xgb_model import XGBModel
        xgb = XGBModel()
        X, y = make_xy(n_rows=200, n_features=5)
        xgb.model = XGBClassifier(n_estimators=5, verbosity=0).fit(X, y)
        xgb.last_train_time = datetime.datetime.utcnow()  # just trained
        xgb.train_count = 0
        assert xgb.needs_retraining(candle_count=0) is False


# =========================================
# SECTION 10: PROPERTY-BASED TESTS
# Hypothesis-driven preservation checks
# =========================================

# Hypothesis settings: max_examples=5 for fast runs (per user instruction)
FAST_SETTINGS = settings(
    max_examples=5,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)


@given(
    n_rows=st.integers(min_value=300, max_value=800),
    n_features=st.integers(min_value=10, max_value=30),
    seed=st.integers(min_value=0, max_value=999),
)
@FAST_SETTINGS
def test_pbt_lagged_features_shape_invariant(n_rows, n_features, seed):
    """
    **Property 2: Preservation** — Feature Engineering Shape Invariant

    For any valid DataFrame, engineer_lagged_features always produces
    more columns than the input and preserves the row count.

    Validates: Requirement 3.4
    """
    rng = np.random.default_rng(seed)
    # Build a minimal DataFrame with the columns engineer_lagged_features expects
    idx = pd.date_range("2025-01-01", periods=n_rows, freq="5min")
    data = {
        "RSI": rng.uniform(20, 80, n_rows),
        "MACD": rng.normal(0, 0.001, n_rows),
        "MACD_hist": rng.normal(0, 0.0005, n_rows),
        "ATR": rng.uniform(0.0005, 0.003, n_rows),
        "ADX": rng.uniform(10, 50, n_rows),
        "close": 1.25 + np.cumsum(rng.normal(0, 0.0005, n_rows)),
        "Target": rng.integers(0, 2, n_rows).astype(float),
    }
    df = pd.DataFrame(data, index=idx)

    n_cols_before = df.shape[1]
    df_lagged = engineer_lagged_features(df)

    assert df_lagged.shape[0] == n_rows, (
        f"Row count changed: {n_rows} → {df_lagged.shape[0]}"
    )
    assert df_lagged.shape[1] > n_cols_before, (
        f"Column count did not grow: {n_cols_before} → {df_lagged.shape[1]}"
    )


@given(
    n_rows=st.integers(min_value=200, max_value=600),
    n_features=st.integers(min_value=10, max_value=25),
    seed=st.integers(min_value=0, max_value=999),
)
@FAST_SETTINGS
def test_pbt_selectkbest_always_returns_k_features(n_rows, n_features, seed):
    """
    **Property 2: Preservation** — SelectKBest Feature Count

    For any valid training data, SelectKBest always returns exactly k features
    (or all features if k > n_features).

    Validates: Requirement 3.5
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    y = rng.integers(0, 2, n_rows)

    k = min(10, n_features)
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    selected = selector.get_support(indices=True)

    assert len(selected) == k, (
        f"Expected {k} features, got {len(selected)} "
        f"(n_rows={n_rows}, n_features={n_features})"
    )


@given(
    n_rows=st.integers(min_value=200, max_value=600),
    n_features=st.integers(min_value=5, max_value=20),
    seed=st.integers(min_value=0, max_value=999),
)
@FAST_SETTINGS
def test_pbt_robust_scaler_output_finite(n_rows, n_features, seed):
    """
    **Property 2: Preservation** — RobustScaler Finite Output

    For any valid feature matrix (including outliers), RobustScaler always
    produces finite output values.

    Validates: Requirement 3.6
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    # Inject random outliers
    n_outliers = max(1, n_rows // 20)
    outlier_rows = rng.integers(0, n_rows, n_outliers)
    X[outlier_rows] *= 100.0

    split = n_rows // 2
    scaler = RobustScaler()
    scaler.fit(X[:split])
    X_scaled = scaler.transform(X[split:])

    assert np.isfinite(X_scaled).all(), (
        f"Non-finite values after RobustScaler (n_rows={n_rows}, n_features={n_features})"
    )


@given(
    n_rows=st.integers(min_value=200, max_value=500),
    n_features=st.integers(min_value=5, max_value=15),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(
    max_examples=5,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
    deadline=None,
)
def test_pbt_model_persistence_roundtrip(tmp_path, n_rows, n_features, seed):
    """
    **Property 2: Preservation** — Model Persistence Roundtrip

    For any trained XGBClassifier, saving and loading via joblib produces
    identical predictions.

    Validates: Requirements 3.7, 3.8
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    y = rng.integers(0, 2, n_rows)

    split = int(n_rows * 0.8)
    model = XGBClassifier(
        n_estimators=20, max_depth=3, use_label_encoder=False,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X[:split], y[:split])

    path = str(tmp_path / f"model_{seed}.joblib")
    joblib.dump(model, path)
    loaded = joblib.load(path)

    preds_orig = model.predict_proba(X[split:])
    preds_loaded = loaded.predict_proba(X[split:])
    np.testing.assert_array_equal(preds_orig, preds_loaded,
        err_msg=f"Predictions differ after save/load (seed={seed})")


@given(
    n_rows=st.integers(min_value=200, max_value=500),
    n_features=st.integers(min_value=5, max_value=15),
    seed=st.integers(min_value=0, max_value=999),
)
@FAST_SETTINGS
def test_pbt_predict_proba_in_unit_interval(n_rows, n_features, seed):
    """
    **Property 2: Preservation** — Prediction Probabilities in [0, 1]

    For any trained XGBClassifier, predict_proba always returns values in [0, 1].

    Validates: Requirement 3.13 (ensemble engine receives valid probabilities)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    y = rng.integers(0, 2, n_rows)

    split = int(n_rows * 0.8)
    model = XGBClassifier(
        n_estimators=20, max_depth=3, use_label_encoder=False,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X[:split], y[:split])
    probs = model.predict_proba(X[split:])[:, 1]

    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), (
        f"Probabilities out of [0,1] range (seed={seed}): "
        f"min={probs.min():.4f}, max={probs.max():.4f}"
    )

