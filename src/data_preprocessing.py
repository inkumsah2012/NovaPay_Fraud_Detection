import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from config.constant import Feature_Engineered_Data


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DATA PREPROCESSING CLASS
# ============================================================

class DataPreprocessing:
    """
    Prepare NovaPay transaction data for machine-learning models.

    The class supports:
    - Chronological train/test splitting
    - Predictor/target separation
    - One-hot encoding for nominal categorical variables
    - Standard scaling for numerical variables
    - Passthrough of binary indicators
    - Transformation of unseen/inference data
    - Persistence of the fitted preprocessing pipeline

    Important
    ---------
    The preprocessing transformer is fitted only on training data
    to prevent information leakage from the future test period.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_features: list[str],
        numeric_features: list[str],
        binary_features: Optional[list[str]] = None,
        target_column: str = "is_fraud",
        timestamp_column: str = "timestamp",
    ) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "The input DataFrame is empty."
            )

        self.df = df.copy()

        self.categorical_features = list(
            categorical_features
        )

        self.numeric_features = list(
            numeric_features
        )

        self.binary_features = list(
            binary_features or []
        )

        self.target_column = target_column
        self.timestamp_column = timestamp_column

        self.all_features = (
            self.categorical_features
            + self.binary_features
            + self.numeric_features
        )

        self.preprocessor: Optional[
            ColumnTransformer
        ] = None

        self._validate_feature_configuration()


    # ========================================================
    # INTERNAL VALIDATION
    # ========================================================

    def _validate_feature_configuration(
        self,
    ) -> None:
        """
        Validate feature groups and ensure that a feature does
        not appear in more than one preprocessing category.
        """

        feature_groups = {
            "categorical": self.categorical_features,
            "binary": self.binary_features,
            "numeric": self.numeric_features,
        }

        all_configured_features = []

        for group_name, features in feature_groups.items():

            if len(features) != len(set(features)):
                raise ValueError(
                    f"Duplicate features found within "
                    f"{group_name} features."
                )

            all_configured_features.extend(features)

        duplicates = {
            feature
            for feature in all_configured_features
            if all_configured_features.count(feature) > 1
        }

        if duplicates:
            raise ValueError(
                "Features cannot belong to multiple "
                f"preprocessing groups: {sorted(duplicates)}"
            )


    def _require_columns(
        self,
        df: pd.DataFrame,
        columns: list[str],
        operation: str,
    ) -> None:
        """
        Verify that all columns required by an operation exist.
        """

        missing_columns = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{operation} cannot be completed. "
                f"Missing columns: {missing_columns}"
            )


    # ========================================================
    # 1. CHRONOLOGICAL TRAIN / TEST SPLIT
    # ========================================================

    def chronological_split(
        self,
        train_size: float = 0.80,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split transactions chronologically into training and
        testing datasets.

        Parameters
        ----------
        train_size : float, default=0.80
            Proportion of earliest transactions assigned to
            the training dataset.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Training and testing DataFrames.
        """

        try:
            logger.info(
                "Starting chronological train-test split"
            )

            if not 0 < train_size < 1:
                raise ValueError(
                    "train_size must be greater than 0 "
                    "and less than 1."
                )

            self._require_columns(
                self.df,
                [
                    self.timestamp_column,
                    self.target_column,
                ],
                operation="Chronological split",
            )

            timestamps = pd.to_datetime(
                self.df[self.timestamp_column],
                errors="coerce",
                utc=True,
            )

            invalid_timestamp_count = (
                timestamps.isna().sum()
            )

            if invalid_timestamp_count > 0:
                raise ValueError(
                    f"{invalid_timestamp_count} invalid "
                    "timestamps were detected."
                )

            self.df[self.timestamp_column] = timestamps

            # mergesort is stable and preserves original order
            # when timestamps are identical.
            sorted_df = (
                self.df
                .sort_values(
                    self.timestamp_column,
                    kind="mergesort",
                )
                .reset_index(drop=True)
            )

            split_index = int(
                len(sorted_df) * train_size
            )

            if (
                split_index == 0
                or split_index == len(sorted_df)
            ):
                raise ValueError(
                    "The selected train_size produces an "
                    "empty training or testing dataset."
                )

            train_df = (
                sorted_df
                .iloc[:split_index]
                .copy()
            )

            test_df = (
                sorted_df
                .iloc[split_index:]
                .copy()
            )

            self._validate_target(
                train_df,
                dataset_name="Training",
            )

            self._validate_target(
                test_df,
                dataset_name="Testing",
            )

            logger.info(
                "Chronological split completed: "
                "%d training rows and %d testing rows",
                len(train_df),
                len(test_df),
            )

            self._log_split_summary(
                train_df,
                test_df,
            )

            return train_df, test_df

        except Exception:
            logger.exception(
                "Chronological train-test split failed"
            )
            raise


    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    def _validate_target(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        """
        Verify that the target contains valid binary labels.
        """

        self._require_columns(
            df,
            [self.target_column],
            operation=f"{dataset_name} target validation",
        )

        if df[self.target_column].isna().any():
            raise ValueError(
                f"{dataset_name} target contains "
                "missing values."
            )

        unique_classes = set(
            df[self.target_column].unique()
        )

        if not unique_classes.issubset({0, 1}):
            raise ValueError(
                f"{dataset_name} target must contain only "
                f"0 and 1. Found: {unique_classes}"
            )

        if len(unique_classes) < 2:
            logger.warning(
                "%s dataset contains only one target class: %s",
                dataset_name,
                unique_classes,
            )


    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    def _log_split_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """
        Display chronological boundaries and fraud prevalence.
        """

        train_fraud_count = int(
            train_df[self.target_column].sum()
        )

        test_fraud_count = int(
            test_df[self.target_column].sum()
        )

        train_fraud_rate = (
            train_df[self.target_column].mean()
        )

        test_fraud_rate = (
            test_df[self.target_column].mean()
        )

        print(
            "\nChronological Train-Test Split:"
        )

        print(
            f"Train: {len(train_df):,} rows "
            f"({train_fraud_count:,} fraud, "
            f"{train_fraud_rate:.3%} fraud rate)"
        )

        print(
            f"Test:  {len(test_df):,} rows "
            f"({test_fraud_count:,} fraud, "
            f"{test_fraud_rate:.3%} fraud rate)"
        )

        print(
            "\nTraining Period:"
        )

        print(
            f"{train_df[self.timestamp_column].min()} "
            f"to "
            f"{train_df[self.timestamp_column].max()}"
        )

        print(
            "\nTesting Period:"
        )

        print(
            f"{test_df[self.timestamp_column].min()} "
            f"to "
            f"{test_df[self.timestamp_column].max()}"
        )

        logger.info(
            "Training fraud rate: %.4f | "
            "Testing fraud rate: %.4f",
            train_fraud_rate,
            test_fraud_rate,
        )


    # ========================================================
    # 2. PREPARE X AND y
    # ========================================================

    def prepare_features_target(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Separate predictor variables and the fraud target.

        Parameters
        ----------
        train_df : pd.DataFrame
            Chronological training dataset.

        test_df : pd.DataFrame
            Chronological testing dataset.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """

        try:
            logger.info(
                "Preparing model predictors and target"
            )

            required_columns = (
                self.all_features
                + [self.target_column]
            )

            self._require_columns(
                train_df,
                required_columns,
                operation="Training feature preparation",
            )

            self._require_columns(
                test_df,
                required_columns,
                operation="Testing feature preparation",
            )

            X_train = (
                train_df[self.all_features]
                .copy()
            )

            X_test = (
                test_df[self.all_features]
                .copy()
            )

            y_train = (
                train_df[self.target_column]
                .astype("int8")
                .copy()
            )

            y_test = (
                test_df[self.target_column]
                .astype("int8")
                .copy()
            )

            logger.info(
                "Predictor and target datasets prepared: "
                "X_train=%s, X_test=%s",
                X_train.shape,
                X_test.shape,
            )

            print(
                "\nTraining and Testing Shapes:"
            )

            print(
                f"X_train: {X_train.shape}, "
                f"y_train: {y_train.shape}"
            )

            print(
                f"X_test:  {X_test.shape}, "
                f"y_test:  {y_test.shape}"
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
            )

        except Exception:
            logger.exception(
                "Failed to prepare predictors and target"
            )
            raise


    # ========================================================
    # 3. BUILD PREPROCESSING PIPELINE
    # ========================================================

    def build_preprocessor(
        self,
    ) -> ColumnTransformer:
        """
        Build the scikit-learn preprocessing transformer.

        Categorical variables
        ---------------------
        - Most-frequent imputation
        - One-hot encoding

        Numerical variables
        -------------------
        - Median imputation
        - Standard scaling

        Binary variables
        ----------------
        - Most-frequent imputation
        - Passed through without one-hot encoding
        """

        try:
            logger.info(
                "Building preprocessing pipeline"
            )

            transformers = []

            # ------------------------------------------------
            # Categorical pipeline
            # ------------------------------------------------

            if self.categorical_features:

                categorical_pipeline = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                )

                transformers.append(
                    (
                        "cat",
                        categorical_pipeline,
                        self.categorical_features,
                    )
                )

            # ------------------------------------------------
            # Binary pipeline
            # ------------------------------------------------

            if self.binary_features:

                binary_pipeline = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                            ),
                        ),
                    ]
                )

                transformers.append(
                    (
                        "bin",
                        binary_pipeline,
                        self.binary_features,
                    )
                )

            # ------------------------------------------------
            # Numerical pipeline
            # ------------------------------------------------

            if self.numeric_features:

                numeric_pipeline = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                )

                transformers.append(
                    (
                        "num",
                        numeric_pipeline,
                        self.numeric_features,
                    )
                )

            if not transformers:
                raise ValueError(
                    "No preprocessing features were configured."
                )

            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=True,
            )

            logger.info(
                "Preprocessing pipeline created successfully"
            )

            return self.preprocessor

        except Exception:
            logger.exception(
                "Failed to build preprocessing pipeline"
            )
            raise


    # ========================================================
    # 4. FIT TRAINING DATA AND TRANSFORM TEST DATA
    # ========================================================

    def transform_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing transformations on training data only,
        then apply the fitted transformer to both datasets.
        """

        try:
            logger.info(
                "Starting preprocessing transformation"
            )

            self._require_columns(
                X_train,
                self.all_features,
                operation="Training transformation",
            )

            self._require_columns(
                X_test,
                self.all_features,
                operation="Testing transformation",
            )

            if self.preprocessor is None:
                self.build_preprocessor()

            # Critical anti-leakage rule:
            # fit only on historical training data.
            X_train_processed = (
                self.preprocessor
                .fit_transform(X_train)
            )

            X_test_processed = (
                self.preprocessor
                .transform(X_test)
            )

            if (
                X_train_processed.shape[1]
                != X_test_processed.shape[1]
            ):
                raise RuntimeError(
                    "Training and testing datasets have "
                    "different processed feature counts."
                )

            if np.isnan(
                np.asarray(
                    X_train_processed,
                    dtype=float,
                )
            ).any():
                raise ValueError(
                    "NaN values remain in processed "
                    "training data."
                )

            if np.isnan(
                np.asarray(
                    X_test_processed,
                    dtype=float,
                )
            ).any():
                raise ValueError(
                    "NaN values remain in processed "
                    "testing data."
                )

            logger.info(
                "Preprocessing completed: "
                "train=%s, test=%s",
                X_train_processed.shape,
                X_test_processed.shape,
            )

            print(
                "\nProcessed Dataset Shapes:"
            )

            print(
                f"X_train processed: "
                f"{X_train_processed.shape}"
            )

            print(
                f"X_test processed:  "
                f"{X_test_processed.shape}"
            )

            print(
                f"\nOriginal model features: "
                f"{len(self.all_features)}"
            )

            print(
                f"Features after preprocessing: "
                f"{X_train_processed.shape[1]}"
            )

            return (
                X_train_processed,
                X_test_processed,
            )

        except Exception:
            logger.exception(
                "Data preprocessing transformation failed"
            )
            raise


    # ========================================================
    # 5. TRANSFORM NEW / INFERENCE DATA
    # ========================================================

    def transform_new_data(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform new transactions using the already-fitted
        preprocessing pipeline.

        This method must never fit the transformer again.
        """

        try:
            if self.preprocessor is None:
                raise RuntimeError(
                    "Preprocessor has not been fitted."
                )

            check_is_fitted(
                self.preprocessor
            )

            self._require_columns(
                df,
                self.all_features,
                operation="Inference transformation",
            )

            transformed_data = (
                self.preprocessor
                .transform(
                    df[self.all_features]
                )
            )

            logger.info(
                "%d new transactions transformed",
                len(df),
            )

            return transformed_data

        except Exception:
            logger.exception(
                "Failed to transform inference data"
            )
            raise


    # ========================================================
    # 6. GET TRANSFORMED FEATURE NAMES
    # ========================================================

    def get_feature_names(
        self,
    ) -> np.ndarray:
        """
        Return the transformed feature names after fitting.
        """

        try:
            if self.preprocessor is None:
                raise RuntimeError(
                    "Preprocessor has not been created."
                )

            check_is_fitted(
                self.preprocessor
            )

            feature_names = (
                self.preprocessor
                .get_feature_names_out()
            )

            logger.info(
                "%d processed feature names retrieved",
                len(feature_names),
            )

            print(
                "\nNumber of Processed Features:"
            )

            print(
                len(feature_names)
            )

            print(
                "\nFirst 20 Processed Features:"
            )

            print(
                feature_names[:20]
            )

            return feature_names

        except Exception:
            logger.exception(
                "Failed to retrieve feature names"
            )
            raise


    # ========================================================
    # 7. SAVE FITTED PREPROCESSOR
    # ========================================================

    def save_preprocessor(
        self,
        output_path: str | Path = (
            "artifacts/preprocessor.joblib"
        ),
    ) -> Path:
        """
        Persist the fitted preprocessing pipeline for inference.
        """

        try:
            if self.preprocessor is None:
                raise RuntimeError(
                    "Preprocessor has not been created."
                )

            check_is_fitted(
                self.preprocessor
            )

            output_path = Path(
                output_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            joblib.dump(
                self.preprocessor,
                output_path,
            )

            logger.info(
                "Fitted preprocessor saved to %s",
                output_path,
            )

            return output_path

        except Exception:
            logger.exception(
                "Failed to save fitted preprocessor"
            )
            raise


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

CATEGORICAL_FEATURES = [
    "channel",
    "kyc_tier",
    "home_country",
    "source_currency",
    "dest_currency",
    "ip_country",
    "transfer_corridor",
    "day_of_week",
]


BINARY_FEATURES = [
    "new_device",
    "location_mismatch",
    "is_weekend",
    "night_hours",
    "account_very_new",
    "account_new",
    "velocity_burst",
    "amount_high",
    "ip_high_risk",
    "device_low_trust",
]


NUMERIC_FEATURES = [
    "amount_src",
    "amount_usd",
    "fee",
    "exchange_rate_src_to_dest",
    "ip_risk_score",
    "account_age_days",
    "device_trust_score",
    "chargeback_history_count",
    "risk_score_internal",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "corridor_risk",
    "time_of_day",
]


# ============================================================
# LOAD FEATURE-ENGINEERED DATA
# ============================================================

def load_feature_engineered_data(
    input_path: str | Path = Feature_Engineered_Data,
) -> pd.DataFrame:
    """
    Load the feature-engineered NovaPay dataset.
    """

    path = Path(
        input_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: {path}"
        )

    logger.info(
        "Loading feature-engineered dataset from %s",
        path,
    )

    df = pd.read_csv(
        path
    )

    if df.empty:
        raise ValueError(
            "Feature-engineered dataset is empty."
        )

    logger.info(
        "Feature-engineered dataset loaded: "
        "%d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )

    return df


# ============================================================
# RUN DATA PREPROCESSING PIPELINE
# ============================================================

def run_pipeline(
    train_size: float = 0.80,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
    np.ndarray,
    DataPreprocessing,
]:
    """
    Execute the complete NovaPay preprocessing pipeline.

    Returns
    -------
    tuple
        X_train_processed
        X_test_processed
        y_train
        y_test
        feature_names
        fitted DataPreprocessing instance
    """

    logger.info(
        "Starting NovaPay data preprocessing pipeline"
    )

    # --------------------------------------------------------
    # Load feature-engineered data
    # --------------------------------------------------------

    nova_pay_data = (
        load_feature_engineered_data()
    )

    print(
        f"\nFeature Engineered Dataset Shape: "
        f"{nova_pay_data.shape}"
    )

    # --------------------------------------------------------
    # Create preprocessing object
    # --------------------------------------------------------

    data_preprocessor = DataPreprocessing(
        df=nova_pay_data,
        categorical_features=CATEGORICAL_FEATURES,
        binary_features=BINARY_FEATURES,
        numeric_features=NUMERIC_FEATURES,
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_df, test_df = (
        data_preprocessor
        .chronological_split(
            train_size=train_size
        )
    )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = (
        data_preprocessor
        .prepare_features_target(
            train_df,
            test_df,
        )
    )

    # --------------------------------------------------------
    # Build transformer
    # --------------------------------------------------------

    data_preprocessor.build_preprocessor()

    # --------------------------------------------------------
    # Fit training / transform testing
    # --------------------------------------------------------

    (
        X_train_processed,
        X_test_processed,
    ) = (
        data_preprocessor
        .transform_data(
            X_train,
            X_test,
        )
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        data_preprocessor
        .get_feature_names()
    )

    # --------------------------------------------------------
    # Save fitted transformer
    # --------------------------------------------------------

    preprocessor_path = (
        data_preprocessor
        .save_preprocessor()
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\nData Preprocessing Summary"
    )

    print(
        "-" * 45
    )

    print(
        f"Original model features: "
        f"{len(data_preprocessor.all_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    print(
        f"Binary features: "
        f"{len(BINARY_FEATURES)}"
    )

    print(
        f"Numeric features: "
        f"{len(NUMERIC_FEATURES)}"
    )

    print(
        f"Processed features: "
        f"{len(feature_names)}"
    )

    print(
        f"X_train: "
        f"{X_train_processed.shape}"
    )

    print(
        f"X_test: "
        f"{X_test_processed.shape}"
    )

    print(
        f"y_train: "
        f"{y_train.shape}"
    )

    print(
        f"y_test: "
        f"{y_test.shape}"
    )

    print(
        f"Preprocessor saved to: "
        f"{preprocessor_path}"
    )

    logger.info(
        "NovaPay data preprocessing pipeline "
        "completed successfully"
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        feature_names,
        data_preprocessor,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()