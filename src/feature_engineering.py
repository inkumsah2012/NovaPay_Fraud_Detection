import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.constant import Cleaned_Data


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FEATURE ENGINEERING CLASS
# ============================================================

class FeatureEngineering:
    """
    Create model-ready features for the NovaPay fraud
    detection pipeline.

    Target-based fraud analysis is intentionally kept separate
    from feature creation to reduce the risk of target leakage.
    """

    # --------------------------------------------------------
    # Threshold configuration
    #
    # These values should be treated as fixed business/model
    # configuration once selected. They should not be
    # recalculated using the test dataset.
    # --------------------------------------------------------

    NIGHT_START_HOUR = 3
    NIGHT_END_HOUR = 7

    VERY_NEW_ACCOUNT_DAYS = 30
    NEW_ACCOUNT_DAYS = 90

    VELOCITY_BURST_THRESHOLD = 3
    HIGH_AMOUNT_THRESHOLD = 2_000.0
    HIGH_IP_RISK_THRESHOLD = 0.8
    LOW_DEVICE_TRUST_THRESHOLD = 0.5

    # --------------------------------------------------------
    # Final model feature definitions
    # --------------------------------------------------------

    CATEGORICAL_FEATURES = [
        "channel",
        "kyc_tier",
        "home_country",
        "source_currency",
        "dest_currency",
        "ip_country",
        "new_device",
        "location_mismatch",
        "transfer_corridor",
        "day_of_week",
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

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned NovaPay transaction dataset.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Input DataFrame is empty."
            )

        # Work on a copy to avoid changing the original
        # DataFrame outside this class.
        self.df = df.copy()

    # ========================================================
    # INTERNAL VALIDATION
    # ========================================================

    def _require_columns(
        self,
        columns: list[str],
        operation: str,
    ) -> None:
        """
        Validate that required columns exist.

        Parameters
        ----------
        columns : list[str]
            Required column names.

        operation : str
            Description of the operation requiring them.

        Raises
        ------
        ValueError
            If one or more required columns are missing.
        """

        missing_columns = [
            column
            for column in columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{operation} cannot be completed. "
                f"Missing columns: {missing_columns}"
            )

    # ========================================================
    # 1. CREATE TRANSFER CORRIDOR
    # ========================================================

    def create_transfer_corridor(
        self,
    ) -> pd.DataFrame:
        """
        Create a transaction corridor using source and
        destination currencies.

        Example
        -------
        USD + CAD -> USD_to_CAD
        """

        try:
            logger.info(
                "Creating transfer corridor feature"
            )

            required_columns = [
                "source_currency",
                "dest_currency",
            ]

            self._require_columns(
                required_columns,
                operation="Transfer corridor creation",
            )

            self.df["transfer_corridor"] = (
                self.df["source_currency"]
                .astype("string")
                .str.strip()
                .str.upper()
                + "_to_"
                + self.df["dest_currency"]
                .astype("string")
                .str.strip()
                .str.upper()
            )

            logger.info(
                "Transfer corridor feature created successfully"
            )

            return self.df

        except Exception:
            logger.exception(
                "Failed to create transfer corridor feature"
            )
            raise

    # ========================================================
    # 2. CREATE TIME FEATURES
    # ========================================================

    def create_time_features(
        self,
    ) -> pd.DataFrame:
        """
        Create transaction-time features from timestamp.

        Features created
        ----------------
        time_of_day
            Hour of transaction from 0 to 23.

        day_of_week
            Monday=0 through Sunday=6.

        is_weekend
            Binary weekend indicator.
        """

        try:
            logger.info(
                "Creating transaction time features"
            )

            self._require_columns(
                ["timestamp"],
                operation="Time feature creation",
            )

            timestamps = pd.to_datetime(
                self.df["timestamp"],
                errors="coerce",
                utc=True,
            )

            invalid_timestamps = (
                timestamps.isna().sum()
            )

            if invalid_timestamps > 0:
                raise ValueError(
                    f"{invalid_timestamps} timestamp values "
                    "could not be converted to datetime."
                )

            self.df["timestamp"] = timestamps

            self.df["time_of_day"] = (
                self.df["timestamp"].dt.hour
            )

            self.df["day_of_week"] = (
                self.df["timestamp"].dt.dayofweek
            )

            self.df["is_weekend"] = (
                self.df["day_of_week"]
                .ge(5)
                .astype("int8")
            )

            logger.info(
                "Transaction time features created successfully"
            )

            return self.df

        except Exception:
            logger.exception(
                "Failed to create transaction time features"
            )
            raise

    # ========================================================
    # 3. CREATE THRESHOLD FEATURES
    # ========================================================

    def create_threshold_features(
        self,
    ) -> pd.DataFrame:
        """
        Create rule-based fraud-risk indicators.

        These thresholds must remain fixed once selected.
        They should not be recalculated using the test set.
        """

        try:
            logger.info(
                "Creating threshold-based features"
            )

            required_columns = [
                "time_of_day",
                "account_age_days",
                "txn_velocity_1h",
                "amount_usd",
                "ip_risk_score",
                "device_trust_score",
            ]

            self._require_columns(
                required_columns,
                operation="Threshold feature creation",
            )

            # ------------------------------------------------
            # Night transaction
            # ------------------------------------------------

            self.df["night_hours"] = (
                self.df["time_of_day"]
                .between(
                    self.NIGHT_START_HOUR,
                    self.NIGHT_END_HOUR,
                    inclusive="both",
                )
                .astype("int8")
            )

            # ------------------------------------------------
            # Account age
            # ------------------------------------------------

            self.df["account_very_new"] = (
                self.df["account_age_days"]
                .lt(self.VERY_NEW_ACCOUNT_DAYS)
                .astype("int8")
            )

            self.df["account_new"] = (
                self.df["account_age_days"]
                .ge(self.VERY_NEW_ACCOUNT_DAYS)
                & self.df["account_age_days"]
                .lt(self.NEW_ACCOUNT_DAYS)
            ).astype("int8")

            # ------------------------------------------------
            # Transaction velocity
            # ------------------------------------------------

            self.df["velocity_burst"] = (
                self.df["txn_velocity_1h"]
                .ge(self.VELOCITY_BURST_THRESHOLD)
                .astype("int8")
            )

            # ------------------------------------------------
            # Transaction amount
            # ------------------------------------------------

            self.df["amount_high"] = (
                self.df["amount_usd"]
                .ge(self.HIGH_AMOUNT_THRESHOLD)
                .astype("int8")
            )

            # ------------------------------------------------
            # IP risk
            # ------------------------------------------------

            self.df["ip_high_risk"] = (
                self.df["ip_risk_score"]
                .ge(self.HIGH_IP_RISK_THRESHOLD)
                .astype("int8")
            )

            # ------------------------------------------------
            # Device trust
            # ------------------------------------------------

            self.df["device_low_trust"] = (
                self.df["device_trust_score"]
                .lt(self.LOW_DEVICE_TRUST_THRESHOLD)
                .astype("int8")
            )

            logger.info(
                "Threshold-based features created successfully"
            )

            return self.df

        except Exception:
            logger.exception(
                "Failed to create threshold-based features"
            )
            raise

    # ========================================================
    # 4. TARGET-BASED FEATURE ANALYSIS
    # ========================================================

    def analyze_fraud_patterns(
        self,
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """
        Analyse fraud behaviour across selected variables.

        IMPORTANT
        ---------
        This method is for exploratory/model-development
        analysis only.

        The calculated fraud rates are NOT added back to the
        model dataset because doing so before train/test
        separation could cause target leakage.

        Returns
        -------
        dict
            Collection of fraud analysis tables.
        """

        try:
            logger.info(
                "Starting fraud-pattern analysis"
            )

            self._require_columns(
                ["is_fraud"],
                operation="Fraud-pattern analysis",
            )

            results: dict[
                str,
                pd.DataFrame | pd.Series
            ] = {}

            # ------------------------------------------------
            # Transfer corridor
            # ------------------------------------------------

            if "transfer_corridor" in self.df.columns:

                corridor_analysis = (
                    self.df
                    .groupby(
                        "transfer_corridor",
                        observed=False,
                    )
                    .agg(
                        transaction_count=(
                            "is_fraud",
                            "size",
                        ),
                        fraud_count=(
                            "is_fraud",
                            "sum",
                        ),
                        fraud_rate=(
                            "is_fraud",
                            "mean",
                        ),
                    )
                    .sort_values(
                        "fraud_rate",
                        ascending=False,
                    )
                )

                results[
                    "fraud_by_transfer_corridor"
                ] = corridor_analysis

            # ------------------------------------------------
            # Time of day
            # ------------------------------------------------

            if "time_of_day" in self.df.columns:

                results[
                    "fraud_by_time_of_day"
                ] = (
                    self.df
                    .groupby("time_of_day")["is_fraud"]
                    .agg(
                        [
                            "count",
                            "sum",
                            "mean",
                        ]
                    )
                    .rename(
                        columns={
                            "count": "transaction_count",
                            "sum": "fraud_count",
                            "mean": "fraud_rate",
                        }
                    )
                )

            # ------------------------------------------------
            # Day of week
            # ------------------------------------------------

            if "timestamp" in self.df.columns:

                day_name = (
                    self.df["timestamp"]
                    .dt.day_name()
                )

                day_analysis = (
                    pd.DataFrame(
                        {
                            "day_name": day_name,
                            "is_fraud": self.df["is_fraud"],
                        }
                    )
                    .groupby("day_name")["is_fraud"]
                    .agg(
                        [
                            "count",
                            "sum",
                            "mean",
                        ]
                    )
                    .rename(
                        columns={
                            "count": "transaction_count",
                            "sum": "fraud_count",
                            "mean": "fraud_rate",
                        }
                    )
                )

                day_order = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]

                results[
                    "fraud_by_day_of_week"
                ] = day_analysis.reindex(
                    day_order
                )

            # ------------------------------------------------
            # Account age
            # ------------------------------------------------

            if "account_age_days" in self.df.columns:

                account_age_bucket = pd.cut(
                    self.df["account_age_days"],
                    bins=[
                        0,
                        90,
                        365,
                        730,
                        np.inf,
                    ],
                    labels=[
                        "0-89 days",
                        "90-364 days",
                        "365-729 days",
                        "730+ days",
                    ],
                    right=False,
                )

                account_age_analysis = (
                    pd.DataFrame(
                        {
                            "account_age_bucket":
                                account_age_bucket,
                            "is_fraud":
                                self.df["is_fraud"],
                        }
                    )
                    .groupby(
                        "account_age_bucket",
                        observed=False,
                    )["is_fraud"]
                    .agg(
                        [
                            "count",
                            "sum",
                            "mean",
                        ]
                    )
                    .rename(
                        columns={
                            "count": "transaction_count",
                            "sum": "fraud_count",
                            "mean": "fraud_rate",
                        }
                    )
                )

                results[
                    "fraud_by_account_age"
                ] = account_age_analysis

            # ------------------------------------------------
            # Transaction amount
            # ------------------------------------------------

            if "amount_usd" in self.df.columns:

                amount_bucket = pd.cut(
                    self.df["amount_usd"],
                    bins=[
                        0,
                        100,
                        500,
                        1_000,
                        2_000,
                        5_000,
                        np.inf,
                    ],
                    labels=[
                        "<$100",
                        "$100-<$500",
                        "$500-<$1k",
                        "$1k-<$2k",
                        "$2k-<$5k",
                        "$5k+",
                    ],
                    right=False,
                )

                amount_analysis = (
                    pd.DataFrame(
                        {
                            "amount_bucket":
                                amount_bucket,
                            "is_fraud":
                                self.df["is_fraud"],
                        }
                    )
                    .groupby(
                        "amount_bucket",
                        observed=False,
                    )["is_fraud"]
                    .agg(
                        [
                            "count",
                            "sum",
                            "mean",
                        ]
                    )
                    .rename(
                        columns={
                            "count": "transaction_count",
                            "sum": "fraud_count",
                            "mean": "fraud_rate",
                        }
                    )
                )

                results[
                    "fraud_by_amount"
                ] = amount_analysis

            # ------------------------------------------------
            # IP risk score
            # ------------------------------------------------

            if "ip_risk_score" in self.df.columns:

                ip_risk_bucket = pd.cut(
                    self.df["ip_risk_score"],
                    bins=[
                        -np.inf,
                        0.3,
                        0.5,
                        0.7,
                        0.8,
                        np.inf,
                    ],
                    labels=[
                        "<0.3",
                        "0.3-<0.5",
                        "0.5-<0.7",
                        "0.7-<0.8",
                        "0.8+",
                    ],
                    right=False,
                )

                ip_risk_analysis = (
                    pd.DataFrame(
                        {
                            "ip_risk_bucket":
                                ip_risk_bucket,
                            "is_fraud":
                                self.df["is_fraud"],
                        }
                    )
                    .groupby(
                        "ip_risk_bucket",
                        observed=False,
                    )["is_fraud"]
                    .agg(
                        [
                            "count",
                            "sum",
                            "mean",
                        ]
                    )
                    .rename(
                        columns={
                            "count": "transaction_count",
                            "sum": "fraud_count",
                            "mean": "fraud_rate",
                        }
                    )
                )

                results[
                    "fraud_by_ip_risk"
                ] = ip_risk_analysis

            # ------------------------------------------------
            # Device trust
            # ------------------------------------------------

            if "device_trust_score" in self.df.columns:

                device_trust_bucket = pd.cut(
                    self.df["device_trust_score"],
                    bins=[
                        -np.inf,
                        0.3,
                        0.5,
                        0.7,
                        0.8,
                        np.inf,
                    ],
                    labels=[
                        "<0.3",
                        "0.3-<0.5",
                        "0.5-<0.7",
                        "0.7-<0.8",
                        "0.8+",
                    ],
                    right=False,
                )

                device_analysis = (
                    pd.DataFrame(
                        {
                            "device_trust_bucket":
                                device_trust_bucket,
                            "is_fraud":
                                self.df["is_fraud"],
                        }
                    )
                    .groupby(
                        "device_trust_bucket",
                        observed=False,
                    )["is_fraud"]
                    .agg(
                        [
                            "count",
                            "sum",
                            "mean",
                        ]
                    )
                    .rename(
                        columns={
                            "count": "transaction_count",
                            "sum": "fraud_count",
                            "mean": "fraud_rate",
                        }
                    )
                )

                results[
                    "fraud_by_device_trust"
                ] = device_analysis

            logger.info(
                "Fraud-pattern analysis completed"
            )

            return results

        except Exception:
            logger.exception(
                "Fraud-pattern analysis failed"
            )
            raise

    # ========================================================
    # 5. VALIDATE ENGINEERED FEATURES
    # ========================================================

    def validate_features(
        self,
    ) -> None:
        """
        Validate engineered features before persistence.
        """

        try:
            logger.info(
                "Validating engineered features"
            )

            required_features = [
                "transfer_corridor",
                "time_of_day",
                "day_of_week",
                "is_weekend",
                "night_hours",
                "account_very_new",
                "account_new",
                "velocity_burst",
                "amount_high",
                "ip_high_risk",
                "device_low_trust",
            ]

            self._require_columns(
                required_features,
                operation="Feature validation",
            )

            missing_counts = (
                self.df[
                    required_features
                ]
                .isna()
                .sum()
            )

            problematic_columns = (
                missing_counts[
                    missing_counts > 0
                ]
            )

            if not problematic_columns.empty:
                raise ValueError(
                    "Missing values detected in engineered "
                    f"features:\n{problematic_columns}"
                )

            binary_features = [
                "is_weekend",
                "night_hours",
                "account_very_new",
                "account_new",
                "velocity_burst",
                "amount_high",
                "ip_high_risk",
                "device_low_trust",
            ]

            for feature in binary_features:

                values = set(
                    self.df[feature]
                    .dropna()
                    .unique()
                )

                if not values.issubset(
                    {0, 1}
                ):
                    raise ValueError(
                        f"{feature} contains non-binary "
                        f"values: {values}"
                    )

            logger.info(
                "Engineered feature validation completed"
            )

        except Exception:
            logger.exception(
                "Engineered feature validation failed"
            )
            raise

    # ========================================================
    # 6. DEFINE MODEL FEATURE SETS
    # ========================================================

    def get_feature_sets(
        self,
    ) -> tuple[
        list[str],
        list[str],
        list[str],
    ]:
        """
        Return the categorical, numeric and complete
        model feature lists.
        """

        categorical_features = [
            feature
            for feature
            in self.CATEGORICAL_FEATURES
            if feature in self.df.columns
        ]

        numeric_features = [
            feature
            for feature
            in self.NUMERIC_FEATURES
            if feature in self.df.columns
        ]

        all_features = (
            categorical_features
            + numeric_features
        )

        logger.info(
            "Feature sets defined: %d categorical, "
            "%d numeric, %d total",
            len(categorical_features),
            len(numeric_features),
            len(all_features),
        )

        return (
            categorical_features,
            numeric_features,
            all_features,
        )

    # ========================================================
    # 7. SAVE FEATURE-ENGINEERED DATA
    # ========================================================

    def save(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save feature-engineered data to CSV.

        Parameters
        ----------
        output_path : str | Path | None
            Optional output location.

        Returns
        -------
        Path
            Path of the saved dataset.
        """

        try:
            if output_path is None:
                output_path = (
                    Path(Cleaned_Data).parent
                    / "feature_engineered_data.csv"
                )

            output_path = Path(
                output_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.df.to_csv(
                output_path,
                index=False,
            )

            logger.info(
                "Feature-engineered dataset saved to %s",
                output_path,
            )

            return output_path

        except Exception:
            logger.exception(
                "Failed to save feature-engineered data"
            )
            raise

    # ========================================================
    # 8. RUN FEATURE TRANSFORMATION
    # ========================================================

    def transform(
        self,
    ) -> pd.DataFrame:
        """
        Execute production feature transformations.

        This method does not perform target-based analysis.
        """

        logger.info(
            "Starting NovaPay feature transformation"
        )

        self.create_transfer_corridor()
        self.create_time_features()
        self.create_threshold_features()

        self.validate_features()

        logger.info(
            "NovaPay feature transformation completed"
        )

        return self.df.copy()


# ============================================================
# LOAD CLEANED DATA
# ============================================================

def load_cleaned_data(
    path: str | Path = Cleaned_Data,
) -> pd.DataFrame:
    """
    Load the cleaned NovaPay dataset.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}"
        )

    logger.info(
        "Loading cleaned dataset from %s",
        input_path,
    )

    df = pd.read_csv(
        input_path,
    )

    if df.empty:
        raise ValueError(
            "Cleaned dataset is empty."
        )

    logger.info(
        "Cleaned dataset loaded successfully: "
        "%d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )

    return df


# ============================================================
# DISPLAY ANALYSIS
# ============================================================

def display_fraud_analysis(
    analysis: dict[str, Any],
) -> None:
    """
    Display exploratory fraud analysis results.

    This function is for development/EDA only.
    """

    if not analysis:
        return

    display_names = {
        "fraud_by_transfer_corridor":
            "Fraud by Transfer Corridor",

        "fraud_by_time_of_day":
            "Fraud by Time of Day",

        "fraud_by_day_of_week":
            "Fraud by Day of Week",

        "fraud_by_account_age":
            "Fraud by Account Age",

        "fraud_by_amount":
            "Fraud by Transaction Amount",

        "fraud_by_ip_risk":
            "Fraud by IP Risk",

        "fraud_by_device_trust":
            "Fraud by Device Trust",
    }

    for name, result in analysis.items():

        title = display_names.get(
            name,
            name,
        )

        print(
            f"\n{'=' * 60}"
        )

        print(title.upper())

        print(
            "=" * 60
        )

        print(result)


# ============================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================

def run_pipeline(
    include_analysis: bool = True,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:
    """
    Execute the complete NovaPay feature engineering pipeline.

    Parameters
    ----------
    include_analysis : bool, default=True
        Run exploratory target-based fraud analysis.

        Set to False for production scoring or datasets
        without the target variable.

    Returns
    -------
    tuple
        Feature-engineered DataFrame,
        categorical feature list,
        numeric feature list,
        all model features.
    """

    logger.info(
        "Starting NovaPay feature engineering pipeline"
    )

    # --------------------------------------------------------
    # Load cleaned dataset
    # --------------------------------------------------------

    nova_pay_data = (
        load_cleaned_data()
    )

    print(
        f"\nCleaned Dataset Shape: "
        f"{nova_pay_data.shape}"
    )

    # --------------------------------------------------------
    # Create feature engineering object
    # --------------------------------------------------------

    feature_engineer = (
        FeatureEngineering(
            nova_pay_data
        )
    )

    # --------------------------------------------------------
    # Production feature transformation
    # --------------------------------------------------------

    engineered_data = (
        feature_engineer.transform()
    )

    # --------------------------------------------------------
    # Optional exploratory fraud analysis
    # --------------------------------------------------------

    if (
        include_analysis
        and "is_fraud"
        in engineered_data.columns
    ):
        analysis = (
            feature_engineer
            .analyze_fraud_patterns()
        )

        display_fraud_analysis(
            analysis
        )

    # --------------------------------------------------------
    # Feature lists
    # --------------------------------------------------------

    (
        categorical_features,
        numeric_features,
        all_features,
    ) = (
        feature_engineer
        .get_feature_sets()
    )

    # --------------------------------------------------------
    # Save engineered dataset
    # --------------------------------------------------------

    output_path = (
        feature_engineer.save()
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\nFeature Engineering Summary"
    )

    print(
        "-" * 40
    )

    print(
        f"Input rows: "
        f"{len(nova_pay_data):,}"
    )

    print(
        f"Output rows: "
        f"{len(engineered_data):,}"
    )

    print(
        f"Output columns: "
        f"{engineered_data.shape[1]}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    print(
        f"Numeric features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Total model features: "
        f"{len(all_features)}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    logger.info(
        "NovaPay feature engineering pipeline "
        "completed successfully"
    )

    return (
        engineered_data,
        categorical_features,
        numeric_features,
        all_features,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline(
        include_analysis=True
    )