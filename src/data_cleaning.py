import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.constant import Cleaned_Data
from src.data_ingestion import data_ingestion


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLEANING CLASS
# ============================================================

class DataCleaning:
    """
    Perform data quality checks and cleaning operations
    for the NovaPay fraud detection dataset.
    """

    NUMERIC_COLUMNS = [
        "amount_src",
        "amount_usd",
        "fee",
        "device_trust_score",
    ]

    NON_NEGATIVE_COLUMNS = [
        "amount_src",
        "amount_usd",
        "fee",
        "device_trust_score",
        "txn_velocity_1h",
        "txn_velocity_24h",
        "account_age_days",
        "chargeback_history_count",
    ]

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data cleaning pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw NovaPay transaction dataset.
        """
        self.df = df.copy()

    # ========================================================
    # 1. CONVERT DATA TYPES
    # ========================================================

    def convert_data_types(self) -> pd.DataFrame:
        """Convert selected columns to their expected data types."""

        try:
            logger.info("Starting data type conversion")

            if "timestamp" in self.df.columns:
                self.df["timestamp"] = pd.to_datetime(
                    self.df["timestamp"],
                    errors="coerce",
                    utc=True,
                )

            for column in self.NUMERIC_COLUMNS:
                if column in self.df.columns:
                    self.df[column] = pd.to_numeric(
                        self.df[column],
                        errors="coerce",
                    )

            logger.info("Data types successfully converted")

            print("\nData Information After Conversion:")
            self.df.info()

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while converting data types"
            )
            raise

    # ========================================================
    # 2. CALCULATE EXCHANGE RATE AND FILL AMOUNT_USD
    # ========================================================

    def fill_amount_usd_with_exchange_rate(
        self,
    ) -> pd.DataFrame:
        """
        Calculate average exchange rates by source currency and
        use them to fill missing USD transaction amounts.
        """

        try:
            logger.info("Starting exchange rate calculation")

            required_columns = {
                "amount_usd",
                "amount_src",
                "source_currency",
            }

            if not required_columns.issubset(self.df.columns):
                missing_columns = (
                    required_columns - set(self.df.columns)
                )

                logger.warning(
                    "Exchange-rate calculation skipped. "
                    "Missing columns: %s",
                    sorted(missing_columns),
                )

                return self.df

            valid_exchange_data = self.df.loc[
                self.df["amount_usd"].notna()
                & self.df["amount_src"].notna()
                & (self.df["amount_src"] > 0)
                & (self.df["amount_usd"] > 0)
            ].copy()

            valid_exchange_data["exchange_rate"] = (
                valid_exchange_data["amount_usd"]
                / valid_exchange_data["amount_src"]
            )

            exchange_rates = (
                valid_exchange_data
                .groupby("source_currency")["exchange_rate"]
                .mean()
                .to_dict()
            )

            logger.info(
                "Exchange rates calculated for %d currencies",
                len(exchange_rates),
            )

            print("\nExchange Rate Dictionary:")
            print(exchange_rates)

            missing_amount_mask = self.df["amount_usd"].isna()

            mapped_exchange_rates = (
                self.df["source_currency"]
                .map(exchange_rates)
            )

            self.df.loc[
                missing_amount_mask,
                "amount_usd",
            ] = (
                self.df.loc[
                    missing_amount_mask,
                    "amount_src",
                ]
                * mapped_exchange_rates.loc[
                    missing_amount_mask
                ]
            )

            remaining_missing = (
                self.df["amount_usd"].isna().sum()
            )

            logger.info(
                "Missing amount_usd values filled. "
                "Remaining missing amount_usd: %d",
                remaining_missing,
            )

            return self.df

        except Exception:
            logger.exception(
                "Error occurred during exchange-rate calculation"
            )
            raise

    # ========================================================
    # 3. HANDLE MISSING VALUES
    # ========================================================

    def handle_missing_values(self) -> pd.DataFrame:
        """Impute selected missing values using business rules."""

        try:
            logger.info("Starting missing value treatment")

            self._fill_missing_fee()
            self._fill_missing_ip_country()
            self._fill_missing_kyc_tier()
            self._fill_missing_device_trust_score()

            print("\nMissing Values After Imputation:")
            print(self.df.isna().sum())

            logger.info(
                "Initial missing value treatment completed"
            )

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while handling missing values"
            )
            raise

    def _fill_missing_fee(self) -> None:
        """Fill missing fees using channel median and overall median."""

        if "fee" not in self.df.columns:
            return

        if "channel" in self.df.columns:
            channel_median = (
                self.df
                .groupby("channel")["fee"]
                .transform("median")
            )

            self.df["fee"] = (
                self.df["fee"]
                .fillna(channel_median)
            )

        overall_median = self.df["fee"].median()

        self.df["fee"] = (
            self.df["fee"]
            .fillna(overall_median)
        )

        logger.info(
            "Missing fee values successfully handled"
        )

    def _fill_missing_ip_country(self) -> None:
        """Use home country as fallback for missing IP country."""

        required_columns = {
            "ip_country",
            "home_country",
        }

        if not required_columns.issubset(self.df.columns):
            return

        self.df["ip_country"] = (
            self.df["ip_country"]
            .fillna(self.df["home_country"])
        )

        logger.info(
            "Missing ip_country values filled using home_country"
        )

    def _fill_missing_kyc_tier(self) -> None:
        """Fill missing KYC tier values with the mode."""

        if "kyc_tier" not in self.df.columns:
            return

        mode_values = self.df["kyc_tier"].mode()

        mode_kyc = (
            mode_values.iloc[0]
            if not mode_values.empty
            else "standard"
        )

        self.df["kyc_tier"] = (
            self.df["kyc_tier"]
            .fillna(mode_kyc)
        )

        logger.info(
            "Missing kyc_tier values filled using mode"
        )

    def _fill_missing_device_trust_score(self) -> None:
        """
        Fill device trust scores using the median within
        new_device and kyc_tier groups, followed by the
        overall median as fallback.
        """

        if "device_trust_score" not in self.df.columns:
            return

        grouping_columns = {
            "new_device",
            "kyc_tier",
        }

        if grouping_columns.issubset(self.df.columns):
            group_median = (
                self.df
                .groupby(
                    ["new_device", "kyc_tier"]
                )["device_trust_score"]
                .transform("median")
            )

            self.df["device_trust_score"] = (
                self.df["device_trust_score"]
                .fillna(group_median)
            )

        overall_median = (
            self.df["device_trust_score"].median()
        )

        self.df["device_trust_score"] = (
            self.df["device_trust_score"]
            .fillna(overall_median)
        )

        logger.info(
            "Missing device_trust_score values "
            "successfully handled"
        )

    # ========================================================
    # 4. STANDARDIZE HOME COUNTRY
    # ========================================================

    def standardize_home_country(self) -> pd.DataFrame:
        """Standardize home-country text formatting."""

        return self._standardize_column(
            column="home_country",
            replacements=None,
            display_name="Home Country",
        )

    # ========================================================
    # 5. STANDARDIZE IP COUNTRY
    # ========================================================

    def standardize_ip_country(self) -> pd.DataFrame:
        """Standardize IP-country values."""

        replacements = {
            "unknown": np.nan,
            "nan": np.nan,
        }

        return self._standardize_column(
            column="ip_country",
            replacements=replacements,
            display_name="IP Country",
        )

    # ========================================================
    # 6. STANDARDIZE CHANNEL
    # ========================================================

    def standardize_channel(self) -> pd.DataFrame:
        """Standardize transaction channel categories."""

        replacements = {
            "weeb": "web",
            "mobille": "mobile",
            "unknown": np.nan,
            "nan": np.nan,
        }

        return self._standardize_column(
            column="channel",
            replacements=replacements,
            display_name="Channel",
        )

    # ========================================================
    # 7. STANDARDIZE KYC TIER
    # ========================================================

    def standardize_kyc_tier(self) -> pd.DataFrame:
        """Standardize KYC tier categories."""

        replacements = {
            "standrd": "standard",
            "enhancd": "enhanced",
            "unknown": np.nan,
            "nan": np.nan,
        }

        return self._standardize_column(
            column="kyc_tier",
            replacements=replacements,
            display_name="KYC Tier",
        )

    def _standardize_column(
        self,
        column: str,
        replacements: dict | None,
        display_name: str,
    ) -> pd.DataFrame:
        """
        Standardize a categorical text column by converting
        values to lowercase, stripping whitespace, and applying
        optional replacement rules.
        """

        try:
            if column not in self.df.columns:
                logger.warning(
                    "%s column not found; standardization skipped",
                    column,
                )
                return self.df

            self.df[column] = (
                self.df[column]
                .astype("string")
                .str.lower()
                .str.strip()
            )

            if replacements:
                self.df[column] = (
                    self.df[column]
                    .replace(replacements)
                )

            logger.info(
                "%s successfully standardized",
                column,
            )

            print(f"\nUnique {display_name} Values:")
            print(self.df[column].unique())

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while standardizing %s",
                column,
            )
            raise

    # ========================================================
    # 8. CHECK AND HANDLE NEGATIVE VALUES
    # ========================================================

    def handle_negative_values(self) -> pd.DataFrame:
        """
        Identify negative numeric values and remove records
        containing negative values in columns where negatives
        are logically invalid.
        """

        try:
            logger.info(
                "Checking numerical columns for negative values"
            )

            numerical_columns = (
                self.df
                .select_dtypes(include=np.number)
                .columns
            )

            print("\nNegative Values Before Cleaning:")

            for column in numerical_columns:
                if column == "is_fraud":
                    continue

                negative_count = (
                    self.df[column] < 0
                ).sum()

                print(
                    f"{column}: {negative_count}"
                )

            existing_columns = [
                column
                for column in self.NON_NEGATIVE_COLUMNS
                if column in self.df.columns
            ]

            rows_before = len(self.df)

            valid_mask = pd.Series(
                True,
                index=self.df.index,
            )

            for column in existing_columns:
                valid_mask &= self.df[column] >= 0

            self.df = (
                self.df.loc[valid_mask]
                .copy()
            )

            rows_removed = (
                rows_before - len(self.df)
            )

            logger.info(
                "%d rows with invalid negative values removed",
                rows_removed,
            )

            print("\nNegative Values After Cleaning:")

            for column in existing_columns:
                negative_count = (
                    self.df[column] < 0
                ).sum()

                print(
                    f"{column}: {negative_count}"
                )

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while handling negative values"
            )
            raise

    # ========================================================
    # 9. CHECK AMOUNT USD / AMOUNT SOURCE RATIO
    # ========================================================

    def check_amount_ratio(self) -> pd.DataFrame:
        """Display descriptive statistics for USD/source ratio."""

        try:
            required_columns = {
                "amount_usd",
                "amount_src",
            }

            if not required_columns.issubset(self.df.columns):
                return self.df

            valid_mask = (
                self.df["amount_src"] > 0
            )

            amount_ratio = (
                self.df.loc[
                    valid_mask,
                    "amount_usd",
                ]
                / self.df.loc[
                    valid_mask,
                    "amount_src",
                ]
            )

            print(
                "\nAmount USD / Amount Source Ratio Statistics:"
            )

            print(
                amount_ratio.describe()
            )

            logger.info(
                "Amount exchange ratio successfully checked"
            )

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while checking amount ratio"
            )
            raise

    # ========================================================
    # 10. CHECK FUTURE TIMESTAMPS
    # ========================================================

    def check_future_timestamps(self) -> pd.DataFrame:
        """Identify transactions with timestamps in the future."""

        try:
            if "timestamp" not in self.df.columns:
                return self.df

            current_time = pd.Timestamp.now(
                tz="UTC"
            )

            future_mask = (
                self.df["timestamp"] > current_time
            )

            future_records = (
                self.df.loc[future_mask]
            )

            print(
                "\nNumber of Future Timestamp Records:"
            )

            print(
                len(future_records)
            )

            if not future_records.empty:
                print(
                    "\nFuture Timestamp Records:"
                )

                print(
                    future_records[
                        ["timestamp"]
                    ].head()
                )

            logger.info(
                "Future timestamp validation completed"
            )

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while checking future timestamps"
            )
            raise

    # ========================================================
    # 11. DROP REMAINING MISSING VALUES
    # ========================================================

    def drop_remaining_missing_values(
        self,
    ) -> pd.DataFrame:
        """Remove records containing unresolved missing values."""

        try:
            logger.info(
                "Dropping remaining missing values"
            )

            rows_before = len(self.df)

            self.df = (
                self.df
                .dropna()
                .reset_index(drop=True)
            )

            rows_removed = (
                rows_before - len(self.df)
            )

            logger.info(
                "%d rows containing remaining missing "
                "values were dropped",
                rows_removed,
            )

            print(
                "\nMissing Values After Final Cleaning:"
            )

            print(
                self.df.isna().sum()
            )

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while dropping missing values"
            )
            raise

    # ========================================================
    # 12. CHECK FRAUD DISTRIBUTION
    # ========================================================

    def check_fraud_distribution(
        self,
        stage: str = "Current Dataset",
    ) -> pd.DataFrame:
        """Display fraud counts and class proportions."""

        try:
            if "is_fraud" not in self.df.columns:
                logger.warning(
                    "is_fraud column not found"
                )
                return self.df

            fraud_counts = (
                self.df["is_fraud"]
                .value_counts()
                .sort_index()
            )

            fraud_percentages = (
                self.df["is_fraud"]
                .value_counts(normalize=True)
                .sort_index()
                .mul(100)
            )

            print(
                f"\nFraud Distribution - {stage}:"
            )

            print(
                "\nFraud Counts:"
            )
            print(fraud_counts)

            print(
                "\nFraud Percentage:"
            )
            print(fraud_percentages)

            logger.info(
                "Fraud distribution checked for %s",
                stage,
            )

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while checking "
                "fraud distribution"
            )
            raise

    # ========================================================
    # 13. SAVE CLEANED DATA
    # ========================================================

    def save_cleaned_data(self) -> pd.DataFrame:
        """Save the cleaned dataset to the configured path."""

        try:
            output_path = Path(Cleaned_Data)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.df.to_csv(
                output_path,
                index=False,
            )

            logger.info(
                "Cleaned dataset saved to %s",
                output_path,
            )

            print(
                "\nCleaned Dataset Saved To:"
            )

            print(output_path)

            return self.df

        except Exception:
            logger.exception(
                "Error occurred while saving cleaned data"
            )
            raise


# ============================================================
# DATA CLEANING PIPELINE
# ============================================================

def run_pipeline() -> pd.DataFrame:
    """
    Execute the complete NovaPay data cleaning pipeline.

    Returns
    -------
    pd.DataFrame
        Final cleaned transaction dataset.
    """

    logger.info(
        "Starting NovaPay data cleaning pipeline"
    )

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    nova_pay_data = data_ingestion()

    print(
        f"\nOriginal Dataset Shape: "
        f"{nova_pay_data.shape}"
    )

    cleaner = DataCleaning(
        nova_pay_data
    )

    # --------------------------------------------------------
    # Fraud distribution before cleaning
    # --------------------------------------------------------

    cleaner.check_fraud_distribution(
        stage="Before Cleaning"
    )

    # --------------------------------------------------------
    # Execute cleaning stages
    # --------------------------------------------------------

    cleaner.convert_data_types()

    cleaner.fill_amount_usd_with_exchange_rate()

    cleaner.handle_missing_values()

    cleaner.standardize_home_country()

    cleaner.standardize_ip_country()

    cleaner.standardize_channel()

    cleaner.standardize_kyc_tier()

    cleaner.handle_negative_values()

    cleaner.check_amount_ratio()

    cleaner.check_future_timestamps()

    cleaner.drop_remaining_missing_values()

    # --------------------------------------------------------
    # Fraud distribution after cleaning
    # --------------------------------------------------------

    cleaner.check_fraud_distribution(
        stage="After Cleaning"
    )

    # --------------------------------------------------------
    # Save cleaned data
    # --------------------------------------------------------

    cleaned_data = (
        cleaner.save_cleaned_data()
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\nFinal Cleaned Dataset:"
    )

    print(
        cleaned_data.head()
    )

    print(
        f"\nFinal Dataset Shape: "
        f"{cleaned_data.shape}"
    )

    print(
        "\nFinal Dataset Information:"
    )

    cleaned_data.info()

    logger.info(
        "NovaPay data cleaning pipeline "
        "successfully completed"
    )

    return cleaned_data


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()