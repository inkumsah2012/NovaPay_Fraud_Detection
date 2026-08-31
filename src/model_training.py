import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DataPreprocessing,
    load_feature_engineered_data,
)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIRECTORY = PROJECT_ROOT / "models"

MODEL_BUNDLE_PATH = (
    MODEL_DIRECTORY
    / "random_forest_bundle.joblib"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RandomForestConfig:
    """
    Configuration for the NovaPay Random Forest model.
    """

    n_estimators: int = 100
    max_depth: int = 10
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1

    train_size: float = 0.80
    decision_threshold: float = 0.50


# ============================================================
# MODEL TRAINER
# ============================================================

class ModelTrainer:
    """
    Train the NovaPay Random Forest fraud-detection model.

    Responsibilities
    ----------------
    - Load feature-engineered data.
    - Create chronological training data.
    - Build the preprocessing + model pipeline.
    - Train using training data only.
    - Save the trained model bundle.

    Model evaluation is intentionally handled separately in
    model_evaluation.py.
    """

    def __init__(
        self,
        config: RandomForestConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RandomForestConfig()
        )

        self.model_pipeline: Pipeline | None = None

        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None

        self.training_start: pd.Timestamp | None = None
        self.training_end: pd.Timestamp | None = None

        self.feature_names: list[str] = []


    # ========================================================
    # 1. PREPARE TRAINING DATA
    # ========================================================

    def prepare_training_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load feature-engineered data and extract only the
        chronological training partition.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            X_train and y_train.
        """

        try:

            logger.info(
                "Preparing chronological training data"
            )

            df = load_feature_engineered_data()

            data_preprocessor = DataPreprocessing(
                df=df,
                categorical_features=CATEGORICAL_FEATURES,
                binary_features=BINARY_FEATURES,
                numeric_features=NUMERIC_FEATURES,
            )

            train_df, _ = (
                data_preprocessor
                .chronological_split(
                    train_size=self.config.train_size
                )
            )

            self.feature_names = (
                data_preprocessor.all_features
            )

            required_columns = (
                self.feature_names
                + [
                    data_preprocessor.target_column,
                    data_preprocessor.timestamp_column,
                ]
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in train_df.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Training dataset is missing required "
                    f"columns: {missing_columns}"
                )

            self.X_train = (
                train_df[
                    self.feature_names
                ]
                .copy()
            )

            self.y_train = (
                train_df[
                    data_preprocessor.target_column
                ]
                .astype("int8")
                .copy()
            )

            self.training_start = (
                train_df[
                    data_preprocessor.timestamp_column
                ].min()
            )

            self.training_end = (
                train_df[
                    data_preprocessor.timestamp_column
                ].max()
            )

            self._validate_training_data()

            logger.info(
                "Training data prepared successfully: "
                "%d observations, %d features",
                len(self.X_train),
                len(self.feature_names),
            )

            return (
                self.X_train,
                self.y_train,
            )

        except Exception:

            logger.exception(
                "Failed to prepare training data"
            )

            raise


    # ========================================================
    # 2. VALIDATE TRAINING DATA
    # ========================================================

    def _validate_training_data(
        self,
    ) -> None:
        """
        Validate the training predictors and target.
        """

        if (
            self.X_train is None
            or self.y_train is None
        ):
            raise RuntimeError(
                "Training data has not been prepared."
            )

        if self.X_train.empty:
            raise ValueError(
                "Training predictor dataset is empty."
            )

        if self.y_train.empty:
            raise ValueError(
                "Training target is empty."
            )

        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                "X_train and y_train contain different "
                "numbers of observations."
            )

        if self.y_train.isna().any():
            raise ValueError(
                "Training target contains missing values."
            )

        target_values = set(
            self.y_train.unique()
        )

        if not target_values.issubset({0, 1}):
            raise ValueError(
                "Training target must contain only 0 and 1."
            )

        if len(target_values) < 2:
            raise ValueError(
                "Training target must contain both "
                "legitimate and fraud observations."
            )


    # ========================================================
    # 3. BUILD MODEL PIPELINE
    # ========================================================

    def build_model_pipeline(
        self,
    ) -> Pipeline:
        """
        Build a single sklearn pipeline containing both
        preprocessing and the Random Forest classifier.
        """

        try:

            logger.info(
                "Building Random Forest model pipeline"
            )

            if self.X_train is None:
                raise RuntimeError(
                    "Training data must be prepared first."
                )

            data_preprocessor = DataPreprocessing(
                df=pd.concat(
                    [
                        self.X_train,
                        self.y_train.rename(
                            "is_fraud"
                        ),
                    ],
                    axis=1,
                ),
                categorical_features=CATEGORICAL_FEATURES,
                binary_features=BINARY_FEATURES,
                numeric_features=NUMERIC_FEATURES,
            )

            preprocessor = (
                data_preprocessor
                .build_preprocessor()
            )

            classifier = RandomForestClassifier(
                n_estimators=(
                    self.config.n_estimators
                ),
                max_depth=(
                    self.config.max_depth
                ),
                class_weight=(
                    self.config.class_weight
                ),
                random_state=(
                    self.config.random_state
                ),
                n_jobs=(
                    self.config.n_jobs
                ),
            )

            self.model_pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "classifier",
                        classifier,
                    ),
                ]
            )

            logger.info(
                "Random Forest pipeline built successfully"
            )

            return self.model_pipeline

        except Exception:

            logger.exception(
                "Failed to build model pipeline"
            )

            raise


    # ========================================================
    # 4. TRAIN MODEL
    # ========================================================

    def train(
        self,
    ) -> Pipeline:
        """
        Fit the complete preprocessing + model pipeline using
        training data only.
        """

        try:

            if (
                self.X_train is None
                or self.y_train is None
            ):
                self.prepare_training_data()

            if self.model_pipeline is None:
                self.build_model_pipeline()

            logger.info(
                "Starting Random Forest training"
            )

            self.model_pipeline.fit(
                self.X_train,
                self.y_train,
            )

            logger.info(
                "Random Forest trained successfully"
            )

            return self.model_pipeline

        except Exception:

            logger.exception(
                "Random Forest training failed"
            )

            raise


    # ========================================================
    # 5. SAVE MODEL BUNDLE
    # ========================================================

    def save_model(
        self,
        output_path: Path = MODEL_BUNDLE_PATH,
    ) -> Path:
        """
        Save the complete fitted pipeline and model metadata.

        The saved pipeline contains both preprocessing and the
        trained Random Forest classifier.
        """

        if self.model_pipeline is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        try:

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            model_bundle: dict[str, Any] = {
                "model": self.model_pipeline,

                "model_type":
                    "RandomForestClassifier",

                "target_column":
                    "is_fraud",

                "feature_names":
                    self.feature_names,

                "decision_threshold":
                    self.config.decision_threshold,

                "train_size":
                    self.config.train_size,

                "training_start":
                    str(self.training_start),

                "training_end":
                    str(self.training_end),

                "model_parameters":
                    asdict(self.config),
            }

            joblib.dump(
                model_bundle,
                output_path,
            )

            logger.info(
                "Model bundle saved successfully to %s",
                output_path,
            )

            return output_path

        except Exception:

            logger.exception(
                "Failed to save model bundle"
            )

            raise


    # ========================================================
    # 6. RUN TRAINING PIPELINE
    # ========================================================

    def run(
        self,
    ) -> Path:
        """
        Execute the complete model-training workflow.
        """

        logger.info(
            "Starting NovaPay model training pipeline"
        )

        self.prepare_training_data()

        self.build_model_pipeline()

        self.train()

        model_path = (
            self.save_model()
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "NOVAPAY RANDOM FOREST TRAINING COMPLETED"
        )

        print(
            "=" * 60
        )

        print(
            f"Training observations: "
            f"{len(self.X_train):,}"
        )

        print(
            f"Training features:     "
            f"{len(self.feature_names)}"
        )

        print(
            f"Fraud observations:    "
            f"{int(self.y_train.sum()):,}"
        )

        print(
            f"Training fraud rate:   "
            f"{self.y_train.mean():.3%}"
        )

        print(
            f"Training period:       "
            f"{self.training_start} "
            f"to {self.training_end}"
        )

        print(
            f"Model saved to:        "
            f"{model_path}"
        )

        print(
            "\nNo final test-set evaluation "
            "was performed during training."
        )

        logger.info(
            "NovaPay model training pipeline "
            "completed successfully"
        )

        return model_path


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def run_pipeline() -> Path:
    """
    Run the NovaPay model-training pipeline.
    """

    trainer = ModelTrainer()

    return trainer.run()


if __name__ == "__main__":
    run_pipeline()