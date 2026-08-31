import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
)
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

BEST_PARAMS_PATH = (
    MODEL_DIRECTORY
    / "random_forest_best_params.json"
)


# ============================================================
# TUNING CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TuningConfig:
    """
    Configuration for Random Forest hyperparameter tuning.
    """

    train_size: float = 0.80

    n_iter: int = 25

    cv_splits: int = 3

    random_state: int = 42

    n_jobs: int = -1

    scoring: str = "average_precision"


# ============================================================
# RANDOM FOREST TUNER
# ============================================================

class RandomForestTuner:
    """
    Tune Random Forest hyperparameters using only the
    chronological training partition.

    The final test dataset is never used during tuning.
    """

    def __init__(
        self,
        config: TuningConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else TuningConfig()
        )

        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None

        self.pipeline: Pipeline | None = None

        self.search: RandomizedSearchCV | None = None

        self.best_params: dict[str, Any] = {}


    # ========================================================
    # 1. PREPARE TRAINING DATA
    # ========================================================

    def prepare_training_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load the feature-engineered dataset and retain only
        the historical training partition.
        """

        logger.info(
            "Preparing data for hyperparameter tuning"
        )

        df = load_feature_engineered_data()

        preprocessor_config = DataPreprocessing(
            df=df,
            categorical_features=CATEGORICAL_FEATURES,
            binary_features=BINARY_FEATURES,
            numeric_features=NUMERIC_FEATURES,
        )

        train_df, _ = (
            preprocessor_config
            .chronological_split(
                train_size=self.config.train_size
            )
        )

        self.X_train = (
            train_df[
                preprocessor_config.all_features
            ]
            .copy()
        )

        self.y_train = (
            train_df[
                preprocessor_config.target_column
            ]
            .astype("int8")
            .copy()
        )

        if self.X_train.empty:
            raise ValueError(
                "Training dataset is empty."
            )

        if self.y_train.nunique() != 2:
            raise ValueError(
                "Training target must contain "
                "both legitimate and fraud classes."
            )

        logger.info(
            "Tuning dataset prepared: "
            "%d rows and %d features",
            len(self.X_train),
            self.X_train.shape[1],
        )

        return (
            self.X_train,
            self.y_train,
        )


    # ========================================================
    # 2. BUILD MODEL PIPELINE
    # ========================================================

    def build_pipeline(
        self,
    ) -> Pipeline:
        """
        Build preprocessing and Random Forest into one
        sklearn Pipeline.

        This ensures preprocessing is fitted independently
        within every cross-validation fold.
        """

        if self.X_train is None:
            raise RuntimeError(
                "Training data must be prepared first."
            )

        tuning_df = pd.concat(
            [
                self.X_train,
                self.y_train.rename("is_fraud"),
            ],
            axis=1,
        )

        preprocessing = DataPreprocessing(
            df=tuning_df,
            categorical_features=CATEGORICAL_FEATURES,
            binary_features=BINARY_FEATURES,
            numeric_features=NUMERIC_FEATURES,
        )

        preprocessor = (
            preprocessing
            .build_preprocessor()
        )

        classifier = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        self.pipeline = Pipeline(
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

        return self.pipeline


    # ========================================================
    # 3. HYPERPARAMETER SEARCH SPACE
    # ========================================================

    @staticmethod
    def parameter_distribution(
    ) -> dict[str, list[Any]]:
        """
        Define candidate Random Forest hyperparameters.

        Pipeline parameter names use the
        'classifier__' prefix.
        """

        return {
            "classifier__n_estimators": [
                100,
                200,
                300,
                500,
            ],

            "classifier__max_depth": [
                None,
                10,
                20,
                30,
            ],

            "classifier__min_samples_split": [
                2,
                5,
                10,
            ],

            "classifier__min_samples_leaf": [
                1,
                2,
                4,
            ],

            "classifier__max_features": [
                "sqrt",
                "log2",
            ],

            "classifier__class_weight": [
                "balanced",
                "balanced_subsample",
            ],
        }


    # ========================================================
    # 4. RUN RANDOMIZED SEARCH
    # ========================================================

    def tune(
        self,
    ) -> RandomizedSearchCV:
        """
        Run randomized hyperparameter search using
        chronological cross-validation.
        """

        try:

            if (
                self.X_train is None
                or self.y_train is None
            ):
                self.prepare_training_data()

            if self.pipeline is None:
                self.build_pipeline()

            logger.info(
                "Starting Random Forest "
                "hyperparameter tuning"
            )

            # -----------------------------------------------
            # Chronological cross-validation
            # -----------------------------------------------

            time_series_cv = TimeSeriesSplit(
                n_splits=self.config.cv_splits
            )

            # -----------------------------------------------
            # RandomizedSearchCV
            # -----------------------------------------------

            self.search = RandomizedSearchCV(
                estimator=self.pipeline,

                param_distributions=(
                    self.parameter_distribution()
                ),

                n_iter=self.config.n_iter,

                scoring=self.config.scoring,

                cv=time_series_cv,

                n_jobs=self.config.n_jobs,

                random_state=(
                    self.config.random_state
                ),

                verbose=1,

                refit=True,

                return_train_score=True,
            )

            self.search.fit(
                self.X_train,
                self.y_train,
            )

            self.best_params = (
                self.search.best_params_
            )

            logger.info(
                "Hyperparameter tuning completed"
            )

            return self.search

        except Exception:

            logger.exception(
                "Random Forest hyperparameter "
                "tuning failed"
            )

            raise


    # ========================================================
    # 5. CLEAN PARAMETER NAMES
    # ========================================================

    def get_best_model_parameters(
        self,
    ) -> dict[str, Any]:
        """
        Remove sklearn Pipeline prefixes from the best
        classifier parameters.
        """

        if not self.best_params:
            raise RuntimeError(
                "Hyperparameter tuning has not been run."
            )

        prefix = "classifier__"

        clean_params = {
            (
                key[len(prefix):]
                if key.startswith(prefix)
                else key
            ): value

            for key, value
            in self.best_params.items()
        }

        # Preserve reproducibility settings
        clean_params["random_state"] = (
            self.config.random_state
        )

        clean_params["n_jobs"] = (
            self.config.n_jobs
        )

        return clean_params


    # ========================================================
    # 6. SAVE BEST PARAMETERS
    # ========================================================

    def save_best_parameters(
        self,
    ) -> Path:
        """
        Save tuned Random Forest parameters for use by
        model_training.py.
        """

        best_params = (
            self.get_best_model_parameters()
        )

        MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        with BEST_PARAMS_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "best_parameters":
                        best_params,

                    "best_cv_score":
                        float(
                            self.search.best_score_
                        ),

                    "scoring":
                        self.config.scoring,

                    "tuning_configuration":
                        asdict(self.config),
                },
                file,
                indent=4,
            )

        logger.info(
            "Best parameters saved to %s",
            BEST_PARAMS_PATH,
        )

        return BEST_PARAMS_PATH


    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    def display_results(
        self,
    ) -> None:
        """Display the best tuning result."""

        best_params = (
            self.get_best_model_parameters()
        )

        print(
            "\n"
            + "=" * 65
        )

        print(
            "RANDOM FOREST HYPERPARAMETER TUNING"
        )

        print(
            "=" * 65
        )

        print(
            "\nBest Parameters:"
        )

        for parameter, value in (
            best_params.items()
        ):
            print(
                f"{parameter}: {value}"
            )

        print(
            f"\nBest CV "
            f"{self.config.scoring} score: "
            f"{self.search.best_score_:.4f}"
        )


    # ========================================================
    # 8. RUN TUNING PIPELINE
    # ========================================================

    def run(
        self,
    ) -> dict[str, Any]:
        """
        Execute the complete hyperparameter tuning workflow.
        """

        logger.info(
            "Starting NovaPay model tuning pipeline"
        )

        self.prepare_training_data()

        self.build_pipeline()

        self.tune()

        self.display_results()

        output_path = (
            self.save_best_parameters()
        )

        logger.info(
            "NovaPay model tuning pipeline "
            "completed successfully"
        )

        return {
            "best_parameters":
                self.get_best_model_parameters(),

            "best_cv_score":
                float(
                    self.search.best_score_
                ),

            "output_path":
                output_path,
        }


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def run_pipeline() -> dict[str, Any]:
    """
    Run the NovaPay Random Forest tuning pipeline.
    """

    tuner = RandomForestTuner()

    return tuner.run()


if __name__ == "__main__":
    run_pipeline()