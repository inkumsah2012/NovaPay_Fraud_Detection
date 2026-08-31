import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_bundle.joblib"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

METRICS_PATH = (
    REPORT_DIRECTORY
    / "random_forest_metrics.json"
)

CLASSIFICATION_REPORT_PATH = (
    REPORT_DIRECTORY
    / "classification_report.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIRECTORY
    / "test_predictions.csv"
)


# ============================================================
# EVALUATION RESULT
# ============================================================

@dataclass(frozen=True)
class EvaluationResult:
    """
    Structured holdout evaluation metrics.
    """

    accuracy: float
    balanced_accuracy: float

    precision: float
    recall: float
    f1_score: float

    roc_auc: float
    pr_auc: float

    specificity: float

    false_positive_rate: float
    false_negative_rate: float

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    decision_threshold: float

    test_rows: int
    fraud_cases: int
    fraud_rate: float


# ============================================================
# MODEL EVALUATOR
# ============================================================

class ModelEvaluator:
    """
    Evaluate the trained NovaPay Random Forest on the untouched
    chronological test dataset.

    No training or fitting occurs in this class.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ) -> None:

        self.model_path = model_path

        self.bundle: dict[str, Any] | None = None

        self.model: Any | None = None

        self.feature_names: list[str] = []

        self.train_size: float = 0.80

        self.decision_threshold: float = 0.50

        self.X_test: pd.DataFrame | None = None

        self.y_test: pd.Series | None = None

        self.predictions: np.ndarray | None = None

        self.probabilities: np.ndarray | None = None

        self.result: EvaluationResult | None = None


    # ========================================================
    # 1. LOAD TRAINED MODEL
    # ========================================================

    def load_model(
        self,
    ) -> dict[str, Any]:
        """
        Load the trained Random Forest model bundle.
        """

        try:

            if not self.model_path.exists():
                raise FileNotFoundError(
                    "Trained model not found at "
                    f"{self.model_path}. "
                    "Run model_training.py first."
                )

            logger.info(
                "Loading trained model from %s",
                self.model_path,
            )

            self.bundle = joblib.load(
                self.model_path
            )

            required_keys = {
                "model",
                "feature_names",
                "decision_threshold",
                "train_size",
            }

            missing_keys = (
                required_keys
                - set(self.bundle.keys())
            )

            if missing_keys:
                raise ValueError(
                    "Saved model bundle is incomplete. "
                    f"Missing keys: {sorted(missing_keys)}"
                )

            self.model = (
                self.bundle["model"]
            )

            self.feature_names = list(
                self.bundle["feature_names"]
            )

            self.decision_threshold = float(
                self.bundle[
                    "decision_threshold"
                ]
            )

            self.train_size = float(
                self.bundle[
                    "train_size"
                ]
            )

            logger.info(
                "Trained model loaded successfully"
            )

            return self.bundle

        except Exception:

            logger.exception(
                "Failed to load trained model"
            )

            raise


    # ========================================================
    # 2. PREPARE HOLDOUT TEST DATA
    # ========================================================

    def prepare_test_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Recreate the untouched chronological test partition.

        The saved model already contains its fitted
        preprocessing pipeline, so no preprocessing is fitted
        during evaluation.
        """

        try:

            logger.info(
                "Preparing chronological holdout dataset"
            )

            if self.bundle is None:
                self.load_model()

            df = load_feature_engineered_data()

            preprocessor_config = (
                DataPreprocessing(
                    df=df,
                    categorical_features=(
                        CATEGORICAL_FEATURES
                    ),
                    binary_features=(
                        BINARY_FEATURES
                    ),
                    numeric_features=(
                        NUMERIC_FEATURES
                    ),
                )
            )

            _, test_df = (
                preprocessor_config
                .chronological_split(
                    train_size=self.train_size
                )
            )

            missing_features = [
                feature
                for feature in self.feature_names
                if feature not in test_df.columns
            ]

            if missing_features:
                raise ValueError(
                    "Test dataset is missing model "
                    f"features: {missing_features}"
                )

            self.X_test = (
                test_df[
                    self.feature_names
                ]
                .copy()
            )

            self.y_test = (
                test_df["is_fraud"]
                .astype("int8")
                .reset_index(drop=True)
            )

            self._validate_test_data()

            logger.info(
                "Holdout dataset prepared: "
                "%d observations",
                len(self.X_test),
            )

            return (
                self.X_test,
                self.y_test,
            )

        except Exception:

            logger.exception(
                "Failed to prepare holdout dataset"
            )

            raise


    # ========================================================
    # 3. VALIDATE TEST DATA
    # ========================================================

    def _validate_test_data(
        self,
    ) -> None:
        """Validate holdout predictors and target."""

        if (
            self.X_test is None
            or self.y_test is None
        ):
            raise RuntimeError(
                "Test data has not been prepared."
            )

        if self.X_test.empty:
            raise ValueError(
                "Test predictor dataset is empty."
            )

        if self.y_test.empty:
            raise ValueError(
                "Test target is empty."
            )

        if len(self.X_test) != len(self.y_test):
            raise ValueError(
                "X_test and y_test contain different "
                "numbers of observations."
            )

        target_values = set(
            self.y_test.unique()
        )

        if not target_values.issubset(
            {0, 1}
        ):
            raise ValueError(
                "Test target must contain only 0 and 1."
            )

        if len(target_values) < 2:
            raise ValueError(
                "Holdout dataset must contain both "
                "legitimate and fraud observations."
            )


    # ========================================================
    # 4. GENERATE PREDICTIONS
    # ========================================================

    def predict(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate fraud probabilities and binary predictions.
        """

        try:

            if self.model is None:
                self.load_model()

            if self.X_test is None:
                self.prepare_test_data()

            logger.info(
                "Generating holdout predictions"
            )

            self.probabilities = (
                self.model
                .predict_proba(
                    self.X_test
                )[:, 1]
            )

            self.predictions = (
                self.probabilities
                >= self.decision_threshold
            ).astype("int8")

            logger.info(
                "Predictions generated successfully"
            )

            return (
                self.predictions,
                self.probabilities,
            )

        except Exception:

            logger.exception(
                "Failed to generate predictions"
            )

            raise


    # ========================================================
    # 5. CALCULATE EVALUATION METRICS
    # ========================================================

    def calculate_metrics(
        self,
    ) -> EvaluationResult:
        """
        Calculate fraud-focused holdout metrics.
        """

        try:

            if (
                self.predictions is None
                or self.probabilities is None
            ):
                self.predict()

            matrix = confusion_matrix(
                self.y_test,
                self.predictions,
                labels=[0, 1],
            )

            tn, fp, fn, tp = (
                matrix.ravel()
            )

            accuracy = accuracy_score(
                self.y_test,
                self.predictions,
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    self.y_test,
                    self.predictions,
                )
            )

            precision = precision_score(
                self.y_test,
                self.predictions,
                zero_division=0,
            )

            recall = recall_score(
                self.y_test,
                self.predictions,
                zero_division=0,
            )

            f1 = f1_score(
                self.y_test,
                self.predictions,
                zero_division=0,
            )

            roc_auc = roc_auc_score(
                self.y_test,
                self.probabilities,
            )

            pr_auc = (
                average_precision_score(
                    self.y_test,
                    self.probabilities,
                )
            )

            specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else 0.0
            )

            false_positive_rate = (
                fp / (fp + tn)
                if (fp + tn) > 0
                else 0.0
            )

            false_negative_rate = (
                fn / (fn + tp)
                if (fn + tp) > 0
                else 0.0
            )

            self.result = EvaluationResult(
                accuracy=float(accuracy),

                balanced_accuracy=float(
                    balanced_accuracy
                ),

                precision=float(
                    precision
                ),

                recall=float(
                    recall
                ),

                f1_score=float(
                    f1
                ),

                roc_auc=float(
                    roc_auc
                ),

                pr_auc=float(
                    pr_auc
                ),

                specificity=float(
                    specificity
                ),

                false_positive_rate=float(
                    false_positive_rate
                ),

                false_negative_rate=float(
                    false_negative_rate
                ),

                true_negative=int(tn),

                false_positive=int(fp),

                false_negative=int(fn),

                true_positive=int(tp),

                decision_threshold=float(
                    self.decision_threshold
                ),

                test_rows=int(
                    len(self.y_test)
                ),

                fraud_cases=int(
                    self.y_test.sum()
                ),

                fraud_rate=float(
                    self.y_test.mean()
                ),
            )

            logger.info(
                "Evaluation metrics calculated successfully"
            )

            return self.result

        except Exception:

            logger.exception(
                "Failed to calculate evaluation metrics"
            )

            raise


    # ========================================================
    # 6. CLASSIFICATION REPORT
    # ========================================================

    def create_classification_report(
        self,
    ) -> pd.DataFrame:
        """
        Return the sklearn classification report as a DataFrame.
        """

        if self.predictions is None:
            self.predict()

        report = classification_report(
            self.y_test,
            self.predictions,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            output_dict=True,
            zero_division=0,
        )

        return (
            pd.DataFrame(report)
            .transpose()
        )


    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    def display_results(
        self,
    ) -> None:
        """
        Display the holdout evaluation summary.
        """

        if self.result is None:
            self.calculate_metrics()

        result = self.result

        matrix = np.array(
            [
                [
                    result.true_negative,
                    result.false_positive,
                ],
                [
                    result.false_negative,
                    result.true_positive,
                ],
            ]
        )

        print(
            "\n"
            + "=" * 65
        )

        print(
            "NOVAPAY RANDOM FOREST MODEL EVALUATION"
        )

        print(
            "=" * 65
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            matrix
        )

        print(
            "\nClassification Report:"
        )

        print(
            self.create_classification_report()
            .round(4)
            .to_string()
        )

        print(
            "\nModel Performance:"
        )

        print(
            "-" * 65
        )

        print(
            f"Accuracy:             "
            f"{result.accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy:    "
            f"{result.balanced_accuracy:.4f}"
        )

        print(
            f"Fraud Precision:      "
            f"{result.precision:.4f}"
        )

        print(
            f"Fraud Recall:         "
            f"{result.recall:.4f}"
        )

        print(
            f"Fraud F1 Score:       "
            f"{result.f1_score:.4f}"
        )

        print(
            f"ROC-AUC:              "
            f"{result.roc_auc:.4f}"
        )

        print(
            f"PR-AUC:               "
            f"{result.pr_auc:.4f}"
        )

        print(
            f"Specificity:          "
            f"{result.specificity:.4f}"
        )

        print(
            f"False Positive Rate:  "
            f"{result.false_positive_rate:.4f}"
        )

        print(
            f"False Negative Rate:  "
            f"{result.false_negative_rate:.4f}"
        )

        print(
            "\nOperational Outcomes:"
        )

        print(
            "-" * 65
        )

        print(
            f"Test transactions:    "
            f"{result.test_rows:,}"
        )

        print(
            f"Actual fraud cases:   "
            f"{result.fraud_cases:,}"
        )

        print(
            f"Fraud detected:       "
            f"{result.true_positive:,}"
        )

        print(
            f"Fraud missed:         "
            f"{result.false_negative:,}"
        )

        print(
            f"False fraud alerts:   "
            f"{result.false_positive:,}"
        )

        print(
            f"Legitimate approved:  "
            f"{result.true_negative:,}"
        )

        print(
            f"Decision threshold:   "
            f"{result.decision_threshold:.2f}"
        )


    # ========================================================
    # 8. SAVE EVALUATION REPORTS
    # ========================================================

    def save_reports(
        self,
    ) -> dict[str, Path]:
        """
        Persist evaluation results independently from
        the trained model.
        """

        if self.result is None:
            raise RuntimeError(
                "Evaluation must be completed before "
                "saving reports."
            )

        try:

            REPORT_DIRECTORY.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            with METRICS_PATH.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    asdict(self.result),
                    file,
                    indent=4,
                )

            # ------------------------------------------------
            # Classification report
            # ------------------------------------------------

            classification_df = (
                self.create_classification_report()
            )

            classification_df.to_csv(
                CLASSIFICATION_REPORT_PATH
            )

            # ------------------------------------------------
            # Test predictions
            # ------------------------------------------------

            predictions_df = pd.DataFrame(
                {
                    "actual_is_fraud":
                        self.y_test.to_numpy(),

                    "predicted_is_fraud":
                        self.predictions,

                    "fraud_probability":
                        self.probabilities,
                }
            )

            predictions_df.to_csv(
                PREDICTIONS_PATH,
                index=False,
            )

            logger.info(
                "Evaluation reports saved successfully"
            )

            return {
                "metrics":
                    METRICS_PATH,

                "classification_report":
                    CLASSIFICATION_REPORT_PATH,

                "predictions":
                    PREDICTIONS_PATH,
            }

        except Exception:

            logger.exception(
                "Failed to save evaluation reports"
            )

            raise


    # ========================================================
    # 9. RUN EVALUATION PIPELINE
    # ========================================================

    def run(
        self,
    ) -> EvaluationResult:
        """
        Execute the complete holdout evaluation workflow.
        """

        logger.info(
            "Starting NovaPay model evaluation pipeline"
        )

        self.load_model()

        self.prepare_test_data()

        self.predict()

        result = (
            self.calculate_metrics()
        )

        self.display_results()

        report_paths = (
            self.save_reports()
        )

        print(
            "\nEvaluation Reports:"
        )

        print(
            "-" * 65
        )

        for name, path in report_paths.items():

            print(
                f"{name}: {path}"
            )

        logger.info(
            "NovaPay model evaluation pipeline "
            "completed successfully"
        )

        return result


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def run_pipeline() -> EvaluationResult:
    """
    Run the standalone NovaPay model-evaluation pipeline.
    """

    evaluator = ModelEvaluator()

    return evaluator.run()


if __name__ == "__main__":
    run_pipeline()