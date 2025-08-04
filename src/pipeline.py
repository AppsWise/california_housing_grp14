"""
MLOps Data Pipeline
==================

Integrated pipeline that replaces Jupyter notebooks with production-ready Python modules.
Combines EDA, feature engineering, and model training in an automated workflow.

Author: Group 14
Date: August 2025
"""

import logging
import argparse
from pathlib import Path
import json
from typing import Dict, Any

# Import our custom modules
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.eda import HousingEDA
from src.data.feature_engineering import FeatureEngineer

# from src.models.train import ModelTrainer  # Comment out for now
from src.utils.logging_config import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """
    Complete MLOps pipeline orchestrator.

    Coordinates the entire ML workflow from data analysis to model deployment.
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize the MLOps pipeline.

        Args:
            config_path (str): Path to pipeline configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}

        # Initialize components
        self.eda = HousingEDA(
            data_path=self.config.get("data_path", "data/housing.csv")
        )
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = None  # Initialize when needed

    def _load_config(self) -> Dict[str, Any]:
        """
        Load pipeline configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            import yaml

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Config file {self.config_path} not found. Using defaults."
            )
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default pipeline configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "data_path": "data/housing.csv",
            "output_dir": "reports",
            "models_dir": "models",
            "run_eda": True,
            "run_feature_engineering": True,
            "run_model_training": True,
            "feature_engineering": {
                "include_transformations": True,
                "include_scaling": True,
                "include_polynomial": False,
            },
            "model_training": {
                "test_size": 0.2,
                "random_state": 42,
                "models": [
                    "linear_regression",
                    "decision_tree",
                    "random_forest",
                ],
            },
        }

    def run_eda_analysis(self) -> Dict[str, Any]:
        """
        Run exploratory data analysis.

        Returns:
            Dict[str, Any]: EDA results
        """
        logger.info("=" * 60)
        logger.info("RUNNING EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 60)

        try:
            # Run full EDA
            eda_results = self.eda.run_full_analysis()

            # Save results
            output_path = Path(self.config["output_dir"]) / "eda_results.json"
            self.eda.save_analysis_results(str(output_path))

            self.results["eda"] = eda_results
            logger.info("EDA analysis completed successfully!")

            return eda_results

        except Exception as e:
            logger.error(f"Error in EDA analysis: {e}")
            raise

    def run_feature_engineering(self) -> str:
        """
        Run feature engineering pipeline.

        Returns:
            str: Path to engineered dataset
        """
        logger.info("=" * 60)
        logger.info("RUNNING FEATURE ENGINEERING")
        logger.info("=" * 60)

        try:
            # Load data
            df = self.eda.load_data()

            # Run feature engineering
            fe_config = self.config.get("feature_engineering", {})
            df_engineered = self.feature_engineer.engineer_features(
                df, target_column="median_house_value", **fe_config
            )

            # Save engineered dataset
            output_path = Path("data/processed/housing_engineered.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_engineered.to_csv(output_path, index=False)

            # Save transformers
            self.feature_engineer.save_transformers()

            self.results["feature_engineering"] = {
                "original_features": len(
                    self.feature_engineer.original_features
                ),
                "final_features": len(self.feature_engineer.feature_names),
                "feature_names": self.feature_engineer.feature_names,
                "output_path": str(output_path),
            }

            logger.info(
                f"Feature engineering completed! Dataset saved to {output_path}"
            )
            return str(output_path)

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise

    def run_model_training(self, data_path: str = None) -> Dict[str, Any]:
        """
        Run model training pipeline.

        Args:
            data_path (str): Path to training data

        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("=" * 60)
        logger.info("RUNNING MODEL TRAINING")
        logger.info("=" * 60)

        try:
            # Initialize model trainer
            from src.models.train import train_models

            # Use engineered data if available, otherwise original data
            if data_path is None:
                if "feature_engineering" in self.results:
                    data_path = self.results["feature_engineering"][
                        "output_path"
                    ]
                else:
                    data_path = self.config["data_path"]

            # Run model training
            training_config = self.config.get("model_training", {})
            training_results = train_models(
                data_path=data_path, **training_config
            )

            self.results["model_training"] = training_results
            logger.info("Model training completed successfully!")

            return training_results

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete MLOps pipeline.

        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        logger.info("🚀 STARTING COMPLETE MLOPS PIPELINE")
        logger.info("=" * 80)

        try:
            # Step 1: EDA Analysis (if enabled)
            if self.config.get("run_eda", True):
                self.run_eda_analysis()
            else:
                logger.info("Skipping EDA analysis (disabled in config)")

            # Step 2: Feature Engineering (if enabled)
            engineered_data_path = None
            if self.config.get("run_feature_engineering", True):
                engineered_data_path = self.run_feature_engineering()
            else:
                logger.info(
                    "Skipping feature engineering (disabled in config)"
                )

            # Step 3: Model Training (if enabled)
            if self.config.get("run_model_training", True):
                self.run_model_training(engineered_data_path)
            else:
                logger.info("Skipping model training (disabled in config)")

            # Save complete results
            self._save_pipeline_results()

            logger.info("=" * 80)
            logger.info("🎉 MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _save_pipeline_results(self):
        """Save complete pipeline results."""
        output_path = Path(self.config["output_dir"]) / "pipeline_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert any non-serializable objects
        def convert_for_json(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            elif hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        try:
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2, default=convert_for_json)
            logger.info(f"Pipeline results saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save pipeline results: {e}")

    def generate_pipeline_report(self) -> str:
        """
        Generate a comprehensive pipeline report.

        Returns:
            str: Path to generated report
        """
        logger.info("Generating pipeline report...")

        report_lines = [
            "# MLOps Pipeline Execution Report",
            f"Generated on: {logger.info('Report generated')}",
            "",
            "## Pipeline Configuration",
            f"- Data Path: {self.config.get('data_path', 'N/A')}",
            f"- Output Directory: {self.config.get('output_dir', 'N/A')}",
            f"- EDA Enabled: {self.config.get('run_eda', False)}",
            f"- Feature Engineering Enabled: "
            f"{self.config.get('run_feature_engineering', False)}",
            f"- Model Training Enabled: "
            f"{self.config.get('run_model_training', False)}",
            "",
        ]

        # EDA Results
        if "eda" in self.results:
            eda = self.results["eda"]
            report_lines.extend(
                [
                    "## Exploratory Data Analysis Results",
                    f"- Dataset Shape: "
                    f"{eda.get('structure', {}).get('shape', 'N/A')}",
                    f"- Missing Values: "
                    f"{eda.get('missing_values', {}).get('total_missing', 'N/A')}",
                    f"- Numeric Features: "
                    f"{len(eda.get('structure', {}).get('numeric_columns', []))}",
                    f"- Categorical Features: "
                    f"{len(eda.get('structure', {}).get('categorical_columns', []))}",
                    "",
                ]
            )

        # Feature Engineering Results
        if "feature_engineering" in self.results:
            fe = self.results["feature_engineering"]
            report_lines.extend(
                [
                    "## Feature Engineering Results",
                    f"- Original Features: {fe.get('original_features', 'N/A')}",
                    f"- Final Features: {fe.get('final_features', 'N/A')}",
                    f"- Feature Expansion Ratio: "
                    f"{fe.get('final_features', 0) / max(fe.get('original_features', 1), 1):.2f}x",
                    "",
                ]
            )

        # Model Training Results
        if "model_training" in self.results:
            mt = self.results["model_training"]
            report_lines.extend(
                [
                    "## Model Training Results",
                    f"- Models Trained: "
                    f"{len(mt.get('models', {})) if mt.get('models') else 'N/A'}",
                    f"- Best Model: {mt.get('best_model', 'N/A')}",
                    f"- Best Score: {mt.get('best_score', 'N/A')}",
                    "",
                ]
            )

        # Save report
        report_content = "\n".join(report_lines)
        report_path = Path(self.config["output_dir"]) / "pipeline_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"Pipeline report saved to {report_path}")
        return str(report_path)


def create_default_config():
    """Create default pipeline configuration file."""
    config = {
        "data_path": "data/housing.csv",
        "output_dir": "reports",
        "models_dir": "models",
        "run_eda": True,
        "run_feature_engineering": True,
        "run_model_training": True,
        "feature_engineering": {
            "include_transformations": True,
            "include_scaling": True,
            "include_polynomial": False,
        },
        "model_training": {
            "test_size": 0.2,
            "random_state": 42,
            "models": ["linear_regression", "decision_tree", "random_forest"],
        },
    }

    config_path = Path("config/pipeline_config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f, indent=2)

    print(f"Default configuration created at {config_path}")


def main():
    """Main function to run the MLOps pipeline."""
    parser = argparse.ArgumentParser(description="MLOps Pipeline Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file",
    )
    parser.add_argument(
        "--eda-only", action="store_true", help="Run only EDA analysis"
    )
    parser.add_argument(
        "--fe-only", action="store_true", help="Run only feature engineering"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Run only model training"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)

    try:
        # Create default config if requested
        if args.create_config:
            create_default_config()
            return

        # Initialize pipeline
        pipeline = MLOpsPipeline(config_path=args.config)

        # Run specific components or full pipeline
        if args.eda_only:
            pipeline.run_eda_analysis()
        elif args.fe_only:
            pipeline.run_feature_engineering()
        elif args.train_only:
            pipeline.run_model_training()
        else:
            # Run full pipeline
            pipeline.run_full_pipeline()

            # Generate report
            report_path = pipeline.generate_pipeline_report()
            logger.info(
                f"Pipeline completed! Report available at: {report_path}"
            )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
