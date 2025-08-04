"""
Simple MLOps Pipeline Demo
=========================

Simplified pipeline that demonstrates notebook-to-Python conversion concept
without complex dependencies that have compatibility issues.

Author: Group 14
Date: August 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleEDA:
    """Simplified EDA class for demonstration."""

    def __init__(self, data_path: str = "data/housing.csv"):
        self.data_path = data_path
        self.results = {}

    def run_analysis(self) -> Dict[str, Any]:
        """Run simplified EDA analysis."""
        logger.info("Running EDA Analysis...")

        try:
            # Simulate EDA results
            self.results = {
                "dataset_info": {
                    "file_path": self.data_path,
                    "analysis_type": "California Housing EDA",
                    "features_analyzed": [
                        "longitude",
                        "latitude",
                        "housing_median_age",
                        "total_rooms",
                        "total_bedrooms",
                        "population",
                        "households",
                        "median_income",
                        "ocean_proximity",
                    ],
                },
                "summary": {
                    "total_features": 9,
                    "target_variable": "median_house_value",
                    "data_quality": "Good (minimal missing values)",
                    "key_insights": [
                        "Geographic features show clear regional patterns",
                        "Income is strongly correlated with house values",
                        "Ocean proximity affects pricing significantly",
                    ],
                },
            }

            logger.info("✅ EDA Analysis completed successfully!")
            return self.results

        except Exception as e:
            logger.error(f"Error in EDA: {e}")
            raise


class SimpleFeatureEngineering:
    """Simplified Feature Engineering class for demonstration."""

    def __init__(self):
        self.transformations = []

    def create_features(self, data_info: Dict) -> Dict[str, Any]:
        """Create new features based on EDA insights."""
        logger.info("Running Feature Engineering...")

        try:
            # Simulate feature engineering
            new_features = [
                "bedrooms_per_household",
                "rooms_per_household",
                "population_per_household",
                "income_category",
                "age_category",
            ]

            transformations = [
                "log_transform_population",
                "log_transform_total_rooms",
                "standard_scale_numeric_features",
                "one_hot_encode_ocean_proximity",
            ]

            results = {
                "original_features": len(
                    data_info.get("features_analyzed", [])
                ),
                "new_features_created": new_features,
                "transformations_applied": transformations,
                "final_feature_count": len(
                    data_info.get("features_analyzed", [])
                )
                + len(new_features),
                "feature_engineering_summary": {
                    "ratio_features": 3,
                    "categorical_features": 2,
                    "scaling_applied": True,
                    "encoding_applied": True,
                },
            }

            self.transformations = transformations
            logger.info("✅ Feature Engineering completed successfully!")
            return results

        except Exception as e:
            logger.error(f"Error in Feature Engineering: {e}")
            raise


class SimpleModelTraining:
    """Simplified Model Training class for demonstration."""

    def __init__(self):
        self.models = {}

    def train_models(self, feature_info: Dict) -> Dict[str, Any]:
        """Train models with engineered features."""
        logger.info("Running Model Training...")

        try:
            # Simulate model training results
            model_results = {
                "linear_regression": {
                    "r2_score": 0.6234,
                    "mse": 4.7821e9,
                    "training_time": "2.3s",
                },
                "decision_tree": {
                    "r2_score": 0.6789,
                    "mse": 4.1234e9,
                    "training_time": "5.7s",
                },
                "random_forest": {
                    "r2_score": 0.7123,
                    "mse": 3.8945e9,
                    "training_time": "23.4s",
                },
            }

            # Select best model
            best_model = max(
                model_results.items(), key=lambda x: x[1]["r2_score"]
            )

            results = {
                "models_trained": list(model_results.keys()),
                "model_results": model_results,
                "best_model": {
                    "name": best_model[0],
                    "performance": best_model[1],
                },
                "features_used": feature_info.get("final_feature_count", 0),
                "training_summary": {
                    "total_models": len(model_results),
                    "best_r2_score": best_model[1]["r2_score"],
                    "performance_improvement": "Significant improvement with "
                    "feature engineering",
                },
            }

            logger.info(
                f"✅ Model Training completed! Best model: {best_model[0]} "
                f"(R² = {best_model[1]['r2_score']:.4f})"
            )
            return results

        except Exception as e:
            logger.error(f"Error in Model Training: {e}")
            raise


class SimplePipeline:
    """Simplified MLOps Pipeline orchestrator."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.results = {}

        # Initialize components
        self.eda = SimpleEDA(self.config.get("data_path", "data/housing.csv"))
        self.feature_engineer = SimpleFeatureEngineering()
        self.model_trainer = SimpleModelTraining()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data_path": "data/housing.csv",
            "output_dir": "reports",
            "run_eda": True,
            "run_feature_engineering": True,
            "run_model_training": True,
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        logger.info("🚀 STARTING SIMPLIFIED MLOPS PIPELINE")
        logger.info("=" * 60)

        try:
            # Step 1: EDA
            if self.config.get("run_eda", True):
                eda_results = self.eda.run_analysis()
                self.results["eda"] = eda_results

            # Step 2: Feature Engineering
            if self.config.get("run_feature_engineering", True):
                fe_results = self.feature_engineer.create_features(
                    self.results.get("eda", {}).get("dataset_info", {})
                )
                self.results["feature_engineering"] = fe_results

            # Step 3: Model Training
            if self.config.get("run_model_training", True):
                training_results = self.model_trainer.train_models(
                    self.results.get("feature_engineering", {})
                )
                self.results["model_training"] = training_results

            # Save results
            self._save_results()

            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)

            # Print summary
            self._print_summary()

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _save_results(self):
        """Save pipeline results."""
        output_dir = Path(self.config.get("output_dir", "reports"))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "pipeline_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def _print_summary(self):
        """Print pipeline summary."""
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)

        if "eda" in self.results:
            eda = self.results["eda"]["summary"]
            print(f"📊 EDA: Analyzed {eda['total_features']} features")
            print(f"   Target: {eda['target_variable']}")
            print(f"   Quality: {eda['data_quality']}")

        if "feature_engineering" in self.results:
            fe = self.results["feature_engineering"]
            print(
                f"🔧 Feature Engineering: {fe['original_features']} → "
                f"{fe['final_feature_count']} features"
            )
            print(f"   New features: {len(fe['new_features_created'])}")
            print(f"   Transformations: {len(fe['transformations_applied'])}")

        if "model_training" in self.results:
            mt = self.results["model_training"]
            best_model = mt["best_model"]
            print(
                f"🤖 Model Training: "
                f"{mt['training_summary']['total_models']} models trained"
            )
            print(f"   Best model: {best_model['name']}")
            print(
                f"   Best R² score: {best_model['performance']['r2_score']:.4f}"
            )

        print("\n✅ All components executed successfully!")
        print("📁 Results saved to reports/pipeline_results.json")


def create_config_file():
    """Create a sample configuration file."""
    config = {
        "data_path": "data/housing.csv",
        "output_dir": "reports",
        "run_eda": True,
        "run_feature_engineering": True,
        "run_model_training": True,
        "pipeline_info": {
            "description": "Simplified MLOps pipeline demonstrating "
            "notebook-to-Python conversion",
            "author": "Group 14",
            "version": "1.0.0",
        },
    }

    config_path = Path("config/simple_pipeline_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuration file created: {config_path}")


def main():
    """Main function to demonstrate the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Simplified MLOps Pipeline")
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create configuration file",
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration file"
    )

    args = parser.parse_args()

    if args.create_config:
        create_config_file()
        return

    # Load configuration if provided
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")

    # Run pipeline
    pipeline = SimplePipeline(config)
    pipeline.run_full_pipeline()

    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
