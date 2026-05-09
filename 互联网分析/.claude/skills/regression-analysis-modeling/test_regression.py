#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import core_regression
import pandas as pd
import numpy as np

def test_regression_analyzer():
    """Test the RegressionAnalyzer class"""
    print("Testing RegressionAnalyzer...")

    try:
        # Test creating an instance
        analyzer = core_regression.RegressionAnalyzer()
        print("✓ Instance created successfully")

        # Test that all required attributes are properly initialized
        print(f"✓ random_state: {analyzer.random_state}")
        print(f"✓ chinese_font: {analyzer.chinese_font}")
        print(f"✓ models: {type(analyzer.models)}")
        print(f"✓ scalers: {type(analyzer.scalers)}")

        # Create sample data for testing
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y = 2 * X['feature1'] + 3 * X['feature2'] + np.random.randn(n_samples) * 0.1

        print(f"✓ Sample data created: X shape {X.shape}, y shape {y.shape}")

        # Test preprocessing
        X_processed, y_processed = analyzer.preprocess_data(X, y)
        print(f"✓ Preprocessing completed: X shape {X_processed.shape}")

        # Test encoding
        X_encoded = analyzer.encode_categorical_features(X_processed)
        print(f"✓ Encoding completed: X shape {X_encoded.shape}")

        # Test training (this will test if all the issues are resolved)
        try:
            results = analyzer.train_models(X_encoded, y_processed)
            print(f"✓ Model training completed")
            print(f"✓ Best model: {analyzer.best_model_name}")
            print(f"✓ Results keys: {list(results.keys())}")
        except Exception as e:
            print(f"✗ Model training failed: {e}")
            return False

        print("All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_regression_analyzer()