"""
Unit tests for shared/depmap.py and baseline feature extraction functions.
"""
from pathlib import Path
import pandas as pd
import pytest
from shared.depmap import DependencyMap
from baseline.train import prepare_features_train
from baseline.app.main import prepare_features_serve


def test_dependency_map_upstream_downstream():
    sample_yaml = Path("samples/dependency_map.yaml")
    assert sample_yaml.exists(), "samples/dependency_map.yaml must exist"
    
    depmap = DependencyMap.from_yaml(sample_yaml)
    
    # Check upstream of total_sulfur_dioxide feature
    up_feature = depmap.upstream("feature:total_sulfur_dioxide")
    assert "code:baseline/app/main.py" in up_feature

    # Check downstream of total_sulfur_dioxide feature
    down_feature = depmap.downstream("feature:total_sulfur_dioxide")
    assert "dataset:batch-0001" in down_feature

    # Check full recursive upstream walk from deployment:prod
    ancestors = depmap.walk_upstream("deployment:prod")
    assert "model:wine-quality@1" in ancestors
    assert "dataset:batch-0001" in ancestors
    assert "feature:alcohol" in ancestors
    assert "code:baseline/app/main.py" in ancestors


def test_dependency_map_owners():
    sample_yaml = Path("samples/dependency_map.yaml")
    depmap = DependencyMap.from_yaml(sample_yaml)
    
    assert depmap.owner("code:baseline/app/main.py") == "reviewer"
    assert depmap.owner("feature:alcohol") == "reviewer"
    assert depmap.owner("dataset:batch-0001") == "checker"
    assert depmap.owner("model:wine-quality@1") == "trainer"
    assert depmap.owner("deployment:prod") == "deployer"


def test_pure_feature_extractors_consistency():
    # 11 sample feature values
    values = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    
    # Serving feature extractor
    df_serve = prepare_features_serve(values)
    assert df_serve.shape == (1, 11)
    
    # Training feature extractor
    df_train_in = df_serve.copy()
    df_train_in["quality"] = 5
    X_train, y_train = prepare_features_train(df_train_in)
    
    # Both must output identical column names in identical order
    assert list(df_serve.columns) == list(X_train.columns)
    assert y_train.iloc[0] == 5
