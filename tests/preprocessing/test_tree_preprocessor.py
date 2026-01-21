from src.preprocessors.factory import PreprocessorFactory


def test_tree_preprocessor_basic(small_df, data_cfg_tree, monkeypatch):
    monkeypatch.setattr(
        "pandas.read_csv",
        lambda _: small_df
    )

    preprocessor = PreprocessorFactory.get_preprocessor(
        model_type="tree",
        data_cfg=data_cfg_tree
    )

    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()

    # Tree models usually don't scale
    assert preprocessor.scaler is None or "scaler" not in artifacts

    # Encoder still required
    assert preprocessor.label_encoder is not None
