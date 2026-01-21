from src.preprocessors.factory import PreprocessorFactory
import numpy as np

def test_nn_preprocessor_fit_and_transform(small_df, data_cfg_nn, monkeypatch):
    # monkeypatch the CSV load if needed
    monkeypatch.setattr(
        "pandas.read_csv",
        lambda _: small_df
    )

    preprocessor = PreprocessorFactory.get_preprocessor(
        model_type="nn",
        data_cfg=data_cfg_nn
    )

    X_train, X_val, y_train, y_val, artifacts = preprocessor.preprocess()

    assert X_train.shape[1] > 0
    assert X_val.shape[1] > 0
    assert len(y_train) > 0

    assert preprocessor.scaler is not None
    assert preprocessor.label_encoder is not None

    assert "scaler" in artifacts
    assert "encoder" in artifacts



#--------------Test data leakage----------------------
# the error we try to detect: fitting the scaler on train + val instead of train only
def test_no_scaler_data_leakage(small_df, data_cfg_nn, monkeypatch):
    monkeypatch.setattr("pandas.read_csv", lambda _: small_df)

    preprocessor = PreprocessorFactory.get_preprocessor(
        model_type="nn",
        data_cfg=data_cfg_nn,
    )

    X_train, X_val, y_train, y_val, _ = preprocessor.preprocess()

    # Train data must be ~zero mean
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)

    # Validation data must NOT be zero mean (otherwise scaler saw val)
    assert not np.allclose(X_val.mean(axis=0), 0, atol=1e-2)
