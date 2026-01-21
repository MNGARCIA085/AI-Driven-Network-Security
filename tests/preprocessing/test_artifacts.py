from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.preprocessors.factory import PreprocessorFactory

def test_load_artifacts_sets_attributes(data_cfg_nn):
    preprocessor = PreprocessorFactory.get_preprocessor(
        model_type="nn",
        data_cfg=data_cfg_nn
    )

    scaler = StandardScaler()
    encoder = LabelEncoder()

    artifacts = {
        "scaler": scaler,
        "label_encoder": encoder,
    }

    preprocessor.load_artifacts(artifacts)

    assert preprocessor.scaler is scaler
    assert preprocessor.label_encoder is encoder
