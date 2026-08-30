import numpy as np

from src.catsdogs.model import predict_features


class FakeClassifier:
    classes_ = np.asarray(["cat", "dog"])

    def predict_proba(self, rows):
        assert rows.shape == (1, 4)
        return np.asarray([[0.2, 0.8]])


def test_predict_features_returns_label_and_probabilities():
    result = predict_features({"model": FakeClassifier()}, np.zeros(4, dtype=np.float32))
    assert result["label"] == "dog"
    assert result["probabilities"] == {"cat": 0.2, "dog": 0.8}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-9

