import pytest

from fraud_engine.features import compute_features
from fraud_engine.synth import generate


@pytest.fixture(scope="session")
def raw():
    return generate(n_customers=120, days=45, seed=3)


@pytest.fixture(scope="session")
def feat(raw):
    return compute_features(raw)
