import pytest, torch
from deepanc.common.Network import CRN, default_channels, ModelType

@pytest.mark.parametrize("mt", list(ModelType))
def test_crn_output_shape(mt):
    model = CRN(default_channels, mt).eval()
    x = torch.randn(1, 2, 8, 161)
    with torch.no_grad():
        assert model(x).shape == x.shape