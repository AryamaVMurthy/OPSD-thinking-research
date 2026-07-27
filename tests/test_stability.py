import pytest

torch = pytest.importorskip("torch")

from opsd_research.stability import TrainableParameterSnapshot


def test_snapshot_reports_adapter_parameter_and_update_norms() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Linear(2, 1, bias=False),
    )
    model[1].weight.requires_grad_(False)
    snapshot = TrainableParameterSnapshot.capture(model)

    with torch.no_grad():
        model[0].weight.add_(0.5)
        model[1].weight.add_(100.0)
    metrics = snapshot.measure(model)

    assert metrics["trainable_parameters"] == 4
    assert metrics["update_norm"] == pytest.approx(1.0)
    assert metrics["parameter_norm"] > 0.0
    assert metrics["update_to_parameter_ratio"] == pytest.approx(
        metrics["update_norm"] / metrics["parameter_norm"]
    )
    assert snapshot.measure(model)["update_norm"] == pytest.approx(0.0)
