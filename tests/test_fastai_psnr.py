"""``FastaiRestorer``'s PSNR — the one metric this project supplies because fastai has no name for it.

`fastai.metrics` publishes `mse` / `mae` / `rmse` / `msle` and nothing scale-aware, so `psnr` — the
quantity the other seven backends report — was unreachable by NAME until `_as_metric` grew a
CONFIGURED ALIAS. What is pinned here is that the alias computes the SAME number as the reference
implementation, over the whole split rather than per batch, and that it survives the one thing a
unit test on CPU tensors will not show you.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastai")

from distilaide.deepinv_model import DeepInvModel  # noqa: E402
from distilaide.fastai_restorer import FastaiRestorer, PeakSignalNoiseRatio  # noqa: E402


def _as_metric(name: str, data_range: float = 1.0) -> Any:
    """Resolve a metric the way a built restorer does — the base's alias-first resolution."""
    restorer = FastaiRestorer(model=DeepInvModel(arch="dncnn", depth=2, nf=4), data_range=data_range)
    return restorer._resolve_metric(name)


def _batches(sizes: Tuple[int, ...] = (7, 5, 3, 1)) -> List[Tuple[Any, Any]]:
    """Ragged batches, because a real final batch is short and that is where averaging goes wrong."""
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.rand(n, 3, 32, 32, generator=generator),
            torch.rand(n, 3, 32, 32, generator=generator),
        )
        for n in sizes
    ]


def _run(metric: PeakSignalNoiseRatio, batches: List[Tuple[Any, Any]]) -> float:
    metric.reset()
    for predictions, targets in batches:
        metric.accumulate(SimpleNamespace(pred=predictions, y=targets))
    return metric.value


class TestItAgreesWithTheReference:
    """One column called `psnr` has to mean one thing in every row of the sweep."""

    def test_it_matches_torchmetrics_over_the_whole_split(self) -> None:
        from torchmetrics.image import PeakSignalNoiseRatio as TorchMetricsPSNR

        batches = _batches()
        reference = TorchMetricsPSNR(data_range=1.0)
        for predictions, targets in batches:
            reference.update(predictions, targets)

        assert _run(PeakSignalNoiseRatio(data_range=1.0), batches) == pytest.approx(
            reference.compute().item(), abs=1e-6
        )

    def test_it_does_not_depend_on_the_batch_SIZE(self) -> None:
        """The property that pins ACCUMULATION without restating the arithmetic.

        PSNR is a logarithm of a mean squared error, so the mean of per-batch PSNRs is not the PSNR
        of the split. An `AvgMetric` over a per-batch value would have been one line and wrong in a
        way that moves with the batch size — which in a cross-backend table is a systematic penalty
        on exactly the backend that computed it that way.
        """
        batches = _batches()
        one_batch = [(torch.cat([p for p, _ in batches]), torch.cat([t for _, t in batches]))]

        assert _run(PeakSignalNoiseRatio(), batches) == pytest.approx(_run(PeakSignalNoiseRatio(), one_batch), abs=1e-9)

    def test_the_per_batch_mean_is_a_DIFFERENT_number(self) -> None:
        """The counterfactual, executed — otherwise the test above passes for a wrong reason."""
        from torchmetrics.image import PeakSignalNoiseRatio as TorchMetricsPSNR

        batches = _batches()
        per_batch = float(np.mean([TorchMetricsPSNR(data_range=1.0)(p, t).item() for p, t in batches]))

        assert _run(PeakSignalNoiseRatio(data_range=1.0), batches) != pytest.approx(per_batch, abs=1e-3)

    def test_data_range_scales_it(self) -> None:
        """The knob is this task's `num_classes` and it is silent when wrong, so it is checked.

        20*log10(255) - 20*log10(1) = 48.13 dB of pure difference from the same tensors.
        """
        batches = _batches()

        unit = _run(PeakSignalNoiseRatio(data_range=1.0), batches)
        byte = _run(PeakSignalNoiseRatio(data_range=255.0), batches)

        assert byte - unit == pytest.approx(20.0 * np.log10(255.0), abs=1e-6)


class TestItSurvivesAnAcceleratorTensor:
    """The failure a CPU-tensor test cannot show, reproduced without needing a device.

    `matrainer.metric_functions` is deliberately framework-free — numpy is its only import — so
    `np.asarray` on a LIVE accelerator tensor raises *"can't convert mps:0 device type tensor to
    numpy"*. It was hit for real on the first sweep, where both fastai rows died mid-fit while every
    unit test here passed, because a `torch.rand(...)` in a test is already on the CPU.
    """

    class _DeviceTensor:
        """A tensor-like that refuses `np.asarray` until `.cpu()` has been called — as MPS does."""

        def __init__(self, data: Any, on_device: bool = True) -> None:
            self._data = data
            self._on_device = on_device

        def detach(self) -> "TestItSurvivesAnAcceleratorTensor._DeviceTensor":
            return self

        def cpu(self) -> "TestItSurvivesAnAcceleratorTensor._DeviceTensor":
            return type(self)(self._data, on_device=False)

        def __array__(self, dtype: Any = None) -> Any:
            if self._on_device:
                raise TypeError("can't convert mps:0 device type tensor to numpy.")
            return np.asarray(self._data, dtype=dtype)

    def test_accumulate_moves_the_batch_off_the_device_first(self) -> None:
        metric = PeakSignalNoiseRatio(data_range=1.0)
        metric.reset()
        predictions = self._DeviceTensor(np.full((2, 3, 4, 4), 0.5))
        targets = self._DeviceTensor(np.zeros((2, 3, 4, 4)))

        metric.accumulate(SimpleNamespace(pred=predictions, y=targets))

        assert metric.value == pytest.approx(20.0 * np.log10(1.0 / 0.5), abs=1e-9)

    def test_the_fake_reproduces_the_real_failure(self) -> None:
        """Without this, the test above would pass against a fake that never refused anything."""
        with pytest.raises(TypeError, match="device type tensor"):
            np.asarray(self._DeviceTensor(np.zeros((2, 2))))


class TestItIsReachableFromAConfig:
    """A config writes a NAME, so the alias has to resolve from one."""

    def test_psnr_resolves_to_the_alias_and_reports_under_the_shared_name(self) -> None:
        metric = _as_metric("psnr", 1.0)

        assert isinstance(metric, PeakSignalNoiseRatio)
        # NOT the class name: a metric reported under its own would open a column no other row fills.
        assert metric.name == "psnr"

    def test_the_alias_receives_the_restorers_data_range(self) -> None:
        assert _as_metric("psnr", 255.0).data_range == 255.0

    def test_the_alias_is_tried_BEFORE_fastai_metrics(self) -> None:
        """An alias exists precisely where the upstream lookup would fail or answer wrongly, so the
        order is load-bearing rather than incidental."""
        import fastai.metrics as fastai_metrics

        assert not hasattr(fastai_metrics, "psnr"), "fastai grew a psnr — re-check which one wins"
        assert isinstance(_as_metric("psnr"), PeakSignalNoiseRatio)

    def test_an_upstream_name_still_resolves_upstream(self) -> None:
        import fastai.metrics as fastai_metrics

        assert _as_metric("mse") is fastai_metrics.mse

    def test_an_unknown_name_names_the_alias_too(self) -> None:
        """The error has to mention what this backend supplies, or a reader concludes psnr is out."""
        with pytest.raises(ValueError, match="psnr"):
            _as_metric("ssim")

    def test_ssim_is_still_absent_and_the_message_says_where_to_get_it(self) -> None:
        """No accumulator would help — there is no shared SSIM implementation to hold."""
        with pytest.raises(ValueError, match="lightning or torch"):
            _as_metric("ssim")
