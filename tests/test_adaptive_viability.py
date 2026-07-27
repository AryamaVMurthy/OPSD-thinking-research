import pytest

from opsd_research.adaptive_viability import (
    ActionEvidence,
    beta_credible_interval,
    beta_posterior_mean,
    next_uncertain_action,
)


def test_integer_beta_interval_matches_closed_form_uniform_cases() -> None:
    assert beta_credible_interval(0, 0, credible_level=0.9) == pytest.approx(
        (0.05, 0.95), abs=1e-10
    )
    lower, upper = beta_credible_interval(0, 2, credible_level=0.9)
    assert lower == pytest.approx(1.0 - 0.95 ** (1.0 / 3.0), abs=1e-9)
    assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / 3.0), abs=1e-9)


def test_posterior_mean_never_turns_two_samples_into_false_certainty() -> None:
    assert beta_posterior_mean(0, 2) == pytest.approx(0.25)
    assert beta_posterior_mean(1, 2) == pytest.approx(0.5)
    assert beta_posterior_mean(2, 2) == pytest.approx(0.75)


def test_adaptive_sampler_stops_when_best_action_is_credibly_separated() -> None:
    separated = [
        ActionEvidence("good", successes=4, trials=4),
        ActionEvidence("bad", successes=0, trials=4),
    ]
    assert (
        next_uncertain_action(
            separated, max_trials=4, credible_level=0.9
        )
        is None
    )


def test_adaptive_sampler_spends_only_on_overlapping_contenders() -> None:
    uncertain = [
        ActionEvidence("a", successes=2, trials=2),
        ActionEvidence("b", successes=0, trials=2),
        ActionEvidence("settled", successes=0, trials=4),
    ]
    selected = next_uncertain_action(
        uncertain, max_trials=4, credible_level=0.9
    )
    assert selected in {"a", "b"}
