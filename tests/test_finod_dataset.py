from __future__ import annotations

import pytest

from opsd_research.finod_dataset import (
    filter_finod_indices_by_sources,
    finod_training_row_from_graph,
    finod_training_row,
    remove_graph_identifiers,
    select_representative_finod_indices,
)
from opsd_research.graf_graph import parse_answer_masked_graph


def test_finod_row_keeps_answer_control_out_of_student_and_guide_fields():
    row = finod_training_row(
        question="Find the requested integer.",
        reference_solution="A private derivation ends with \\boxed{73}.",
        answer_masked_guide="Check parity, then compare the two boundary cases.",
        source_index=11,
    )

    assert row["problem"] == "Find the requested integer."
    assert row["solution"] == "Check parity, then compare the two boundary cases."
    assert row["finod_answer_control"] == "73"
    assert "73" not in row["problem"]
    assert "73" not in row["solution"]
    assert row["finod_source_index"] == 11


def test_graph_identifiers_cannot_accidentally_reveal_a_short_answer():
    guide = "\n".join(
        [
            "State 4: viable",
            "- 4 [viable]: Isolate n. Check: Verify the inequality.",
        ]
    )

    cleaned = remove_graph_identifiers(guide)
    row = finod_training_row(
        question="Find the requested integer.",
        reference_solution=r"A private derivation ends with \boxed{4}.",
        answer_masked_guide=cleaned,
        source_index=35,
    )

    assert "State 4" not in row["solution"]
    assert "- 4 " not in row["solution"]
    assert "Isolate n" in row["solution"]


def test_fixed_graph_preamble_is_not_part_of_finod_guidance():
    guide = "\n".join(
        [
            (
                "Use this answer-masked strategy graph as a scaffold. It "
                "contains no final answer; independently solve and verify "
                "the problem."
            ),
            "State initial: Compare the symbolic cases.",
            (
                "- route [viable]: Factor the expression. "
                "Check: Verify every condition. Recovery: Try substitution."
            ),
        ]
    )

    cleaned = remove_graph_identifiers(guide)
    row = finod_training_row(
        question="Choose the correct option.",
        reference_solution=r"A private derivation ends with \boxed{A}.",
        answer_masked_guide=cleaned,
        source_index=36,
    )

    assert not cleaned.startswith("Use this answer-masked")
    assert row["solution"].startswith("Strategy state:")
    assert "Factor the expression" in row["solution"]


def test_graph_is_replayed_through_exact_rendered_training_path():
    payload = {
        "forks": [
            {
                "fork_id": "f1",
                "state": "Compare symbolic routes.",
                "actions": [
                    {
                        "action_id": "a1",
                        "description": "Factor the expression.",
                        "status": "viable",
                        "validation_test": "Verify every condition.",
                        "recovery_action": "Try substitution.",
                    },
                    {
                        "action_id": "a2",
                        "description": "Use a parity argument.",
                        "status": "risky",
                        "validation_test": "Check both cases.",
                        "recovery_action": "Return to factorization.",
                    },
                ],
            }
        ]
    }
    graph = parse_answer_masked_graph(
        payload,
        problem="Classify the route.",
        reference_solution=r"A private derivation ends with \boxed{viable}.",
    )

    with pytest.raises(ValueError, match="reference answer"):
        finod_training_row_from_graph(
            question="Classify the route.",
            reference_solution=(
                r"A private derivation ends with \boxed{viable}."
            ),
            graph=graph,
            source_index=37,
        )


def test_semantic_answer_leak_is_still_rejected_after_identifier_removal():
    with pytest.raises(ValueError, match="reference answer"):
        finod_training_row(
            question="Find the requested integer.",
            reference_solution=r"A private derivation ends with \boxed{4}.",
            answer_masked_guide="Test 4 as the final candidate.",
            source_index=35,
        )


def test_finod_row_rejects_equivalent_latex_fraction_answer():
    with pytest.raises(ValueError, match="reference answer"):
        finod_training_row(
            question="Find the volume.",
            reference_solution=r"A private derivation ends with \boxed{\frac{4}{3}}.",
            answer_masked_guide="Compute both triple products; ensure the volume is 4/3.",
            source_index=36,
        )


def test_finod_row_rejects_explicit_numerical_intermediate_result():
    with pytest.raises(ValueError, match="numerical result"):
        finod_training_row(
            question="Find the volume.",
            reference_solution=r"A private derivation ends with \boxed{\frac{4}{3}}.",
            answer_masked_guide="Compute the triple product; verify that it is 8.",
            source_index=37,
        )


def test_representative_selection_is_exact_stratified_and_reproducible():
    rows = []
    for source in ("olympiads", "aops_forum", "cn_contest"):
        for length in range(1, 41):
            rows.append(
                {
                    "data_source": source,
                    "question": f"{source}-question-{length}",
                    "response": "x" * length,
                    "response_length": length,
                }
            )
    eligible = [index for index in range(len(rows)) if index % 7 != 0]

    selected_one, manifest_one = select_representative_finod_indices(
        rows,
        eligible_indices=eligible,
        limit=60,
        seed=73,
    )
    selected_two, manifest_two = select_representative_finod_indices(
        rows,
        eligible_indices=reversed(eligible),
        limit=60,
        seed=73,
    )

    assert selected_one == selected_two
    assert manifest_one == manifest_two
    assert len(selected_one) == 60
    assert len(set(selected_one)) == 60
    assert set(selected_one).issubset(eligible)
    assert set(manifest_one["selected_by_data_source"]) == {
        "aops_forum",
        "cn_contest",
        "olympiads",
    }
    assert all(
        count > 0
        for count in manifest_one["selected_by_data_source"].values()
    )
    assert all(
        count > 0
        for count in manifest_one["selected_by_length_stratum"].values()
    )
    assert len(manifest_one["selected_indices_sha256"]) == 64


def test_representative_selection_refuses_insufficient_eligible_rows():
    rows = [
        {
            "data_source": "olympiads",
            "question": f"q-{index}",
            "response": "solution",
        }
        for index in range(8)
    ]

    with pytest.raises(ValueError, match="eligible"):
        select_representative_finod_indices(
            rows,
            eligible_indices=range(8),
            limit=9,
            seed=73,
        )


def test_source_filter_keeps_only_requested_contest_families():
    rows = [
        {"data_source": "olympiads"},
        {"data_source": "aops_forum"},
        {"data_source": "amc_aime"},
        {"data_source": "cn_contest"},
    ]

    selected = filter_finod_indices_by_sources(
        rows,
        eligible_indices=[3, 2, 1, 0],
        data_sources=("amc_aime", "aops_forum"),
    )

    assert selected == [1, 2]
