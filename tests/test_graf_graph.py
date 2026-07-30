import unittest

from opsd_research.graf_graph import branch_target, parse_answer_masked_graph, render_graph_scaffold


class GrafGraphTests(unittest.TestCase):
    def payload(self):
        return {
            "forks": [{
                "fork_id": "f1",
                "state": "A polynomial constraint has been simplified.",
                "actions": [
                    {"action_id": "factor", "description": "Factor the symbolic expression.", "status": "viable", "validation_test": "Substitute each candidate.", "recovery_action": "Return to the unsimplified expression."},
                    {"action_id": "discriminant", "description": "Analyze discriminant conditions.", "status": "conditionally_viable", "validation_test": "Check the domain.", "recovery_action": "Compare roots against constraints."},
                    {"action_id": "guess", "description": "Guess a root from a decimal approximation.", "status": "invalid", "validation_test": "Verify exactly.", "recovery_action": "Use an exact method."},
                ],
            }]
        }

    def test_graph_rejects_reference_answer_and_copied_phrases(self):
        graph = parse_answer_masked_graph(self.payload(), problem="Solve x.", reference_solution="First factor carefully. Therefore \\boxed{7}.")
        self.assertEqual(len(graph.forks), 1)
        leaked = self.payload()
        leaked["forks"][0]["actions"][0]["description"] = "The final answer is 7."
        with self.assertRaisesRegex(ValueError, "answer"):
            parse_answer_masked_graph(leaked, problem="Solve x.", reference_solution="First factor carefully. Therefore \\boxed{7}.")

    def test_graph_rejects_equivalent_latex_fraction_answer(self):
        leaked = self.payload()
        leaked["forks"][0]["actions"][0]["validation_test"] = (
            "Ensure the resulting volume is 4/3."
        )

        with self.assertRaisesRegex(ValueError, "reference answer"):
            parse_answer_masked_graph(
                leaked,
                problem="Find the volume.",
                reference_solution=r"The volume is therefore \boxed{\frac{4}{3}}.",
            )

    def test_graph_rejects_explicit_numerical_intermediate_result(self):
        leaked = self.payload()
        leaked["forks"][0]["actions"][0]["validation_test"] = (
            "Ensure the dot product is 8 before continuing."
        )

        with self.assertRaisesRegex(ValueError, "numerical result"):
            parse_answer_masked_graph(
                leaked,
                problem="Find the volume of a tetrahedron with the stated vertices.",
                reference_solution=r"The final volume is \boxed{\frac{4}{3}}.",
            )

    def test_graph_rejects_fraction_equivalent_to_decimal_answer(self):
        leaked = self.payload()
        leaked["forks"][0]["actions"][0]["description"] = (
            "Try 1/2 as the candidate before checking the constraints."
        )

        with self.assertRaisesRegex(ValueError, "reference answer"):
            parse_answer_masked_graph(
                leaked,
                problem="Find the requested value.",
                reference_solution=r"The value is \boxed{0.5}.",
            )

    def test_graph_rejects_unicode_square_root_answer_alias(self):
        leaked = self.payload()
        leaked["forks"][0]["actions"][0]["description"] = (
            "Try √2 as the candidate before checking the constraints."
        )

        with self.assertRaisesRegex(ValueError, "reference answer"):
            parse_answer_masked_graph(
                leaked,
                problem="Find the requested value.",
                reference_solution=r"The value is \boxed{\sqrt{2}}.",
            )

    def test_graph_rejects_unbraced_latex_fraction_answer_alias(self):
        leaked = self.payload()
        leaked["forks"][0]["actions"][0]["description"] = (
            "Try 0.5 as the candidate before checking the constraints."
        )

        with self.assertRaisesRegex(ValueError, "reference answer"):
            parse_answer_masked_graph(
                leaked,
                problem="Find the requested value.",
                reference_solution=r"The value is \boxed{\frac12}.",
            )

    def test_copy_only_graph_can_be_structurally_checked_before_sanitizing(self):
        copied = self.payload()
        copied["forks"][0]["actions"][0]["description"] = (
            "Use the same exact symbolic transformation from the reference."
        )
        reference = "Use the same exact symbolic transformation from the reference before calculating."
        with self.assertRaisesRegex(ValueError, "copies"):
            parse_answer_masked_graph(copied, problem="Solve x.", reference_solution=reference)
        graph = parse_answer_masked_graph(
            copied,
            problem="Solve x.",
            reference_solution=reference,
            check_reference_fragments=False,
        )
        self.assertEqual(len(graph.forks), 1)

    def test_viability_target_removes_invalid_actions(self):
        graph = parse_answer_masked_graph(self.payload(), problem="Solve x.", reference_solution="Therefore \\boxed{7}.")
        target = branch_target(graph.forks[0].actions, {"factor": 0.8, "discriminant": 0.7}, temperature=0.2)
        self.assertAlmostEqual(sum(target), 1.0)
        self.assertEqual(target[-1], 0.0)
        self.assertGreater(target[0], target[1])

    def test_graph_rejects_a_fork_with_no_viable_target_action(self):
        payload = self.payload()
        payload["forks"][0]["actions"] = [
            {
                "action_id": "dead", "description": "Follow an inconsistent case.",
                "status": "dead_end", "validation_test": "Check the assumption.",
                "recovery_action": "Return to the valid constraints.",
            },
            {
                "action_id": "invalid", "description": "Discard a contradictory branch.",
                "status": "invalid", "validation_test": "Find the contradiction.",
                "recovery_action": "Use a consistent alternative.",
            },
        ]
        with self.assertRaisesRegex(ValueError, "non-invalid"):
            parse_answer_masked_graph(
                payload, problem="Solve x.", reference_solution="Therefore \\boxed{7}."
            )

    def test_scaffold_is_answer_masked_and_contains_actions(self):
        graph = parse_answer_masked_graph(
            self.payload(), problem="Solve x.",
            reference_solution="First factor carefully. Therefore \\boxed{7}.",
        )
        scaffold = render_graph_scaffold(graph)
        self.assertIn("Factor the symbolic expression.", scaffold)
        self.assertNotIn("7", scaffold)
        self.assertNotIn("\\boxed", scaffold)

    def test_graph_allows_variable_arity_but_enforces_total_budget(self):
        payload = self.payload()
        payload["forks"][0]["actions"].append({
            "action_id": "substitute",
            "description": "Substitute into a reduced relation.",
            "status": "recoverable",
            "validation_test": "Check every original constraint.",
            "recovery_action": "Return to the original constraint system.",
        })
        graph = parse_answer_masked_graph(
            payload, problem="Solve x.", reference_solution="Therefore \\boxed{7}.",
            max_actions_per_fork=5, graph_budget=4,
        )
        self.assertEqual(len(graph.forks[0].actions), 4)
        with self.assertRaisesRegex(ValueError, "graph_budget"):
            parse_answer_masked_graph(
                payload, problem="Solve x.", reference_solution="Therefore \\boxed{7}.",
                max_actions_per_fork=5, graph_budget=3,
            )
