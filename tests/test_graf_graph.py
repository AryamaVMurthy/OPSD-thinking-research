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

    def test_viability_target_removes_invalid_actions(self):
        graph = parse_answer_masked_graph(self.payload(), problem="Solve x.", reference_solution="Therefore \\boxed{7}.")
        target = branch_target(graph.forks[0].actions, {"factor": 0.8, "discriminant": 0.7}, temperature=0.2)
        self.assertAlmostEqual(sum(target), 1.0)
        self.assertEqual(target[-1], 0.0)
        self.assertGreater(target[0], target[1])

    def test_scaffold_is_answer_masked_and_contains_actions(self):
        graph = parse_answer_masked_graph(
            self.payload(), problem="Solve x.",
            reference_solution="First factor carefully. Therefore \\boxed{7}.",
        )
        scaffold = render_graph_scaffold(graph)
        self.assertIn("Factor the symbolic expression.", scaffold)
        self.assertNotIn("7", scaffold)
        self.assertNotIn("\\boxed", scaffold)
