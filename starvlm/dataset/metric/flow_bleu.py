from typing import Any
from starvlm.dataset.metric.flow_similarity import (
    FlowSimilarityMetric,
    FlowTree,
    Constants,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowBLEUMetric(FlowSimilarityMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.EXCLUSION_LIST = [
            (Constants.FLOW, Constants.TRIGGER),
            (Constants.FLOW, Constants.COMPONENTS),
        ]

    def __call__(
        self, candidates: list[str], references: list[list[str]]
    ) -> dict[str, Any]:
        """Compute flow similarity with and without inputs for a list of candidates and references.

        :param candidates: List of candidate flows in JSON format
        :param references: List of reference flows in JSON format
        :return: Dictionary with flow similarity with and without inputs
        """
        all_flow_bleu_no_inputs = []
        all_flow_bleu_with_inputs = []
        for candidate, reference in zip(candidates, references):

            # assuming there is always one (and only one) reference
            # we pick the first one in any case
            if len(reference) == 0:
                raise ValueError("No reference provided")
            reference = reference[0]

            # get code block
            candidate = self.get_code_block(candidate)
            reference = self.get_code_block(reference)

            # remove triple backticks
            candidate = self.remove_json_backticks(candidate)
            reference = self.remove_json_backticks(reference)

            # parse flows
            candidate_dict = self.parse_flow_json_with_fallback(candidate)
            reference_dict = self.parse_flow_json_with_fallback(reference)

            # compute tree similarity without inputs
            try:
                candidate_tree_no_inputs = FlowTree(
                    candidate_dict, with_inputs=False, decompose_encoded_queries=False
                )
                reference_tree_no_inputs = FlowTree(
                    reference_dict, with_inputs=False, decompose_encoded_queries=False
                )
                flow_bleu_no_inputs = self.compute_treebleu(
                    candidate_tree_no_inputs, reference_tree_no_inputs
                )
                all_flow_bleu_no_inputs.append(float(flow_bleu_no_inputs))
            except Exception as e:
                logger.error(f"Error computing FlowBLEU without inputs: {e}")
                all_flow_bleu_no_inputs.append(0)

            # compute tree similarity with inputs
            try:
                candidate_tree_with_inputs = FlowTree(
                    candidate_dict, with_inputs=True, decompose_encoded_queries=True
                )
                reference_tree_with_inputs = FlowTree(
                    reference_dict, with_inputs=True, decompose_encoded_queries=True
                )
                flow_bleu_with_inputs = self.compute_treebleu(
                    candidate_tree_with_inputs, reference_tree_with_inputs
                )
                all_flow_bleu_with_inputs.append(float(flow_bleu_with_inputs))
            except Exception as e:
                logger.error(f"Error computing FlowBLEU with inputs: {e}")
                all_flow_bleu_with_inputs.append(0)

        # compute macro average similarity (no weighting based on flow size)
        flow_bleu_no_inputs = self.get_macro_average_similarity(all_flow_bleu_no_inputs)
        flow_bleu_with_inputs = self.get_macro_average_similarity(
            all_flow_bleu_with_inputs
        )

        return {
            "flow_bleu_no_inputs": flow_bleu_no_inputs,
            "flow_bleu_with_inputs": flow_bleu_with_inputs,
        }

    def extract_1_height_subtrees(self, flow_tree: FlowTree) -> set[tuple[str, str]]:
        """
        Extracts all 1-height subtrees (parent-child relationships) from a zss.Node tree.
        Borrowed from https://github.com/ServiceNow/star-markup/blob/003f04e4ce28bc18db2f09f88e76f64f77f30b4e/starweb_bench/eval/eval_codeedit.py#L150C1-L164C51

        :param tree: Flow tree
        :return: Set of (parent, child) tuples
        """
        subtrees = set()

        def traverse(node):
            for child in node.children:
                subtrees.add((node.label, child.label))  # Store (parent, child) tuple
                traverse(child)

        traverse(flow_tree.tree)
        return subtrees

    def compute_treebleu(
        self, generated_flow_tree: FlowTree, reference_flow_tree: FlowTree
    ) -> float:
        """
        Computes the TreeBLEU score given a generated and reference zss.Node tree.
        Borrowed from https://github.com/ServiceNow/star-markup/blob/003f04e4ce28bc18db2f09f88e76f64f77f30b4e/starweb_bench/eval/eval_codeedit.py#L150C1-L164C51

        :param generated_tree: Generated flow tree
        :param reference_tree: Reference flow tree
        :return: TreeBLEU score
        """
        S_t = self.extract_1_height_subtrees(generated_flow_tree)
        S_tr = self.extract_1_height_subtrees(reference_flow_tree)

        # remove "free" edges
        S_t = {edge for edge in S_t if edge not in self.EXCLUSION_LIST}
        S_tr = {edge for edge in S_tr if edge not in self.EXCLUSION_LIST}

        if not S_tr:
            return 0.0

        return len(S_t.intersection(S_tr)) / len(S_tr)
