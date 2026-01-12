from typing import Any, Optional
from starvlm.dataset.metric.flow_similarity import Constants, FlowSimilarityMetric
import json


class FlowMatchMetric(FlowSimilarityMetric):
    """Metric for Flow Match. Computes the number of matching nodes between two flows."""

    def __init__(self, **kwargs) -> None:
        pass

    def __call__(
        self, candidates: list[str], references: list[list[str]]
    ) -> dict[str, Any]:
        trigger_match_scores = []
        component_match_scores = []
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

            try:
                trigger_match_score = self.get_trigger_match(
                    candidate_dict, reference_dict
                )
            except Exception as e:
                trigger_match_score = 0.0
            try:
                component_match_score = self.get_components_match(
                    candidate_dict, reference_dict
                )
            except Exception as e:
                component_match_score = 0.0

            if trigger_match_score is not None:
                trigger_match_scores.append(trigger_match_score)
            if component_match_score is not None:
                component_match_scores.append(component_match_score)

        trigger_match_score = (
            sum(trigger_match_scores) / len(trigger_match_scores)
            if trigger_match_scores
            else 0.0
        )
        component_match_score = (
            sum(component_match_scores) / len(component_match_scores)
            if component_match_scores
            else 0.0
        )

        return {
            "flow_trigger_match_score": trigger_match_score,
            "flow_component_match_score": component_match_score,
        }

    def get_trigger_match(
        self, candidate_dict: dict, reference_dict: dict
    ) -> Optional[bool]:

        candidate_trigger = candidate_dict.get(Constants.TRIGGER, None)
        reference_trigger = reference_dict.get(Constants.TRIGGER, None)

        if candidate_trigger is None and reference_trigger is None:
            return None

        candidate_trigger_str = "{}"
        if candidate_trigger is not None:
            candidate_trigger_type = candidate_trigger.get(Constants.TRIGGER_TYPE, None)
            # sort candidate inputs
            candidate_inputs = candidate_trigger.get(Constants.INPUTS, [])
            if candidate_inputs:
                candidate_inputs = sorted(
                    candidate_inputs, key=lambda x: x.get(Constants.FIELD_NAME, "")
                )

            candidate_trigger_for_comparison = {
                Constants.TRIGGER_TYPE: candidate_trigger_type,
                Constants.INPUTS: candidate_inputs,
            }
            candidate_trigger_str = json.dumps(candidate_trigger_for_comparison)

        reference_trigger_str = "{}"
        if reference_trigger is not None:
            reference_trigger_type = reference_trigger.get(Constants.TRIGGER_TYPE, None)
            # sort reference inputs
            reference_inputs = reference_trigger.get(Constants.INPUTS, [])
            if reference_inputs:
                reference_inputs = sorted(
                    reference_inputs, key=lambda x: x.get(Constants.FIELD_NAME, "")
                )

            reference_trigger_for_comparison = {
                Constants.TRIGGER_TYPE: reference_trigger_type,
                Constants.INPUTS: reference_inputs,
            }
            reference_trigger_str = json.dumps(reference_trigger_for_comparison)

        return candidate_trigger_str == reference_trigger_str

    def get_components_match(
        self, candidate_dict: dict, reference_dict: dict
    ) -> Optional[bool]:
        candidate_components = candidate_dict.get(Constants.COMPONENTS, [])
        reference_components = reference_dict.get(Constants.COMPONENTS, [])

        if candidate_components is None or reference_components is None:
            return 0.0
        elif not isinstance(candidate_components, list) or not isinstance(
            reference_components, list
        ):
            return 0.0

        def get_component_repr(component_dict: dict) -> str:
            component_category = component_dict.get(Constants.CATEGORY, None)
            component_definition = component_dict.get(Constants.DEFINITION, None)
            component_scope = component_dict.get(Constants.SCOPE, None)
            return f"{component_category}___{component_definition}___{component_scope}"

        candidate_components_repr = [
            get_component_repr(component) for component in candidate_components
        ]
        reference_components_repr = [
            get_component_repr(component) for component in reference_components
        ]

        intersection = set(candidate_components_repr) & set(reference_components_repr)
        union = set(candidate_components_repr) | set(reference_components_repr)
        matching_components = len(intersection)
        total_components = len(union)
        if total_components == 0:
            return None
        match_score = matching_components / total_components
        return match_score
