import json
import logging
import re
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional, Tuple
from zss import Node, distance
from starflow.dataset.metric.base import VLMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowSimilarityMetric(VLMetric):
    """Metric for Flow Similarity. Computes normalized Tree Edit Distance between two flows using the Zhang-Shasha algorithm."""

    def __init__(self, **kwargs):
        pass

    def __call__(
        self, candidates: list[str], references: list[list[str]]
    ) -> dict[str, Any]:
        """Compute flow similarity with and without inputs for a list of candidates and references.

        :param candidates: List of candidate flows in JSON format
        :param references: List of reference flows in JSON format
        :return: Dictionary with flow similarity with and without inputs
        """
        flow_similarities_no_inputs = []
        flow_similarities_with_inputs = []
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
                flow_similarity_no_inputs = self.similarity(
                    candidate_tree_no_inputs, reference_tree_no_inputs
                )
                flow_similarities_no_inputs.append(float(flow_similarity_no_inputs))
            except Exception as e:
                logger.error(f"Error computing flow similarity without inputs: {e}")
                flow_similarities_no_inputs.append(0)

            # compute tree similarity with inputs
            try:
                candidate_tree_with_inputs = FlowTree(
                    candidate_dict, with_inputs=True, decompose_encoded_queries=True
                )
                reference_tree_with_inputs = FlowTree(
                    reference_dict, with_inputs=True, decompose_encoded_queries=True
                )
                flow_similarity_with_inputs = self.similarity(
                    candidate_tree_with_inputs, reference_tree_with_inputs
                )
                flow_similarities_with_inputs.append(float(flow_similarity_with_inputs))
            except Exception as e:
                logger.error(f"Error computing flow similarity with inputs: {e}")
                flow_similarities_with_inputs.append(0)

        # compute macro average similarity (no weighting based on flow size)
        flow_sim_no_inputs = self.get_macro_average_similarity(
            flow_similarities_no_inputs
        )
        flow_sim_with_inputs = self.get_macro_average_similarity(
            flow_similarities_with_inputs
        )

        return {
            "flow_sim_no_inputs": flow_sim_no_inputs,
            "flow_sim_with_inputs": flow_sim_with_inputs,
        }

    @staticmethod
    def get_code_block(flow_json: str) -> str:
        if not "```json" in flow_json:
            return flow_json
        # find the first ```json and return the content between the first and last ```json
        start = flow_json.find("```json") + 7
        end = flow_json.rfind("```")
        return flow_json[start:end].strip()

    @staticmethod
    def remove_json_backticks(flow_json: str) -> str:
        return flow_json.strip().removeprefix("```json").removesuffix("```").strip()

    @staticmethod
    def parse_flow_json_with_fallback(flow_json: str) -> Dict[str, Any]:
        try:
            return json.loads(flow_json)
        except JSONDecodeError:
            # default to empty flow
            return {Constants.TRIGGER: None, Constants.COMPONENTS: []}

    @staticmethod
    def get_macro_average_similarity(similarities: List[float]) -> float:
        return sum(similarities) / len(similarities) if similarities else 0

    ###########################
    # Tree Edit Distance Cost #
    ###########################

    @staticmethod
    def insert_cost(node: Node) -> int:
        """
        Cost of inserting a node into the tree.
        We define "free nodes" that we don't count towards the edit distance.
        """
        if any(node.label == x for x in Constants.FREE_NODES):
            return 0
        elif node.label != "":
            return 1
        else:
            return 0

    @staticmethod
    def remove_cost(node: Node) -> int:
        """
        Cost of removing a node from the tree.
        We define "free nodes" that we don't count towards the edit distance.
        """
        if any(node.label == x for x in Constants.FREE_NODES):
            return 0
        elif node.label != "":
            return 1
        else:
            return 0

    @staticmethod
    def update_cost(node1: Node, node2: Node) -> int:
        """
        Cost of updating a node in the tree.
        We count edits to trigger and components as more costly than other nodes.
        """
        if node1.label == node2.label:
            return 0
        elif any(node1.label == x for x in Constants.TRIGGER_TYPES) or any(
            node2.label == x for x in Constants.TRIGGER_TYPES
        ):
            # special scoring for updating trigger types
            return 2
        elif (
            node1.label.startswith(Constants.ACTION)
            or node1.label.startswith(Constants.FLOW_LOGIC)
            or node1.label.startswith(Constants.SUBFLOW)
            or node2.label.startswith(Constants.ACTION)
            or node2.label.startswith(Constants.FLOW_LOGIC)
            or node2.label.startswith(Constants.SUBFLOW)
        ):
            # special scoring for updating components (action, flow logic, subflow)
            return 2
        else:
            return 1

    ######################
    # Comparison Methods #
    ######################

    def edit_distance(self, flow1: "FlowTree", flow2: "FlowTree") -> float:
        """Compute the edit distance between two flows."""
        if flow1.with_inputs != flow2.with_inputs:
            raise ValueError(
                "Cannot compare a tree with inputs and a tree without inputs"
            )
        elif flow1.decompose_encoded_queries != flow2.decompose_encoded_queries:
            raise ValueError(
                "Cannot compare a tree with decomposed encoded queries and a tree without decomposed encoded queries"
            )
        dist, _ = distance(
            flow1.tree,
            flow2.tree,
            get_children=Node.get_children,
            insert_cost=self.insert_cost,
            remove_cost=self.remove_cost,
            update_cost=self.update_cost,
            return_operations=True,
        )
        return dist

    def similarity(self, flow1: "FlowTree", flow2: "FlowTree") -> float:
        """Compute the similarity between two flows."""
        if len(flow1) == 0 and len(flow2) == 0:
            return 1.0
        return 1 - (self.edit_distance(flow1, flow2) / (len(flow1) + len(flow2)))


class Constants:

    # Flow content
    FLOW = "flow"
    TRIGGER = "trigger"
    COMPONENTS = "components"
    INPUTS = "inputs"
    SUBFLOW_INPUTS = "inputs"
    SUBFLOW_OUTPUTS = "outputs"
    CONTENT = "content"
    # Trigger
    TRIGGER_TYPE = "type"
    TRIGGER_INPUTS = "inputs"

    # record triggers
    RECORD_CREATE_TRIGGER = "record_create"
    RECORD_UPDATE_TRIGGER = "record_update"
    RECORD_CREATE_OR_UPDATE_TRIGGER = "record_create_or_update"

    # scheduled triggers
    DAILY_TRIGGER = "daily"
    WEEKLY_TRIGGER = "weekly"
    MONTHLY_TRIGGER = "monthly"
    REPEAT_TRIGGER = "repeat"
    RUN_ONCE_TRIGGER = "run_once"

    # application triggers
    EMAIL_TRIGGER = "email"
    SLA_TASK_TRIGGER = "sla_task"
    SERVICE_CATALOG_TRIGGER = "service_catalog"
    METRIC_TRIGGER = "metric"
    REST_ASYNC_TRIGGER = "rest_async"

    # trigger categories
    RECORD_TRIGGER = "record_trigger"
    SCHEDULED_TRIGGER = "scheduled_trigger"
    APPLICATION_TRIGGER = "application_trigger"

    RECORD_TRIGGERS = {
        RECORD_CREATE_TRIGGER,
        RECORD_UPDATE_TRIGGER,
        RECORD_CREATE_OR_UPDATE_TRIGGER,
    }
    SCHEDULED_TRIGGERS = {
        DAILY_TRIGGER,
        WEEKLY_TRIGGER,
        MONTHLY_TRIGGER,
        REPEAT_TRIGGER,
        RUN_ONCE_TRIGGER,
    }
    APPLICATION_TRIGGERS = {
        EMAIL_TRIGGER,
        SLA_TASK_TRIGGER,
        SERVICE_CATALOG_TRIGGER,
        METRIC_TRIGGER,
        REST_ASYNC_TRIGGER,
    }
    TRIGGER_CATEGORIES = {RECORD_TRIGGER, SCHEDULED_TRIGGER, APPLICATION_TRIGGER}
    TRIGGER_TYPES = RECORD_TRIGGERS | SCHEDULED_TRIGGERS | APPLICATION_TRIGGERS

    # Subflow inputs / outputs
    SUBFLOW_INPUT_TYPE = "type"
    ITEMS = "items"

    # Component categories
    FLOW_LOGIC = "flowlogic"
    ACTION = "action"
    SUBFLOW = "subflow"

    # Component Keys
    CATEGORY = "category"
    DEFINITION = "definition"
    SCOPE = "scope"
    ORDER = "order"
    BLOCK = "block"

    # Input types
    FIELD = "field"
    FIELD_NAME = "name"
    FIELD_OPERATOR = "operator"
    FIELD_VALUE = "value"

    # record / reference inputs
    REFERENCE = "reference"
    RECORDS = "records"

    # condition Inputs
    CONDITION_INPUT = "condition"
    CONDITIONS_INPUT = "conditions"
    EMAIL_CONDITIONS_INPUT = "email_conditions"
    CONDITION_INPUT_NAMES = {CONDITION_INPUT, CONDITIONS_INPUT, EMAIL_CONDITIONS_INPUT}

    # field values
    VALUES_INPUT = "values"
    FIELDS_INPUT = "fields"
    FIELD_VALUES_INPUT = "field_values"
    VALUE_INPUT_NAMES = {FIELDS_INPUT, VALUES_INPUT, FIELD_VALUES_INPUT}

    # Logical Operators
    LOGICAL_OPERATORS = ["^EQ", "^OR", "^NQ", "^", "OR"]

    # Comparison Operators
    COMPARISON_OPERATORS = [
        "VALCHANGES",
        "RELATIVEGT",
        "RELATIVEGE",
        "RELATIVELE",
        "RELATIVELT",
        "BETWEEN",
        "CHANGESTO",
        "CHANGESFROM",
        "EMPTYSTRING",
        "ISEMPTY",
        "ISNOTEMPTY",
        "STARTSWITH",
        "ENDSWITH",
        "SAMEAS",
        "NOTON",
        "NOT LIKE",
        "NOT IN",
        "LIKE",
        "NOT",
        "IN",
        "ON",
        "!=",
        ">=",
        "<=",
        "=",
        ">",
        "<",
    ]

    # removing "free nodes", including `flow`, `trigger`, `components`, `inputs`, and `outputs`
    FREE_NODES = {FLOW, TRIGGER, COMPONENTS, INPUTS, SUBFLOW_INPUTS, SUBFLOW_OUTPUTS}
    # add all condition inputs and field value inputs
    FREE_NODES.update(CONDITION_INPUT_NAMES, VALUE_INPUT_NAMES)


class FlowTree:
    def __init__(
        self,
        flow_dict: Dict[str, Any],
        with_inputs: bool = True,
        decompose_encoded_queries: bool = True,
    ) -> None:

        if with_inputs is False and decompose_encoded_queries is True:
            logger.warning(
                "Decomposing encoded queries without inputs will not have any effect. Setting `decompose_encoded_queries` to False."
            )
            decompose_encoded_queries = False

        self.with_inputs = with_inputs
        self.decompose_encoded_queries = decompose_encoded_queries
        self.tree = self.get_tree_from_flow(flow_dict)

    ##################
    # Inputs Related #
    ##################

    def get_tree_from_parsed_parts(
        self, parsed_parts: List[Dict[str, str]], logical_operators: List[str]
    ) -> Node:

        # first part tree
        first_part = parsed_parts[0]
        first_part_tree = (
            Node(first_part.get(Constants.FIELD, ""))
            .addkid(Node(first_part.get(Constants.FIELD_OPERATOR, "^EQ")))
            .addkid(Node(first_part.get(Constants.FIELD_VALUE, "")))
        )

        # base case -> no logical operators
        if len(logical_operators) == 0:
            return first_part_tree

        # if there are logical operators, we recursively build the tree
        base_tree = Node(logical_operators[0])
        base_tree.addkid(first_part_tree)
        subtree = self.get_tree_from_parsed_parts(
            parsed_parts[1:], logical_operators[1:]
        )
        base_tree.addkid(subtree)
        return base_tree

    def get_tree_from_encoded_query(self, encoded_query: str) -> Node:
        parsed_encoded_query_parts, encoded_query_logical_operators = (
            parse_encoded_query_field(encoded_query)
        )
        if (
            parsed_encoded_query_parts is None
            and encoded_query_logical_operators is None
        ):
            return Node("")  # alternatively, Node("^EQ")
        # for our first version of the tree, we do not consider the order of the logical operators or the encoded query parts
        # in order for the similarity metric to be order agnostic, we sort so that the order of the parts does not affect the similarity
        parsed_encoded_query_parts = sorted(
            parsed_encoded_query_parts, key=lambda x: x.get(Constants.FIELD, "")
        )
        encoded_query_logical_operators = sorted(encoded_query_logical_operators)
        encoded_query_tree = self.get_tree_from_parsed_parts(
            parsed_encoded_query_parts, encoded_query_logical_operators
        )
        return encoded_query_tree

    def get_tree_from_single_input(self, input_json: Dict[str, str]) -> Node:
        input_name = input_json.get(Constants.FIELD_NAME, "")
        input_value = str(input_json.get(Constants.FIELD_VALUE, ""))
        if self.decompose_encoded_queries and (
            input_name in Constants.CONDITION_INPUT_NAMES
            or input_name in Constants.VALUE_INPUT_NAMES
        ):
            # encoded query, we have a `condition` or `value` node, along with the decomposed encoded query tree as child
            input_tree = Node(input_name)
            input_value_tree = self.get_tree_from_encoded_query(input_value)
            input_tree.addkid(input_value_tree)
        else:
            # simple value, we have a single node with `name: value` label
            input_node_label = input_name + ": " + input_value
            input_tree = Node(input_node_label)
        return input_tree

    def get_tree_from_inputs(self, inputs_json: List[Dict[str, str]]):
        inputs_tree = Node(Constants.INPUTS)
        # since the order of the inputs is not important for similarity computation, but it is important for the tree structure,
        # we sort the inputs by their name to ensure that the order of the inputs does not affect the similarity
        inputs_json = sorted(inputs_json, key=lambda x: x.get(Constants.FIELD_NAME, ""))
        for input_json in inputs_json:
            input_tree = self.get_tree_from_single_input(input_json)
            inputs_tree.addkid(input_tree)
        return inputs_tree

    ########################
    # Flow Content Related #
    ########################

    def get_tree_from_trigger(self, trigger_json: Optional[Dict[str, Any]]) -> Node:
        trigger_tree = Node(Constants.TRIGGER)
        if trigger_json is None:
            return trigger_tree

        # trigger type
        trigger_type_tree = Node(trigger_json.get(Constants.TRIGGER_TYPE, ""))

        # trigger inputs
        if self.with_inputs and Constants.TRIGGER_INPUTS in trigger_json:
            trigger_inputs_tree = self.get_tree_from_inputs(
                trigger_json.get(Constants.TRIGGER_INPUTS, [])
            )
            trigger_type_tree.addkid(trigger_inputs_tree)

        trigger_tree.addkid(trigger_type_tree)
        return trigger_tree

    def get_tree_from_subflow_io(
        self,
        subflow_io_json: Optional[Dict[str, Any]],
        subflow_io_type: Literal[Constants.SUBFLOW_INPUTS, Constants.SUBFLOW_OUTPUTS],  # type: ignore
    ) -> Node:
        subflow_io_tree = Node(subflow_io_type)
        if subflow_io_json is None:
            return subflow_io_tree

        for item in subflow_io_json.get(Constants.ITEMS, []):
            input_name = item.get(Constants.FIELD_NAME, "")
            input_type = item.get(Constants.SUBFLOW_INPUT_TYPE, "string")
            input_repr = f"{input_name}: {input_type}"
            if (
                input_type in (Constants.REFERENCE, Constants.RECORDS)
                and Constants.REFERENCE in item
            ):
                input_reference = item.get(Constants.REFERENCE, "task")
                input_repr += f" [{input_reference}]"
            subflow_io_tree.addkid(Node(input_repr))

        return subflow_io_tree

    def get_tree_from_component(
        self, component_json: Dict[str, Any], components_json: List[Dict[str, Any]]
    ) -> Node:

        # get component info
        component_category = component_json.get(
            Constants.CATEGORY, "action"
        )  # default to action
        component_definition = component_json.get(Constants.DEFINITION, "")
        if component_category == Constants.FLOW_LOGIC:
            # HACK: we prompt the model with camelCase for flowlogic elements, but in the reference dataset, they are UPPERCASE
            # we convert to UPPERCASE here
            component_definition = component_definition.upper()
        component_scope = component_json.get(
            Constants.SCOPE, "global"
        )  # default to global
        component_order = component_json.get(
            Constants.ORDER, -100
        )  # default to int value that won't be used in flows
        component_label = (
            f"{component_category}___{component_definition}___{component_scope}"
        )
        component_tree = Node(component_label)

        # add inputs
        if self.with_inputs and Constants.INPUTS in component_json:
            component_inputs_tree = self.get_tree_from_inputs(
                component_json.get(Constants.INPUTS, [])
            )
            component_tree.addkid(component_inputs_tree)

        # if component has children, add them
        for comp in components_json:
            if (
                comp.get(Constants.ORDER, -999) <= component_order
            ):  # get order or default to -999 (different value from default component order of -100)
                continue
            elif comp.get(Constants.BLOCK, -999) == component_order:
                # TODO: do we need to truncate the list of components?
                child_component_tree = self.get_tree_from_component(
                    comp, components_json=components_json
                )
                component_tree.addkid(child_component_tree)

        return component_tree

    def get_tree_from_components(self, components_json: List[Dict[str, Any]]) -> Node:
        components_tree = Node(Constants.COMPONENTS)
        for i, component_json in enumerate(components_json):
            if component_json.get(Constants.BLOCK) is None:
                # TODO: do we need to truncate the list of components?
                component_tree = self.get_tree_from_component(
                    component_json, components_json=components_json
                )
                components_tree.addkid(component_tree)
        return components_tree

    def get_tree_from_flow(self, flow_json: Dict[str, Any]) -> Node:
        flow_tree = Node(Constants.FLOW)

        # trigger
        if Constants.TRIGGER in flow_json:
            trigger_tree = self.get_tree_from_trigger(
                flow_json.get(Constants.TRIGGER, None)
            )
            flow_tree.addkid(trigger_tree)

        if Constants.SUBFLOW_INPUTS in flow_json:
            subflow_inputs_tree = self.get_tree_from_subflow_io(
                flow_json.get(Constants.SUBFLOW_INPUTS, []),
                subflow_io_type=Constants.SUBFLOW_INPUTS,
            )
            flow_tree.addkid(subflow_inputs_tree)
        if Constants.SUBFLOW_OUTPUTS in flow_json:
            subflow_outputs_tree = self.get_tree_from_subflow_io(
                flow_json.get(Constants.SUBFLOW_OUTPUTS, []),
                subflow_io_type=Constants.SUBFLOW_OUTPUTS,
            )
            flow_tree.addkid(subflow_outputs_tree)

        # components
        if Constants.COMPONENTS in flow_json:
            components_tree = self.get_tree_from_components(
                flow_json.get(Constants.COMPONENTS, [])
            )
            flow_tree.addkid(components_tree)

        return flow_tree

    #################
    # Magic Methods #
    #################

    def __len__(self) -> int:
        flow_nodes = [elem for elem in str(self.tree).split("\n")]
        flow_nodes = [elem for elem in flow_nodes if elem != ""]
        # keep only rows starting with `digit` + ":"
        flow_nodes = [elem for elem in flow_nodes if re.match(r"^\d+:", elem)]
        flow_nodes = [elem.split(":")[1] for elem in flow_nodes]
        flow_nodes = [elem for elem in flow_nodes if elem not in Constants.FREE_NODES]
        return len(flow_nodes)


def parse_encoded_query_field(
    input_str: str,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[List[str]]]:
    """Parse the encoded query field into sections. Return None if the input is not valid.

    :param input_str: Encoded query field
    :return: Tuple containing the parsed parts and the logical operators found
    """
    # known exceptions
    if input_str == "^EQ":
        return None, None
    elif input_str == "a to z":
        return None, None

    # if input startswith logical operator, remove it
    for logical_operator in Constants.LOGICAL_OPERATORS:
        input_str.removeprefix(logical_operator)

    # split by logical operators
    field_parts, logical_operators = split_encoded_query_field_with_logical_operators(
        input_str
    )

    parsed_parts = []
    for part in field_parts:
        # split by comparison operators
        comparison_parts, comparison_operators = (
            split_encoded_query_field_with_comparison_operators(part)
        )
        if len(comparison_parts) != 2 or len(comparison_operators) != 1:
            return None, None
        parsed_parts.append(
            {
                "field": comparison_parts[0],
                "operator": comparison_operators[0],
                "value": comparison_parts[1],
            }
        )

    return parsed_parts, logical_operators


def split_encoded_query_field_with_operators(
    input_str: str, operators: List[str], exclusion_pattern: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """Split the encoded query field into parts using the operators as separators.

    :param input_str: Encoded query field
    :param operators: List of operators used as separators
    :param exclusion_pattern: Used to exclude some example from the pattern, defaults to None
    :return: Tuple containing the split parts and the operators found
    """
    # Create a regex pattern by joining the separators with '|'
    operator_pattern = "|".join(re.escape(operator) for operator in operators)
    if exclusion_pattern is not None:
        operator_pattern = exclusion_pattern + "(" + operator_pattern + ")"

    # Split the input string using the separator pattern
    operators_found = re.findall(operator_pattern, input_str)
    split_result = re.split(operator_pattern, input_str)

    # make sure the operator is not in the split result
    split_result = [
        elem for elem in split_result if elem.strip() not in operators_found
    ]

    return split_result, operators_found


def split_encoded_query_field_with_logical_operators(
    input_str: str,
) -> Tuple[List[str], List[str]]:
    """Split the encoded query field into parts using the logical operators as separators.

    :param input_str: Encoded query field
    :return: Tuple containing the split parts and the logical operators found
    """
    return split_encoded_query_field_with_operators(
        input_str, Constants.LOGICAL_OPERATORS
    )


def split_encoded_query_field_with_comparison_operators(
    input_str: str,
) -> Tuple[List[str], List[str]]:
    """Split the encoded query field into parts using the comparison operators as separators.

    :param input_str: Encoded query field
    :return: Tuple containing the split parts and the comparison operators found
    """
    return split_encoded_query_field_with_operators(
        input_str, Constants.COMPARISON_OPERATORS, exclusion_pattern="(?<![A-Z=,])"
    )
