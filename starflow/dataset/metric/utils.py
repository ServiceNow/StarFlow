import csv
import io
import json
import re
import sys
from typing import Optional, Union

import cairosvg
import pydot
import torch
from PIL import Image as PILImage
from timeout_decorator import timeout


def preprocess_latex_table(latex_code: str) -> str:
    if r'```' in latex_code:
        # Only keep the content inside the code block
        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n+```', latex_code, flags=re.DOTALL)
        if match:
            latex_code = match.group(1)

    # Remove LaTeX comments
    latex_code = re.sub(r'%.*', '', latex_code)

    # Remove \label{...} commands
    latex_code = re.sub(r'\\label\{[^}]+\}', '', latex_code)

    # Remove citation commands like \cite{...}, \citep{...}, \citet{...}, etc.
    latex_code = re.sub(r'\\cite\w*\{[^}]+\}', '', latex_code)

    # Remove other common reference commands like \ref{...}, \eqref{...}, \pageref{...}
    latex_code = re.sub(r'\\(ref|eqref|pageref)\{[^}]+\}', '', latex_code)

    # Ensure exactly one space around each & symbol, converting && to & &
    latex_code = re.sub(r'\s*&\s*', ' & ', latex_code)

    # Replace multiple spaces and tabs with a single space, keeping newlines
    latex_code = re.sub(r'[ \t]+', ' ', latex_code)

    # Remove spaces immediately after the first $ and before the second $ in $$...$$ blocks
    latex_code = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'$$\1$$', latex_code)

    # Strip leading and trailing whitespace but preserve internal newlines
    latex_code = latex_code.strip()

    return latex_code


def preprocess_markdown(md_code: str) -> str:
    if r'```' in md_code:
        # Only keep the content inside the code block
        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n+```', md_code, flags=re.DOTALL)
        if match:
            md_code = match.group(1)
    # Remove Markdown comments
    md_code = re.sub(r'<!--.*?-->', '', md_code, flags=re.DOTALL)

    # Remove Markdown links
    md_code = re.sub(r'\[.*?\]\(.*?\)', '', md_code)

    # Replace multiple spaces and tabs with a single space, but keep newlines
    md_code = re.sub(r'[ \t]+', ' ', md_code)

    # Strip leading and trailing whitespace
    md_code = md_code.strip()

    return md_code


@timeout(2)
def cairosvg_wrapper(**kwargs):
    return cairosvg.svg2png(**kwargs)


def validate_plot_svg(svg_code: str):
    if r'```' in svg_code:
        # Only keep the content inside the code block
        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n+```', svg_code, flags=re.DOTALL)
        if match:
            svg_code = match.group(1)
    svg_code = svg_code.encode('utf-8')
    png_output = io.BytesIO()
    try:
        cairosvg_wrapper(bytestring=svg_code, write_to=png_output)
    except Exception as e:
        return None
    png_output.seek(0)
    image = PILImage.open(png_output)
    return image


def validate_graphviz(graphviz_code: str) -> bool:
    # Deprecated function; use validate_graphviz_pydot instead
    if r'```' in graphviz_code:
        # Only keep the content inside the code block
        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n+```', graphviz_code, flags=re.DOTALL)
        if match:
            graphviz_code = match.group(1)
    try:
        graphs = pydot.graph_from_dot_data(graphviz_code)
        if graphs:
            return True
        return False
    except Exception:
        return False


def extract_triplet_from_graphviz(graphviz_code: str) -> Optional[list[tuple[str, Optional[str], str]]]:
    # Deprecated function; use extract_triplet_from_graphviz_pydot instead
    # Validate graphviz code
    if r'```' in graphviz_code:
        # Only keep the content inside the code block
        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n+```', graphviz_code, flags=re.DOTALL)
        if match:
            graphviz_code = match.group(1)
    if not validate_graphviz(graphviz_code):
        return None

    # Remove comments (both single-line // and multi-line /* */)
    def remove_comments(code: str) -> str:
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    clean_code = remove_comments(graphviz_code)

    # Extract node definitions
    # Node regex: ^node_id [attr_list];
    # node_id can be quoted or unquoted
    # ^ and $ ensure that the entire line matches a node definition
    node_pattern = re.compile(
        r'^\s*(?P<id>"[^"]+"|\w+)\s*(\[\s*(?P<attrs>[^\]]+)\])?\s*;$',
        re.MULTILINE
    )

    # Extract edge definitions
    # Edge regex: ^source -> target [attr_list];
    # ^ and $ ensure that the entire line matches an edge definition
    edge_pattern = re.compile(
        r'^\s*(?P<source>"[^"]+"|\w+)\s*->\s*(?P<target>"[^"]+"|\w+)\s*(\[\s*(?P<attrs>[^\]]+)\])?\s*;$',
        re.MULTILINE
    )

    # Parse attributes to extract labels
    def parse_attrs(attr_str: str) -> dict:
        attrs = {}
        if not attr_str:
            return attrs
        # Split by comma, but handle commas inside quotes
        # This regex matches key="value with, comma" or key=value
        attr_pairs = re.findall(r'(\w+)\s*=\s*("(?:\\.|[^"\\])*"|[^,]+)', attr_str)
        for key, value in attr_pairs:
            # Remove surrounding quotes if present and unescape quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
            attrs[key.strip()] = value.strip()
        return attrs

    # Build node id to label mapping
    node_id_to_label = {}
    for match in node_pattern.finditer(clean_code):
        node_id_raw = match.group('id').strip()
        node_id = node_id_raw[1:-1] if node_id_raw.startswith('"') and node_id_raw.endswith('"') else node_id_raw
        attrs_str = match.group('attrs')
        attrs = parse_attrs(attrs_str)
        label = attrs.get('label', node_id)
        node_id_to_label[node_id] = label

    # Extract edges and build triplets
    triplets = []
    for match in edge_pattern.finditer(clean_code):
        source_raw = match.group('source').strip()
        target_raw = match.group('target').strip()
        source_id = source_raw[1:-1] if source_raw.startswith('"') and source_raw.endswith('"') else source_raw
        target_id = target_raw[1:-1] if target_raw.startswith('"') and target_raw.endswith('"') else target_raw
        attrs_str = match.group('attrs')
        attrs = parse_attrs(attrs_str)
        edge_label = attrs.get('label', None)

        # Get source and target labels
        source_label = node_id_to_label.get(source_id, source_id)
        target_label = node_id_to_label.get(target_id, target_id)
        triplets.append((source_label, edge_label, target_label))

    return triplets


def parse_global_defaults(dot_code):
    node_shapes = {}
    current_shape = None
    default_shape = None

    # Regex patterns to match node default settings and node definitions
    default_pattern = re.compile(r'node\s*\[\s*shape\s*=\s*(\w+)', re.IGNORECASE)
    node_pattern = re.compile(r'(\w+)\s*\[.*?label\s*=\s*".*?".*?shape\s*=\s*(\w+).*?\];', re.IGNORECASE)
    node_without_shape_pattern = re.compile(r'(\w+)\s*\[.*?label\s*=\s*".*?".*?\];', re.IGNORECASE)

    # Loop through the DOT file line by line
    for line in dot_code.splitlines():
        # Check for a change in global node defaults (shape)
        default_match = default_pattern.search(line)
        if default_match:
            default_shape = default_match.group(1)

        # Check for specific node definitions with an explicit shape
        node_match = node_pattern.search(line)
        if node_match:
            node_name = node_match.group(1)
            node_shape = node_match.group(2)
            node_shapes[node_name] = node_shape

        # Check for node definitions without an explicit shape
        node_no_shape_match = node_without_shape_pattern.search(line)
        if node_no_shape_match:
            node_name = node_no_shape_match.group(1)
            node_shapes[
                node_name] = default_shape if default_shape else "ellipse"  # Default is "ellipse" if nothing is set

    return node_shapes


def get_node_shape(graphviz_code, node_name) -> Optional[str]:
    node_shapes = parse_global_defaults(graphviz_code)

    if node_name not in node_shapes:
        return 'ellipse'  # Default shape if not found

    return node_shapes[node_name]


def extract_triplet_from_graphviz_pydot(graphviz_code: str,
                                        use_shape: bool = False,
                                        use_name: bool = True) -> Optional[list[tuple[str, Optional[str], str]]]:
    # Use this instead of extract_triplet_from_graphviz
    # extract the code from "digraph" onwards
    assert use_name or use_shape, "At least one of use_name or use_shape must be True"

    if r'```' in graphviz_code:
        # Only keep the content inside the code block
        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n+```', graphviz_code, flags=re.DOTALL)
        if match:
            graphviz_code = match.group(1)

    if "digraph" in graphviz_code:
        graphviz_code = re.sub(r'^.*?digraph', 'digraph', graphviz_code, flags=re.DOTALL)
    elif "graph" in graphviz_code:
        graphviz_code = re.sub(r'^.*?graph', 'graph', graphviz_code, flags=re.DOTALL)

    with suppress_stdout():
        try:
            graphs = pydot.graph_from_dot_data(graphviz_code)
        except Exception:
            return None
    if not graphs:
        return None

    # Assuming there is at least one graph parsed
    graph = graphs[0]
    node_label_map = {node.get_name(): node.get('label') or node.get_name() for node in graph.get_nodes()}

    triplets = []

    for edge in graph.get_edges():
        source = edge.get_source()
        target = edge.get_destination()
        edge_label = edge.get('label')

        source_label = node_label_map.get(source, source)
        target_label = node_label_map.get(target, target)

        source_shape = get_node_shape(graphviz_code, source)
        target_shape = get_node_shape(graphviz_code, target)

        if use_shape:
            if use_name:
                source_label = f"{source_label}#{source_shape}"
                target_label = f"{target_label}#{target_shape}"
            else:
                source_label = source_shape
                target_label = target_shape

        triplets.append((source_label, edge_label, target_label))

    return triplets


def extract_triplet_from_json(json_str: str,
                              use_shape: bool = False,
                              use_name: bool = True) -> Optional[list[tuple[str, Optional[str], str]]]:
    # Parse the JSON string into a Python dictionary
    # Find the start and end of the JSON object
    assert use_name or use_shape, "At least one of use_name or use_shape must be True"

    if r'```' in json_str:
        # Only keep the content inside the code block
        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n+```', json_str, flags=re.DOTALL)
        if match:
            json_str = match.group(1)

    start = json_str.find('{')
    end = json_str.rfind('}')
    if start == -1 or end == -1:
        return None
    # Attempt to parse the JSON object only
    json_str = json_str[start:end + 1]
    try:
        graph_data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    try:
        # Build a mapping from node ID to node label
        node_id_to_label = {}
        nodes = graph_data.get("nodes", [])
        for node in nodes:
            node_id = node.get("id")
            if node_id is None:
                continue  # Skip nodes without an ID
            if use_shape:
                shape = node.get("shape", "rectangle")  # Default to rectangle if shape is missing
                if use_name:
                    label = f"{node.get('label', node_id)}#{shape}"
                else:
                    label = shape
            else:  # use_name is True
                label = node.get("label", node_id)  # Default to node ID if label is missing
            node_id_to_label[node_id] = label

        # Iterate through edges to extract triplets
        triplets = []
        edges = graph_data.get("edges", [])
        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            edge_label = edge.get("label")  # Can be None

            if source_id is None or target_id is None:
                continue  # Skip edges without source or target

            # Remove any trailing semicolons from target_id
            target_id = target_id.rstrip(';')

            # Get labels for source and target; default to IDs if labels are missing
            source_label = node_id_to_label.get(source_id, source_id)
            target_label = node_id_to_label.get(target_id, target_id)

            triplets.append((source_label, edge_label, target_label))

        return triplets
    except Exception:
        return None


def calculate_triplet_f1_score(prediction: list[tuple[str, Optional[str], str]],
                               reference: list[tuple[str, Optional[str], str]]) -> float:
    prediction_set = set(prediction)
    reference_set = set(reference)

    true_positives = len(reference_set & prediction_set)
    false_positives = len(prediction_set - reference_set)
    false_negatives = len(reference_set - prediction_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


class suppress_stdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def flatten_dict(d, parent_key='', sep='#'):
    items = {}
    for k, v in d.items():
        # Create a new key by appending the current key to the parent key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # new_key = k
        if isinstance(v, dict):
            # If the value is a dictionary, recurse
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            # Otherwise, add the key-value pair to the items
            items[new_key] = v
    return items


def json_to_csv_split_headers(data, output_csv_path):
    # Initialize a dictionary to hold the flattened data
    flattened = {}

    # Iterate over each task and its metrics
    for task, metrics in data.items():
        # Flatten the metrics dictionary
        task_metrics = flatten_dict(metrics, parent_key=task, sep='#')
        # Update the main flattened dictionary
        flattened.update(task_metrics)

    # Sort the keys for consistent ordering (optional)
    sorted_keys = sorted(flattened.keys())

    # Split the sorted keys into task names and metric names
    task_names = []
    metric_names = []
    for key in sorted_keys:
        parts = key.split('#')  # Split into task and metric
        task, metric = parts[0], parts[-1]
        task_names.append(task)
        metric_names.append(metric)

    # Write the data to a CSV file with split headers
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the first header row (Task Names)
        writer.writerow(task_names)
        # Write the second header row (Metric Names)
        writer.writerow(metric_names)
        # Write the corresponding metric values
        writer.writerow([flattened[key] for key in sorted_keys])

    print(f"CSV file with split headers has been created at: {output_csv_path}")


def preprocess_bbox_string(box_str: str) -> Optional[tuple[float, float, float, float]]:
    box_str = box_str.strip()
    match = re.search(r'\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]', box_str)
    if match:
        return list(map(float, match.groups()))  # Convert matched groups to floats
    else:
        return None  # Return None if no valid format is found


def bootstrap_std(data: Union[torch.Tensor, list[float]], n_bootstrap=1000, ci=0.95):
    """
    Args:
        data: a list of values
        n_bootstrap: number of bootstrap samples
        ci: confidence interval
    Returns:
        a tuple of std, lower bound, upper bound
    """
    # Convert data to a 1D PyTorch tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    data = data.to(torch.float32)

    n = data.size(0)

    # Generate bootstrap indices: shape (n_bootstrap, n)
    # Each row contains indices for one bootstrap sample
    bootstrap_indices = torch.randint(low=0, high=n, size=(n_bootstrap, n))

    # Gather bootstrap samples: shape (n_bootstrap, n)
    bootstrap_samples = data[bootstrap_indices]

    # Compute the mean of each bootstrap sample: shape (n_bootstrap,)
    bootstrap_means = bootstrap_samples.mean(dim=1)

    # Compute the confidence interval bounds
    lower_bound = torch.quantile(bootstrap_means, (1 - ci) / 2)
    upper_bound = torch.quantile(bootstrap_means, (1 + ci) / 2)

    # Calculate the standard deviation of the bootstrap means
    std = torch.std(bootstrap_means, unbiased=False)

    # Return results as Python floats
    return std.item(), lower_bound.item(), upper_bound.item()


def eval_multi_references(eval_fn: callable, candidates: str, references: list[str]):
    """
    Evaluate a metric function that supports multiple references
    """
    if not references:
        return None
    if len(references) == 1:
        return eval_fn(candidates, references[0])
    max_score = 0
    for ref in references:
        score = eval_fn(candidates, ref)
        max_score = max(max_score, score)
    return max_score


def process_pred_and_reference(candidates: list[str], references: list[list[str]], process_fn: callable):
    return ([process_fn(candidate) for candidate in candidates],
            [[process_fn(reference) for reference in instance] for instance in references])


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


if __name__ == "__main__":
    pass
