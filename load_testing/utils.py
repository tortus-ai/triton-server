from request_schema import (
    parse_pbtxt_to_dict,
    parse_data_for_request,
    validate_request_data_against_schema,
    convert_input_schema_into_request_data_dict,
)
from typing import Dict
import json


def format_data(input_data, model_conf):
    """
    Parse the data dictionary into a format that can be sent in the request.
    """
    data_item = parse_data_for_request(input_data)
    payload = model_conf.copy()
    for sub_dict in payload["inputs"]:
        key = sub_dict["name"]
        data = [data_item[key]]
        sub_dict["data"].append(data)

    if not validate_request_data_against_schema(model_conf, input_data):
        raise ValueError("Data dictionary does not conform to schema")

    return payload


def create_payload(data_path, model_conf_path) -> Dict:

    model_conf = parse_pbtxt_to_dict(model_conf_path)
    schema = convert_input_schema_into_request_data_dict(model_conf)
    with open(data_path, "r") as f:
        data = json.load(f)

    payload = format_data(data, schema)

    return payload
