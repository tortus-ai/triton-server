# coding=utf-8

"""
Utility functions to parse the model config
into python objects that can be used to
generate load tests
"""

import base64
from typing import List
import numpy as np

def parse_pbtxt_to_dict(filepath):
    """
    This function takes a defined .pbtxt file and parses it into a dictionary.
    This dictionary can be used to populate the load testing class automatically.
    """
    config = {}
    current_list = None
    current_dict = None

    with open(filepath, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:
                continue  # Skip empty lines
            if stripped_line.endswith('['):
                key = stripped_line.split('[')[0].strip()
                config[key] = []
                current_list = config[key]
            elif stripped_line == ']':
                current_list = None
                current_dict = None
            elif current_list is not None and stripped_line.endswith('{'):
                current_dict = {}
                current_list.append(current_dict)
            elif current_dict is not None and stripped_line == '}':
                current_dict = None
            elif current_dict is not None:
                if ':' in stripped_line:
                    key, value = stripped_line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')  # Remove any potential quotes
                    if value.isdigit():  # Handle numeric values
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:  # Handle float values
                        value = float(value)
                    current_dict[key] = value
                else:
                    continue  # Skip malformed lines within dictionaries
            else:
                if ':' in stripped_line:
                    key, value = stripped_line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')  # Remove any potential quotes
                    if value.isdigit():  # Handle numeric values
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:  # Handle float values
                        value = float(value)
                    config[key] = value
                else:
                    continue  # Skip malformed lines

    return config

def map_data_type_to_request_type(data_type:str):
    """
    Map the data type from the schema to the request type.
    See: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes
    :param data_type: str, config data type
    :return: str,
    """
    if data_type == "TYPE_STRING":
        return "BYTES"
    else:
        return data_type.replace("TYPE_", "")

def convert_input_schema_into_request_data_dict(schema:dict):
    """
    Parse the schema dictionary into a dictionary that can be used to send requests.

    :param schema: dict,
    :return: dict,
    """
    base_dict = {"inputs": []}
    input_dict = schema["input"] # list of dictionaries
    for input_item in input_dict:
        input_data = {
            "name": input_item["name"],
            "shape": eval(input_item["dims"]), # converts string to list of ints
            "datatype": map_data_type_to_request_type(input_item["data_type"]),
            "data": []
        }
        base_dict["inputs"].append(input_data) 
    return base_dict

def parse_data_for_request(data:dict):
    """
    Parse the data dictionary into a format that can be sent in the request.
    """
    output_data = {}
    for key, value in data.items():
        if isinstance(value, str) and value.split(".")[1] in [".png", ".jpg"]:
            # If the value is a path to an image, convert to base64
            with open(value, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
                output_data[key] = image_base64
        else:
            output_data[key] = eval(value)
    
    return output_data

def is_correct_type(value, type_str:str):
    """
    Check input data type against the expected type.
    """
    # Mapping of type strings to actual Python data types
    type_mapping = {
        "BOOL": bool,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "FLOAT16": np.float16,
        "FLOAT": float,
        "DOUBLE": float,
        "BYTES": bytes
    }

    expected_type = type_mapping.get(type_str.strip().upper(), None)

    if expected_type is None:
        raise ValueError(f"Unsupported type string: {type_str}")

    return isinstance(value, expected_type)

def validate_request_data_against_schema(schema_dict:dict, data:dict):
    """
    Validate that the data dictionary conforms to the schema.
    """
    inputs = schema_dict["inputs"]
    for input_dict in inputs:
        name = input_dict["name"]
        data_type = input_dict["datatype"]
        value = data.get(name, None)
        if value is None:
            raise ValueError(f"Input {name} not found in data dictionary")
        if not is_correct_type(value, data_type):
            raise ValueError(f"Input {name} is not of the correct type {data_type}")
    return True
