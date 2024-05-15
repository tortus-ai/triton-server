# coding=utf-8

"""

"""

import base64
import os
from abc import ABC
from Typing import Any

from locust import HttpUser, task, tag, constant_throughput, events

from request_schema import \
    parse_pbtxt_to_dict, \
    convert_input_schema_into_request_data_dict, \
    parse_data_for_request

class LoadTestBase(ABC):
    def __init__(self,
                 schema_path:str,
                 host:str,
                 authorization:str,
                 wait_time:float,
                 data:dict):
        assert os.path.exists(schema_path), f"Schema file for model not found at {schema_path}"
        self.schema_path = schema_path
        self.parse_schema()
        self.get_request_input_data()
        self.verify_schema_inputs_against_data(data)
        self.format_data()
        self.host = host
        self.authorization = authorization
        self.wait_time = wait_time

    def parse_schema(self):
        """
        Convert the .pbtxt file for a given Triton model
        into a dictionary.
        """
        self.schema = parse_pbtxt_to_dict(self.schema_path)

    def get_request_input_data(self):
        """
        Method that parses the expected input dictionary
        for the request from the model schema
        """
        self.input_data_body = convert_input_schema_into_request_data_dict(self.schema)

    def verify_schema_inputs_against_data(self, data:dict):
        """
        For a given schema, verify that the provided
        data dictionary is valid.
        :param data: dict, data to be sent in the request
        """
        inputs = self.input_data_body["inputs"] # List of dicts
        for input_dict in inputs:
            assert input_dict["name"] in data.keys(), f"Input {input_dict['name']} not found in data dictionary"

        self.data = data

    def format_data(self):
        """
        Parse the data dictionary into a format that can be sent in the request. 
        """
        data_item = parse_data_for_request(self.data)
        for sub_dict in self.input_data_body["inputs"]:
            key = sub_dict["name"]
            data = data_item[key]
            sub_dict["data"].append(data)

    def on_start(self):
        """
        Method that runs at the beginning of the test.
        """
        raise NotImplementedError
