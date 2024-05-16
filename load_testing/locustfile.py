# coding=utf-8

"""
Locust file for load testing
"""

import os
import json
from locust import HttpUser, task, tag, constant_throughput, events
from request_schema import \
    parse_pbtxt_to_dict, \
    convert_input_schema_into_request_data_dict, \
    parse_data_for_request, \
    validate_request_data_against_schema


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--schema", type=str, env_var="SCHEMA_PATH", help="Path to locust schema")
    parser.add_argument("--authorization", env_var="AUTH_TOKEN",help="Bearer token")
    parser.add_argument("--data", type=str, env_var="DATA_PATH", help="Path to data file")


@events.test_start.add_listener
def _(environment, **kw):
    print(f"Custom argument supplied: {environment.parsed_options.data}")
    print(f"Custom argument supplied: {environment.parsed_options.schema}")



class LoadTest(HttpUser):
    wait_time = constant_throughput(1./300)

    def _read_env_vars(self):
        """
        Extends the LoadTestBase class to load test an endpoint.
        """
        kwargs = self.environment.parsed_options.__dict__
        self.schema_path = kwargs.get("schema", os.environ.get("SCHEMA_PATH"))
        self.host = kwargs.get("host", os.environ.get("HOST"))
        self.authorization = kwargs.get("authorization", os.environ.get("AUTH_TOKEN"))
        self.data_path = kwargs.get("data", os.environ.get("DATA_PATH"))
        

        # load the dictionary from the data string
        assert os.path.exists(self.schema_path)
        assert os.path.exists(self.data_path)
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        self.data = data

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
        if not validate_request_data_against_schema(self.input_data_body, self.data):
            raise ValueError("Data dictionary does not conform to schema")
        
        data_item = parse_data_for_request(self.data)
        for sub_dict in self.input_data_body["inputs"]:
            key = sub_dict["name"]
            data = [data_item[key]]
            sub_dict["data"].append(data)


    @tag("inference")
    @task
    def predict(self):
        headers = {
            "Content-Type": "application/json"
        }
        self.client.post(self.host, json=self.input_data_body)

    def on_start(self):
        self._read_env_vars()
        self.parse_schema()
        self.get_request_input_data()
        self.verify_schema_inputs_against_data(self.data)
        self.format_data()