# coding=utf-8

"""
Locust file for load testing
"""

import os
import json
from locust import HttpUser, task, tag, constant_throughput, events, LoadTestShape
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
    parser.add_argument("--starting-users", type=int, env_var="STARTING_USERS", help="Number of users to start with", default=100)
    parser.add_argument("--bulk-ramp", type=int, env_var="BULKRAMP", help="How many users to bulk add at a given timestep", default=10)
    parser.add_argument("--bulk-interval", type=float, env_var="BULKINTERVAL", help="How often to add users in seconds", default=60)

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
        data_item = parse_data_for_request(self.data)
        for sub_dict in self.input_data_body["inputs"]:
            key = sub_dict["name"]
            data = [data_item[key]]
            sub_dict["data"].append(data)
        
        if not validate_request_data_against_schema(self.input_data_body, self.data):
            raise ValueError("Data dictionary does not conform to schema")



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
        self.format_data()
        self.verify_schema_inputs_against_data(self.data)


class CustomLoadShape(LoadTestShape):
    """
    A custom load shape to spawn 100 users immediately and then 10 users every minute.
    """

    def tick(self):
        starting_users = self.runner.environment.parsed_options.starting_users
        bulk_interval = self.runner.environment.parsed_options.bulk_interval
        bulk_ramp = self.runner.environment.parsed_options.bulk_ramp
        run_time = self.get_run_time()
        
        if run_time < 60:  # First minute
            return starting_users, starting_users  # Start with 100 users
        else:
            users = 100 + ((run_time) // bulk_interval) * bulk_ramp  # 10 users every minute
            spawn_rate = bulk_ramp # we spawn all the users at once
            return users, spawn_rate
