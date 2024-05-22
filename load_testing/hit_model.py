from utils import create_payload
from argparse import ArgumentParser
import requests
import time

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--schema_path", help="Path to the schema file")
    parser.add_argument("--data_path", help="Path to the data file")
    parser.add_argument("--host", help="Endpoint name")
    parser.add_argument("--model", help="Model name", default="llama3_8b")
    # TODO use this to run n number of times and avg
    parser.add_argument("--num_runs", help="Model name", default=1, type=int)

    args = parser.parse_args()

    payload = create_payload(args.data_path, args.schema_path)

    API_URL = f"{args.host}/v2/models/{args.model}/infer"
    start = time.time()
    response = requests.post(API_URL, json=payload)
    if response.status_code != 200:
        print(
            f"Request failed with status code {response.status_code}\n{response.text}"
        )

    print("Response from server")
    print(response.json())

    time_taken = time.time() - start
    print(f"Time taken: {time_taken}s")
