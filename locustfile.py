#Locust load test script for the LLMIS API
#simulates several fake users and their behavior
import os

from locust import HttpUser, task, constant

#quantization toggle
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "0") == "1"

class GenerateUser(HttpUser):
    wait_time = constant(0)

    @task
    def generate(self):
        self.client.post(
            "/generate",
            json={"prompt": "ROMEO:", "max_new_tokens": 200, "use_quantization": USE_QUANTIZATION},
        )
