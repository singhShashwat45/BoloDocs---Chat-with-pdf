import boto3
import json

# Define the input text
prompt_data = """
Act as a Shakespeare and write a poem on Generative AI
"""

# Initialize the Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Define the payload matching the "amazon.nova-micro-v1:0" structure
payload = {
    "messages": [
        {
            "role": "user",
            "content": prompt_data  # Input text for the model
        }
    ],
    "inferenceConfig": {
        "max_new_tokens": 1000  # Similar to maxTokens in the previous model
    }
}

# Convert the payload to JSON format
body = json.dumps(payload)

# Specify the model ID
model_id = "amazon.nova-micro-v1:0"

# Invoke the model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

# Parse the response body
response_body = json.loads(response.get("body").read())

# Extract and print the generated text
response_text = response_body['messages'][0]['content']
print(response_text)
