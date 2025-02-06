import boto3
import json

prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

bedrock=boto3.client(service_name="bedrock-runtime")

# Define the payload without [INST] structure
payload = {
    "prompt": prompt_data,
    "max_gen_len": 512,  # Adjust length as needed
    "temperature": 0.5,  # Controls randomness
    "top_p": 0.9         # Controls diversity
}


body=json.dumps(payload)
model_id="meta.llama3-70b-instruct-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get("body").read())
repsonse_text=response_body['generation']
print(repsonse_text)