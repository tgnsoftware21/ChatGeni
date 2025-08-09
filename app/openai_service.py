import os
from openai import AzureOpenAI
from flask import current_app

def generate_sql_query(prompt):
    

    client = AzureOpenAI(

    api_version=os.getenv('AZURE_API_VERSION'),
    api_key = current_app.config['OPENAI_API_KEY'],

    azure_endpoint=current_app.config['OPENAI_ENDPOINT'],
    )
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    response = client.completions.create(
    model=deployment_name,
    prompt=prompt,
    temperature=0,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    best_of=1,
    stop=["#",";"]
    )
    print(response.choices)
    return response.choices[0].text
