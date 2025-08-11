import os
from dotenv import load_dotenv
import json

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
    OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    DATABASE_CONNECTIONS = json.loads(os.getenv('DATABASE_CONNECTIONS_JSON', '{}').replace("'", '"'))
    SQL_DRIVER = 'ODBC Driver 17 for SQL Server'
