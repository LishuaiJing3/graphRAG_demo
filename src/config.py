import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    return {
        "url": os.getenv("NEO_DB_HOST_AURA"),
        "username": os.getenv("NEO_DB_USER"),
        "password": os.getenv("NEO_DB_PASSWORD_AURA"),
        "database_name": "neo4j"
    }
