#%
#%%
import os
from src.utils.graph_helpers import connect_to_graph, execute_query
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, DatabaseUnavailable
from langchain.graphs import Neo4jGraph

# Load configuration
load_dotenv()

# Adjust URL if using a secure or self-signed connection scheme
def adjust_neo4j_url(url):
    if url.startswith("bolt+s://") or url.startswith("bolt://"):
        return url.replace("bolt+s://", "neo4j://").replace("bolt://", "neo4j://")
    return url

adjusted_url = adjust_neo4j_url(os.getenv("NEO_DB_HOST_DEV"))

# Debugging: Print loaded environment variables to ensure correctness
print("NEO_DB_HOST_DEV:", adjusted_url)
print("NEO_DB_USER_DEV:", os.getenv("NEO_DB_USER_DEV"))
print("NEO_DB_PASSWORD_DEV:", "********")  # Mask the password for security reasons
print("NEO_DB_DATABASE_DEV:", os.getenv("NEO_DB_DATABASE_DEV"))

# Test connection using Neo4j Python Driver
def test_neo4j_connection():
    driver = GraphDatabase.driver(
        adjusted_url,
        auth=(os.getenv("NEO_DB_USER_DEV"), os.getenv("NEO_DB_PASSWORD_DEV"))
    )
    try:
        # Explicitly use the database provided by the environment variable
        database_name = os.getenv("NEO_DB_DATABASE_DEV")
        if not database_name:
            print("Error: No database name provided.")
            return
        
        with driver.session(database=database_name) as session:
            result = session.run("RETURN 1")
            for record in result:
                print("Connection successful, returned value:", record[0])
    except DatabaseUnavailable as e:
        print("Database is unavailable. Please ensure the database is started and available:", e)
    except ServiceUnavailable as e:
        print("Failed to connect to Neo4j database:", e)
    finally:
        driver.close()

# Run the test connection
test_neo4j_connection()
# %%
# Test connection using Langchain's Neo4jGraph
try:
    graph = Neo4jGraph(
        url=os.getenv("NEO_DB_HOST_DEV"),
        username=os.getenv("NEO_DB_USER_DEV"),
        password=os.getenv("NEO_DB_PASSWORD_DEV"),
        database=os.getenv("NEO_DB_DATABASE_DEV")
    )
    print("Connection to Neo4j established successfully using Neo4jGraph.")
except DatabaseUnavailable as e:
    print("Database is unavailable. Please ensure the database is started and available:", e)
except ServiceUnavailable as e:
    print("Failed to connect to Neo4j database:", e)
# %%
