#%%
import os
import json
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# Load environment variables
load_dotenv()

# Constants
EMBEDDINGS_MODEL = "text-embedding-3-large"
EMBEDDINGS_DIMENSION = 256
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("NEO_DB_HOST_DEV")
DB_USERNAME = os.getenv("NEO_DB_USER_DEV")
DB_PASSWORD = os.getenv("NEO_DB_PASSWORD_DEV")
DB_DATABASE = os.getenv("NEO_DB_DATABASE_DEV")

client = OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class KnowledgeGraphConfig:
    entity_types: Dict[str, str]
    relation_types: Dict[str, str]
    entity_relationship_match: Dict[str, str]

@dataclass
class Product:
    id: str
    title: str
    description: str
    price_amount: float
    price_currency: str
    similarity: float = 0.0

class EmbeddingModel:
    @staticmethod
    def create_embedding(text: str) -> List[float]:
        result = client.embeddings.create(model=EMBEDDINGS_MODEL, input=text, dimensions=EMBEDDINGS_DIMENSION)
        embedding = result.data[0].embedding
        return embedding

class GraphOperations:
    def __init__(self, driver):
        self.driver = driver

    def similarity_search(self, embedding: List[float], threshold: float = 0.5) -> List[Product]:
            # Ensure the embedding is a list of floats
            if not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
                raise ValueError("Embedding must be a list of floats")

            # Debug: Print the embedding and threshold
            print(f"Embedding: {embedding}")
            print(f"Threshold: {threshold}")

            # Neo4j query using native vector similarity, including the score in the RETURN statement
            query = '''
                    WITH $embedding AS inputEmbedding
                    MATCH (p:Product)
                    WITH p, vector.similarity.cosine(inputEmbedding, p.embedding) AS similarity
                    WHERE similarity > $threshold
                    RETURN p, similarity
                    ORDER BY similarity DESC
                    LIMIT 10
                    '''

            # Execute the query
            with self.driver.session(database=DB_DATABASE) as session:
                result = session.run(query, embedding=embedding, threshold=threshold)
                records = list(result)  # Convert the result to a list immediately

            # Debug: Print the number of records found
            print(f"Number of records found: {len(records)}")

            # Parse the result and collect the matched products with similarity scores
            return [
                Product(
                    id=r['p']['productId'],
                    title=r['p']['title'],
                    description=r['p']['description'],
                    price_amount=r['p']['priceAmount'],
                    price_currency=r['p']['priceCurrency'],
                    similarity=r['similarity']
                    ) for r in records
                ]

prompt = "Matinique Jakkesæt i Navy"
# Use Neo4j's official driver to connect to the database
driver = GraphDatabase.driver(
    DB_URL,
    auth=(DB_USERNAME, DB_PASSWORD)
)
graph_operations = GraphOperations(driver)

# Generate the embedding for the given prompt
embedding = EmbeddingModel.create_embedding(prompt)

# Perform similarity search
returned_products = graph_operations.similarity_search(embedding, threshold=0.7)
print(returned_products)

#%%
# Define your Cypher query to get all labels
cypher_query = "CALL db.labels()"

# Execute the query
result = graph_operations.run_query(cypher_query)

# Process and print the results
print("Labels in the database:")
for record in result:
    label = record['label']
    print(label)