#%%
import os
import json
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from src.utils.amazon_kg_demo_helper import GraphOperations, EmbeddingModel

# Load environment variables
load_dotenv()

# Constants
EMBEDDINGS_MODEL = "text-embedding-3-large"
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

class QueryGenerator:
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config

    def get_system_prompt(self) -> str:
        return f'''
            You are a helpful agent designed to fetch information from a graph database.
            The graph database links products to the following entity types:
            {json.dumps(self.config.entity_types)}
            Each link has one of the following relationships:
            {json.dumps(self.config.relation_types)}
            Please respond with a JSON object that contains the relevant entities for the query.
            Ensure your response is in strict JSON format.
        '''

    def define_query(self, prompt: str, model: str = "gpt-4o") -> Dict:
        system_prompt = self.get_system_prompt()
        try:
            # Send the request to OpenAI API with JSON enforced output
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract the content of the response
            response_content = completion.choices[0].message.content.strip()
            
            # Attempt to parse the response as JSON
            try:
                parsed_response = json.loads(response_content)
                return parsed_response
            except json.JSONDecodeError:
                # If parsing fails, print the response content for debugging and return an empty dictionary
                print(f"Response content was not JSON: {response_content}")
                return {}

        except Exception as e:
            print(f"An error occurred while generating the query: {e}")
            return {}

    def create_cypher_query(self, entities: Dict, similarity_threshold: float) -> str:
        # Generate a Cypher query based on the entities provided
        match_clauses = []
        where_clauses = []

        for entity, value in entities.items():
            if entity in self.config.entity_relationship_match:
                relationship = self.config.entity_relationship_match[entity]
                match_clauses.append(f"OPTIONAL MATCH (p)-[:{relationship}]->({entity})")
                where_clauses.append(f"{entity}.value = '{value}'")

        match_clause = "MATCH (p:Product) " + " ".join(match_clauses)
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        {match_clause}
        {where_clause}
        RETURN p.productId AS id, p.title AS title, p.description AS description, p.priceAmount AS price_amount, p.priceCurrency AS price_currency
        """
        return query.strip()

# Main Functionality
entity_types = {
    "Product": "The type of item, e.g., 'jewelry', 'watch', 'ring'.",
    "Brand": "The brand of the product, e.g., 'La Joaillière'.",
    "Identifier": "Unique identifier related to the product, such as GTIN or SKU.",
    "Webshop": "The webshop where the product is sold, e.g., 'Zalando'.",
    "Country": "Country where the product is available, represented by the country code, e.g., 'US', 'DE'."
}

relation_types = {
    "BELONGS_TO_BRAND": "Relationship between Product and Brand.",
    "HAS_IDENTIFIER": "Relationship between Product and Identifier.",
    "SOLD_BY": "Relationship between Product and Webshop.",
    "AVAILABLE_IN": "Relationship between Product and Country."
}

entity_relationship_match = {
    "Brand": "BELONGS_TO_BRAND",
    "Identifier": "HAS_IDENTIFIER",
    "Webshop": "SOLD_BY",
    "Country": "AVAILABLE_IN"
}

# Create Knowledge Graph Config
config = KnowledgeGraphConfig(entity_types, relation_types, entity_relationship_match)

# Query using LLM to generate entities and Cypher query
query_generator = QueryGenerator(config)
prompt = "Billede - Magic Magnolia (1 Part) Wide - 60 x 40 cm - Premium Print"

entities = query_generator.define_query(prompt)
#%%
if entities:

    driver = GraphDatabase.driver(
        os.getenv("NEO_DB_HOST_DEV"),
        auth=(os.getenv("NEO_DB_USER_DEV"), os.getenv("NEO_DB_PASSWORD_DEV"))
    )

    graph_operations = GraphOperations(driver)

    # Generate the embedding for the given prompt
    embedding = EmbeddingModel.create_embedding(prompt)

    # Perform similarity search
    print(graph_operations.similarity_search(
        embedding
    ))

    # Query the graph
    print("Querying the graph based on LLM output...")
    similarity_threshold = 0
    result = graph_operations.query_graph(entities, query_generator, similarity_threshold)

    # Display results
    if result:
        for r in result:
            print(f"Product ID: {r['id']}, Name: {r['title']}, Description: {r['description']}")
    else:
        print("No products found with the given query.")

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