import os
import json
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

# Load environment variables
load_dotenv()

# Constants
EMBEDDINGS_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("NEO_DB_HOST_AURA")
DB_USERNAME = os.getenv("NEO_DB_USER")
DB_PASSWORD = os.getenv("NEO_DB_PASSWORD_AURA")

client = OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class Product:
    id: str
    name: str
    similarity: float = 0.0

@dataclass
class KnowledgeGraphConfig:
    entity_types: Dict[str, str]
    relation_types: Dict[str, str]
    entity_relationship_match: Dict[str, str]

class DataProcessor:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        with open(file_path, 'r') as file:
            jsonData = json.load(file)
        return pd.read_json(file_path), jsonData

    @staticmethod
    def sanitize(text: str) -> str:
        return str(text).replace("'", "").replace('"', '').replace('{', '').replace('}', '')


class GraphConstructor:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=DB_URL,
            username=DB_USERNAME,
            password=DB_PASSWORD
        )

    def add_products_to_db(self, json_data: List[Dict]):
        for i, obj in enumerate(json_data, start=1):
            print(f"{i}. {obj['product_id']} -{obj['relationship']}-> {obj['entity_value']}")
            query = f'''
                MERGE (product:Product {{id: {obj['product_id']}}})
                ON CREATE SET product.name = "{DataProcessor.sanitize(obj['product'])}",
                               product.title = "{DataProcessor.sanitize(obj['TITLE'])}",
                               product.bullet_points = "{DataProcessor.sanitize(obj['BULLET_POINTS'])}",
                               product.size = {DataProcessor.sanitize(obj['PRODUCT_LENGTH'])}
                MERGE (entity:{obj['entity_type']} {{value: "{DataProcessor.sanitize(obj['entity_value'])}"}})
                MERGE (product)-[:{obj['relationship']}]->(entity)
                '''
            self.graph.query(query)


class VectorIndexManager:
    def __init__(self):
        self.url = DB_URL
        self.username = DB_USERNAME
        self.password = DB_PASSWORD
        self.model = EMBEDDINGS_MODEL

    def embed_product_text(self, index_name: str, node_label: str, text_node_properties: List[str]):
        return Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(model=self.model),
            url=self.url,
            username=self.username,
            password=self.password,
            index_name=index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property='embedding',
        )

    def embed_entities(self, df: pd.DataFrame):
        entities_list = df['entity_type'].unique()
        for entity_type in entities_list:
            self.embed_product_text(
                index_name=entity_type,
                node_label=entity_type,
                text_node_properties=['value']
            )

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

            Depending on the user prompt, determine if it is possible to answer with the graph database.
                
            The graph database can match products with multiple relationships to several entities.
            
            Example user input:
            "Which blue clothing items are suitable for adults?"
            
            For each relationship to analyze, add a key-value pair with the key being an exact match for one of the entity types provided, and the value being the value relevant to the user query.
            
            Return your response as a JSON object with key-value pairs representing the relevant entities.
            
            Example output:
            {{
                "color": "blue",
                "category": "clothing",
                "age_group": "adults"
            }}
            
            If there are no relevant entities in the user prompt, return an empty JSON object: {{}}
        '''

    def define_query(self, prompt: str, model: str = "gpt-4") -> Dict:
        system_prompt = self.get_system_prompt()
        try:
            # Send the request to OpenAI API
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract the content of the response
            response_content = completion.choices[0].message.content.strip()
            
            # Check if the response content is valid JSON
            if not response_content:
                raise ValueError("The response from OpenAI API is empty.")

            # Attempt to parse the response as JSON
            return json.loads(response_content)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from OpenAI response: {e}")
            print(f"Response content was: {response_content}")
            return {}
        except Exception as e:
            print(f"An error occurred while generating the query: {e}")
            return {}

    def create_cypher_query(self, entities: Dict, threshold: float = 0) -> str:
        embeddings_data = [f"${key}Embedding AS {key}Embedding" for key in entities]
        query = "WITH " + ",\n".join(embeddings_data)

        match_data = [f"(p)-[:{self.config.entity_relationship_match[key]}]->({key}Var:{key})" for key in entities]
        query += "\nMATCH (p:Product)\nMATCH " + ",\n".join(match_data)

        similarity_data = [
            f"vector.similarity.cosine({key}Var.embedding, ${key}Embedding) > {threshold}"
            for key in entities
        ]
        query += "\nWHERE " + " AND ".join(similarity_data)
        query += "\nRETURN p"
        return query


class GraphOperations:
    def __init__(self, graph_connection):
        self.graph = graph_connection

    def similarity_search(self, embedding: List[float], threshold: float = 0.5) -> List[Product]:
        # Ensure the embedding is a list of floats
        if not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
            raise ValueError("Embedding must be a list of floats")

        # Neo4j query using native vector similarity, including the score in the RETURN statement
        query = '''
                WITH $embedding AS inputEmbedding
                MATCH (p:Product)
                WITH p, vector.similarity.cosine(inputEmbedding, p.embedding) AS similarity
                WHERE similarity > $threshold
                RETURN p, similarity
                '''

        # Execute the query
        result = self.graph.query(query, params={'embedding': embedding, 'threshold': threshold})

        # Parse the result and collect the matched products with similarity scores
        return [Product(id=r['p']['id'], name=r['p']['name'], similarity=r['similarity']) for r in result]

    def query_graph(self, response: Dict, query_generator: QueryGenerator, similarity_threshold: float):
        print(response)
        query = query_generator.create_cypher_query(response, similarity_threshold)
        print(query)

        # Generate embeddings for all values in the response dictionary
        embeddings_params = {f"{key}Embedding": EmbeddingModel.create_embedding(value) for key, value in response.items()}

        # Debug prints to verify embeddings parameters
        for key, value in embeddings_params.items():
            print(f"Key: {key}, Type: {type(value)}, Value: {value}")

        # Execute the query with embeddings as parameters
        result = self.graph.query(query, params=embeddings_params)

        return result



class EmbeddingModel:
    @staticmethod
    def create_embedding(text: str) -> List[float]:
        result = client.embeddings.create(model=EMBEDDINGS_MODEL, input=text)
        return result.data[0].embedding
