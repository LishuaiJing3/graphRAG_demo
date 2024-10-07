#%%
# Merged Neo4j and OpenAI RAG code with main function
import os
import json
import re
import pandas as pd
from typing import List, Dict, Any, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from openai import OpenAI
from langchain.schema import AgentAction, AgentFinish

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
class Product:
    id: str
    title: str
    description: str
    image_url: str
    image_count: int
    updated_at: str
    price_amount: float
    price_currency: str
    similarity: float = 0.0

@dataclass
class KnowledgeGraphConfig:
    entity_types: Dict[str, str]
    relation_types: Dict[str, str]
    entity_relationship_match: Dict[str, str]

class DataProcessor:
    @staticmethod
    def sanitize(text: str) -> str:
        return str(text).replace("'", "").replace('"', '').replace('{', '').replace('}', '')

class GraphConstructor:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=DB_URL,
            username=DB_USERNAME,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )

    def add_products_to_db(self, product_data: pd.DataFrame):
        for _, row in product_data.iterrows():
            # Create Product node
            query_product = """
            MERGE (p:Product {productId: $productId})
            SET p.title = $title,
                p.description = $description,
                p.imageUrl = $imageUrl,
                p.imageCount = $imageCount,
                p.updatedAt = $updatedAt,
                p.priceAmount = $priceAmount,
                p.priceCurrency = $priceCurrency
            """
            self.graph.query(query_product, parameters={
                "productId": row.product_id,
                "title": row.title if row.title else None,
                "description": row.description if row.description else None,
                "imageUrl": row.image_url if row.image_url else None,
                "imageCount": int(row.image_count) if row.image_count else None,
                "updatedAt": row.updated_at.isoformat() if row.updated_at else None,
                "priceAmount": float(row.price_amount) if row.price_amount else None,
                "priceCurrency": row.price_currency if row.price_currency else None
            })

            # Create/merge Brand node and relationship
            if row.brand_name:
                query_brand = """
                MERGE (b:Brand {name: $brand_name})
                MERGE (p:Product {productId: $productId})
                MERGE (p)-[:BELONGS_TO_BRAND]->(b)
                """
                self.graph.query(query_brand, parameters={
                    "brand_name": row.brand_name,
                    "productId": row.product_id
                })

            # Create/merge Identifier node and relationship
            if row.identifier_id:
                query_identifier = """
                MERGE (i:Identifier {identifierId: $identifierId})
                SET i.type = $identifierType
                MERGE (p:Product {productId: $productId})
                MERGE (p)-[:HAS_IDENTIFIER]->(i)
                """
                self.graph.query(query_identifier, parameters={
                    "identifierId": row.identifier_id,
                    "identifierType": row.identifier_type if row.identifier_type else None,
                    "productId": row.product_id
                })

            # Create/merge Webshop node and relationship
            if row.domain:
                query_webshop = """
                MERGE (w:Webshop {webshopId: $webshopId, name: $webshopName})
                MERGE (p:Product {productId: $productId})
                MERGE (p)-[:SOLD_BY]->(w)
                """
                self.graph.query(query_webshop, parameters={
                    "webshopId": row.domain,
                    "webshopName": row.domain,
                    "productId": row.product_id
                })

            # Create/merge Country node and relationship
            if row.country_code:
                query_country = """
                MERGE (c:Country {countryCode: $countryCode})
                MERGE (p:Product {productId: $productId})
                MERGE (p)-[:AVAILABLE_IN]->(c)
                """
                self.graph.query(query_country, parameters={
                    "countryCode": row.country_code,
                    "productId": row.product_id
                })

class VectorIndexManager:
    def __init__(self):
        self.url = DB_URL
        self.username = DB_USERNAME
        self.password = DB_PASSWORD
        self.model = EMBEDDINGS_MODEL

    def embed_product_text(self, index_name: str, node_label: str, text_node_properties: List[str]):
        return Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(model=self.model, embedding_dim=EMBEDDINGS_DIMENSION),
            url=self.url,
            username=self.username,
            password=self.password,
            index_name=index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property='embedding',
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
        '''

    def define_query(self, prompt: str, model: str = "gpt-4") -> Dict:
        system_prompt = self.get_system_prompt()
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            response_content = completion.choices[0].message.content.strip()
            return json.loads(response_content)
        except Exception as e:
            print(f"An error occurred while generating the query: {e}")
            return {}

class GraphOperations:
    def __init__(self, graph_connection):
        self.graph = graph_connection

    def similarity_search(self, embedding: List[float], threshold: float = 0.5) -> List[Product]:
        query = '''
                WITH $embedding AS inputEmbedding
                MATCH (p:Product)
                WITH p, vector.similarity.cosine(inputEmbedding, p.embedding) AS similarity
                WHERE similarity > $threshold
                RETURN p.productId AS productId, p.title AS title, p.description AS description, p.imageUrl AS imageUrl, p.imageCount AS imageCount, p.updatedAt AS updatedAt, p.priceAmount AS priceAmount, p.priceCurrency AS priceCurrency, similarity
                '''
        result = self.graph.query(query, params={'embedding': embedding, 'threshold': threshold})
        return [Product(
            id=r['productId'],
            title=r['title'],
            description=r['description'],
            image_url=r['imageUrl'],
            image_count=r['imageCount'],
            updated_at=r['updatedAt'],
            price_amount=r['priceAmount'],
            price_currency=r['priceCurrency'],
            similarity=r['similarity']
        ) for r in result]

    def query_graph(self, response: Dict, query_generator: QueryGenerator, similarity_threshold: float):
        query = query_generator.create_cypher_query(response, similarity_threshold)
        embeddings_params = {f"{key}Embedding": EmbeddingModel.create_embedding(value) for key, value in response.items()}
        result = self.graph.query(query, params=embeddings_params)
        return result

class EmbeddingModel:
    @staticmethod
    def create_embedding(text: str) -> List[float]:
        result = client.embeddings.create(model=EMBEDDINGS_MODEL, input=text)
        return result.data[0].embedding

# Main Functionality
graph = Neo4jGraph(
    url=DB_URL,
    username=DB_USERNAME,
    password=DB_PASSWORD,
    database=DB_DATABASE
)
graph_operations = GraphOperations(graph_connection=graph)

entity_types = {
    "product": "The type of item, e.g., 'jewelry', 'watch', 'ring'.",
    "brand": "The brand of the product, e.g., 'La Joaillière'.",
    "identifier": "Unique identifier related to the product, such as GTIN or SKU.",
    "webshop": "The webshop where the product is sold, e.g., 'Zalando'.",
    "country": "Country where the product is available, represented by the country code, e.g., 'US', 'DE'."
}

relation_types = {
    "BELONGS_TO_BRAND": "Relationship between Product and Brand.",
    "HAS_IDENTIFIER": "Relationship between Product and Identifier.",
    "SOLD_BY": "Relationship between Product and Webshop.",
    "AVAILABLE_IN": "Relationship between Product and Country."
}

entity_relationship_match = {
    "product": "Product",
    "brand": "BELONGS_TO_BRAND",
    "identifier": "HAS_IDENTIFIER",
    "webshop": "SOLD_BY",
    "country": "AVAILABLE_IN"
}

#%%
config = KnowledgeGraphConfig(entity_types, relation_types, entity_relationship_match)

#%%

# Query using LLM to generate entities and Cypher query
query_generator = QueryGenerator(config)
prompt = "Which blue clothing items are suitable for adults?"
entities = query_generator.define_query(prompt)

#%%


def query_db(params):
    matches = []
    result = graph_operations.query_graph(params, QueryGenerator(KnowledgeGraphConfig(entity_types, relation_types, {})), 0.5)
    for r in result:
        product_id = r["p"]["productId"]
        matches.append({"id": product_id, "name": r["p"]["title"]})
    return matches

def similarity_search(prompt: str, threshold: float = 0.5) -> list:
    matches = []
    embedding = EmbeddingModel.create_embedding(prompt)
    result = graph_operations.similarity_search(embedding, threshold)
    for r in result:
        matches.append({
            "id": r.id,
            "title": r.title,
            "description": r.description,
            "similarity": r.similarity
        })
    return matches

tools = [
    Tool(
        name="Query",
        func=query_db,
        description="Use this tool to find entities in the user prompt that can be used to generate queries",
    ),
    Tool(
        name="Similarity Search",
        func=similarity_search,
        description="Use this tool to perform a similarity search with the products in the database",
    ),
]

tool_names = [f"{tool.name}: {tool.description}" for tool in tools]

prompt_template = """Your goal is to find a product in the database that best matches the user prompt.
You have access to these tools:

{tools}

Use the following format:

Question: the input prompt from the user
Thought: you should always think about what to do
Action: the action to take (refer to the rules below)
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules to follow:

1. Start by using the Query tool with the prompt as parameter. If you found results, stop here.
2. If the result is an empty array, use the similarity search tool with the full initial user prompt. If you found results, stop here.
3. If you cannot still cannot find the answer with this, probe the user to provide more context on the type of product they are looking for.

Keep in mind that we can use entities of the following types to search for products:

{entity_types}.

3. Repeat Step 1 and 2. If you found results, stop here.

4. If you cannot find the final answer, say that you cannot help with the question.

Never return results if you did not find any results in the array returned by the query tool or the similarity search tool.

If you didn't find any result, reply: "Sorry, I didn't find any suitable products."

If you found results from the database, this is your final answer, reply to the user by announcing the number of results and returning results in this format (each new result should be on a new line):

name_of_the_product (id_of_the_product)"

Only use exact names and ids of the products returned as results when providing your final answer.


User prompt:
{input}

{agent_scratchpad}
"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([
            f"{tool.name}: {tool.description}" for tool in tools
        ])
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        kwargs["entity_types"] = json.dumps(entity_types)
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=prompt_template,
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

output_parser = CustomOutputParser()

llm = ChatOpenAI(temperature=0, model="gpt-4")

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\Observation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

def agent_interaction(user_prompt):
    agent_executor.run(user_prompt)

prompt1 = "I'm searching for pink shirts"
agent_interaction(prompt1)
# %%
