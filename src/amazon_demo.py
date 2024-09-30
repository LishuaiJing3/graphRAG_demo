#%%
import os
import json 
import pandas as pd


from dotenv import load_dotenv
load_dotenv()

#%%
# Loading a json dataset from a file
file_path = 'data/amazon_product_kg.json'

with open(file_path, 'r') as file:
    jsonData = json.load(file)
df =  pd.read_json(file_path)
df.head()
# %%
# DB credentials
url = os.getenv("NEO_DB_HOST_AURA")
username =os.getenv("NEO_DB_USER")
password = os.getenv("NEO_DB_PASSWORD_AURA")
from langchain.graphs import Neo4jGraph

graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password
)
# %%
def sanitize(text):
    text = str(text).replace("'","").replace('"','').replace('{','').replace('}', '')
    return text

# Loop through each JSON object and add them to the db
i = 1
for obj in jsonData:
    print(f"{i}. {obj['product_id']} -{obj['relationship']}-> {obj['entity_value']}")
    i+=1
    query = f'''
        MERGE (product:Product {{id: {obj['product_id']}}})
        ON CREATE SET product.name = "{sanitize(obj['product'])}", 
                       product.title = "{sanitize(obj['TITLE'])}", 
                       product.bullet_points = "{sanitize(obj['BULLET_POINTS'])}", 
                       product.size = {sanitize(obj['PRODUCT_LENGTH'])}

        MERGE (entity:{obj['entity_type']} {{value: "{sanitize(obj['entity_value'])}"}})

        MERGE (product)-[:{obj['relationship']}]->(entity)
        '''
    graph.query(query)
# %%
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings_model = "text-embedding-3-small"

#%%
# embed product text
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(model=embeddings_model),
    url=url,
    username=username,
    password=password,
    index_name='products',
    node_label="Product",
    text_node_properties=['name', 'title'],
    embedding_node_property='embedding',
)
# %%
# embed the text from selected nodes
def embed_entities(entity_type):
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model=embeddings_model),
        url=url,
        username=username,
        password=password,
        index_name=entity_type,
        node_label=entity_type,
        text_node_properties=['value'],
        embedding_node_property='embedding',
    )
    
entities_list = df['entity_type'].unique()

for t in entities_list:
    embed_entities(t)
# %%
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True,allow_dangerous_requests=True
)
# %%
# this will be quite error prone. We can see that the cyher query generated from LLM does not make sense 
chain.run("""
Help me find GPS tracking device
""")
# %%
# We need to We will instead use LLMs to decide what to search for, and then generate the corresponding Cypher queries using templates.
# For this purpose, we will instruct our model to find relevant entities in the user prompt that can be used to query our database.
entity_types = {
    "product": "Item detailed type, for example 'high waist pants', 'outdoor plant pot', 'chef kitchen knife'",
    "category": "Item category, for example 'home decoration', 'women clothing', 'office supply'",
    "characteristic": "if present, item characteristics, for example 'waterproof', 'adhesive', 'easy to use'",
    "measurement": "if present, dimensions of the item", 
    "brand": "if present, brand of the item",
    "color": "if present, color of the item",
    "age_group": "target age group for the product, one of 'babies', 'children', 'teenagers', 'adults'. If suitable for multiple age groups, pick the oldest (latter in the list)."
}

relation_types = {
    "hasCategory": "item is of this category",
    "hasCharacteristic": "item has this characteristic",
    "hasMeasurement": "item is of this measurement",
    "hasBrand": "item is of this brand",
    "hasColor": "item is of this color", 
    "isFor": "item is for this age_group"
 }

entity_relationship_match = {
    "category": "hasCategory",
    "characteristic": "hasCharacteristic",
    "measurement": "hasMeasurement", 
    "brand": "hasBrand",
    "color": "hasColor",
    "age_group": "isFor"
}
#%%

system_prompt = f'''
    You are a helpful agent designed to fetch information from a graph database. 
    
    The graph database links products to the following entity types:
    {json.dumps(entity_types)}
    
    Each link has one of the following relationships:
    {json.dumps(relation_types)}

    Depending on the user prompt, determine if it possible to answer with the graph database.
        
    The graph database can match products with multiple relationships to several entities.
    
    Example user input:
    "Which blue clothing items are suitable for adults?"
    
    There are three relationships to analyse:
    1. The mention of the blue color means we will search for a color similar to "blue"
    2. The mention of the clothing items means we will search for a category similar to "clothing"
    3. The mention of adults means we will search for an age_group similar to "adults"
    
    
    Return a json object following the following rules:
    For each relationship to analyse, add a key value pair with the key being an exact match for one of the entity types provided, and the value being the value relevant to the user query.
    
    For the example provided, the expected output would be:
    {{
        "color": "blue",
        "category": "clothing",
        "age_group": "adults"
    }}
    
    If there are no relevant entities in the user prompt, return an empty json object.
'''

print(system_prompt)
# %%

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the entities to look for
def define_query(prompt, model="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format= {
            "type": "json_object"
        },
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
    return completion.choices[0].message.content

example_queries = [
    "Which pink items are suitable for children?",
    "Help me find gardening gear that is waterproof",
    "I'm looking for a bench with dimensions 100x50 for my living room"
]

for q in example_queries:
    print(f"Q: '{q}'\n{define_query(q)}\n")
# %%
#Generating queries
#Now that we know what to look for, we can generate the corresponding Cypher queries to query our database.
#However, the entities extracted might not be an exact match with the data we have, so we will use the GDS cosine similarity function to return products that have relationships with entities similar to what the user is asking.
def create_embedding(text):
    result = client.embeddings.create(model=embeddings_model, input=text)
    return result.data[0].embedding


# The threshold defines how closely related words should be. Adjust the threshold to return more or less results
def create_query(text, threshold=0.81):
    query_data = json.loads(text)
    # Creating embeddings
    embeddings_data = []
    for key, val in query_data.items():
        if key != 'product':
            embeddings_data.append(f"${key}Embedding AS {key}Embedding")
    query = "WITH " + ",\n".join(e for e in embeddings_data)
    # Matching products to each entity
    query += "\nMATCH (p:Product)\nMATCH "
    match_data = []
    for key, val in query_data.items():
        if key != 'product':
            relationship = entity_relationship_match[key]
            match_data.append(f"(p)-[:{relationship}]->({key}Var:{key})")
    query += ",\n".join(e for e in match_data)
    similarity_data = []
    for key, val in query_data.items():
        if key != 'product':
            similarity_data.append(f"gds.similarity.cosine({key}Var.embedding, ${key}Embedding) > {threshold}")
    query += "\nWHERE "
    query += " AND ".join(e for e in similarity_data)
    query += "\nRETURN p"
    return query

def query_graph(response):
    embeddingsParams = {}
    query = create_query(response)
    query_data = json.loads(response)
    for key, val in query_data.items():
        embeddingsParams[f"{key}Embedding"] = create_embedding(val)
    result = graph.query(query, params=embeddingsParams)
    return result

example_response = '''{
    "category": "clothes",
    "color": "blue",
    "age_group": "adults"
}'''

#%%
# this part does not work with the gds. 
result = query_graph(example_response)
# %%
# find similar items from graph database: 
# Adjust the relationships_threshold to return products that have more or less relationships in common
def query_similar_items(product_id, relationships_threshold = 3):
    
    similar_items = []
        
    # Fetching items in the same category with at least 1 other entity in common
    query_category = '''
            MATCH (p:Product {id: $product_id})-[:hasCategory]->(c:category)
            MATCH (p)-->(entity)
            WHERE NOT entity:category
            MATCH (n:Product)-[:hasCategory]->(c)
            MATCH (n)-->(commonEntity)
            WHERE commonEntity = entity AND p.id <> n.id
            RETURN DISTINCT n;
        '''
    

    result_category = graph.query(query_category, params={"product_id": int(product_id)})
    #print(f"{len(result_category)} similar items of the same category were found.")
          
    # Fetching items with at least n (= relationships_threshold) entities in common
    query_common_entities = '''
        MATCH (p:Product {id: $product_id})-->(entity),
            (n:Product)-->(entity)
            WHERE p.id <> n.id
            WITH n, COUNT(DISTINCT entity) AS commonEntities
            WHERE commonEntities >= $threshold
            RETURN n;
        '''
    result_common_entities = graph.query(query_common_entities, params={"product_id": int(product_id), "threshold": relationships_threshold})
    #print(f"{len(result_common_entities)} items with at least {relationships_threshold} things in common were found.")

    for i in result_category:
        similar_items.append({
            "id": i['n']['id'],
            "name": i['n']['name']
        })
            
    for i in result_common_entities:
        result_id = i['n']['id']
        if not any(item['id'] == result_id for item in similar_items):
            similar_items.append({
                "id": result_id,
                "name": i['n']['name']
            })
    return similar_items

product_ids = ['1519827', '2763742']

for product_id in product_ids:
    print(f"Similar items for product #{product_id}:\n")
    result = query_similar_items(product_id)
    print("\n")
    for r in result:
        print(f"{r['name']} ({r['id']})")
    print("\n\n")
# %%
'''
Final result
Now that we have all the pieces working, we will stitch everything together.

We can also add a fallback option to do a product name/title similarity search if we can't find relevant entities in the user prompt.

We will explore 2 options, one with a Langchain agent for a conversational experience, and one that is more deterministic based on code only.

Depending on your use case, you might choose one or the other option and tailor it to your needs.
'''
def query_db(params):
    matches = []
    # Querying the db
    result = query_graph(params)
    for r in result:
        product_id = r['p']['id']
        matches.append({
            "id": product_id,
            "name":r['p']['name']
        })
    return matches 


def similarity_search(prompt, threshold=0.8):
    matches = []
    embedding = create_embedding(prompt)
    query = '''
            WITH $embedding AS inputEmbedding
            MATCH (p:Product)
            WHERE gds.similarity.cosine(inputEmbedding, p.embedding) > $threshold
            RETURN p
            '''
    result = graph.query(query, params={'embedding': embedding, 'threshold': threshold})
    for r in result:
        product_id = r['p']['id']
        matches.append({
            "id": product_id,
            "name":r['p']['name']
        })
    return matches


prompt_similarity = "I'm looking for nice curtains"
print(similarity_search(prompt_similarity))
# %%
