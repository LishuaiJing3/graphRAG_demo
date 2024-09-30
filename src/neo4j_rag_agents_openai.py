
#%%
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.graphs import Neo4jGraph


url = os.getenv("NEO_DB_HOST_AURA")
username = os.getenv("NEO_DB_USER")
password = os.getenv("NEO_DB_PASSWORD_AURA")
database_name = "neo4j"

#%%

graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password
)

# %%
import requests

url = "https://gist.githubusercontent.com/tomasonjo/08dc8ba0e19d592c4c3cde40dd6abcc3/raw/da8882249af3e819a80debf3160ebbb3513ee962/microservices.json"
import_query = requests.get(url).json()['query']
graph.query(
    import_query
)
# %%
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings_model = "text-embedding-3-small"

#%%
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name='tasks',
    node_label="Task",
    text_node_properties=['name', 'description', 'status'],
    embedding_node_property='embedding',
)

# %%
response = vector_index.similarity_search(
    "How will RecommendationService be updated?"
)
print(response[0].page_content)
# name: BugFix
# description: Add a new feature to RecommendationService to provide ...
# status: In Progress
# %%
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vector_index.as_retriever()
)
vector_qa.run(
    "How will recommendation service be updated?"
)
# The RecommendationService is currently being updated to include a new feature 
# that will provide more personalized and accurate product recommendations to 
# users. This update involves leveraging user behavior and preference data to 
# enhance the recommendation algorithm. The status of this update is currently
# in progress.
# %%
vector_qa.run(
    "How many open tickets are there?"
)
# There are 4 open tickets.
# %%
graph.query(
    "MATCH (t:Task {status:'Open'}) RETURN count(*)"
)
# [{'count(*)': 5}]
'''
There are five open tasks in our toy graph. Vector similarity search is excellent for sifting through relevant information in unstructured text, but lacks the capability to analyze and aggregate structured information. Using Neo4j, this problem is easily solved by employing Cypher, a structured query language for graph databases.
'''
# %%

from langchain.chains import GraphCypherQAChain

graph.refresh_schema()

cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini'),
    qa_llm = ChatOpenAI(temperature=0), graph=graph, verbose=True,allow_dangerous_requests = True
)

cypher_chain.run(
    "How many open tickets there are?"
)
# this is quite error prone as well since LLM does not have context on the values from the schema. if it infers status = open to Open, then it returns zero results.  
# %%
cypher_chain.run(
    "Which team has the most open tasks?"
)

# %%
cypher_chain.run(
    "Which services depend on Database indirectly?"
)

#%% knowledge graph agents
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Tasks",
        func=vector_qa.run,
        description="""Useful when you need to answer questions about descriptions of tasks.
        Not useful for counting the number of tasks.
        Use full question as input.
        """,
    ),
    Tool(
        name="Graph",
        func=cypher_chain.run,
        description="""Useful when you need to answer questions about microservices,
        their dependencies or assigned people. Also useful for any sort of 
        aggregation like counting the number of tasks, etc.
        Use full question as input.
        """,
    ),
]

mrkl = initialize_agent(
    tools, 
    ChatOpenAI(temperature=0, model_name='gpt-4o-mini'),
    agent=AgentType.OPENAI_FUNCTIONS, verbose=True
)

# %%
response = mrkl.run("Which team is assigned to maintain PaymentService?")
print(response)
# %%
response = mrkl.run("Which tasks have optimization in their description?")
print(response)
# %%
