#%%
from src.config import load_config
from src.utils.graph_helpers import connect_to_graph,  execute_query
from src.utils.vector_helpers import create_vector_index, load_vector_index, similarity_search
from src.utils.qa_helpers import create_retrieval_qa, perform_qa
from src.utils.agent_helpers import create_cypher_chain, initialize_tools, initialize_mrkl_agent
from src.graph_data_import import import_graph_data_sample

# Load configuration
config = load_config()
url = config["url"]
username = config["username"]
password = config["password"]
database_name = config["database_name"]

update_graph = False
rebuild_graph = False
# Connect to graph
graph = connect_to_graph(url, username, password, database_name)

if update_graph:
    # Import microservices or other setup steps here
    import_graph_data_sample(graph)

# Load or create vector index
if rebuild_graph:
    vector_index = create_vector_index(url, username, password)
else:
    vector_index = load_vector_index(url, username, password)

#%%
# Perform similarity search
query_result = similarity_search(vector_index, "How will RecommendationService be updated?")
print(query_result[0].page_content)

# Create QA instance and perform question answering
vector_qa = create_retrieval_qa(vector_index)
qa_result = perform_qa(vector_qa, "How will recommendation service be updated?")
print(qa_result)

# Execute a Cypher query using GraphCypherQAChain
cypher_chain = create_cypher_chain(graph)
cypher_result = cypher_chain.run("How many open tickets there are?")
print(cypher_result)

# Initialize tools and agent
tools = initialize_tools(vector_qa, cypher_chain)
mrkl_agent = initialize_mrkl_agent(tools)

# Run agent queries
mrkl_response = mrkl_agent.run("Which team is assigned to maintain PaymentService?")
print(mrkl_response)
