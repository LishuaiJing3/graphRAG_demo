#%%
from graphdatascience import GraphDataScience

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

URI = os.getenv('NEO4J_URI', 'neo4j+s://demo.neo4jlabs.com')
USER = os.getenv('NEO4J_USERNAME','companies')
PASSWORD = os.getenv('NEO4J_PASSWORD','companies')
DATABASE = os.getenv('NEO4J_DATABASE','companies')
# Use Neo4j URI and credentials according to your setup
# NEO4J_URI could look similar to "bolt://my-server.neo4j.io:7687"
gds = GraphDataScience(URI, auth=(USER, PASSWORD), database=DATABASE)

# Check the installed GDS version on the server
print(gds.version())
assert gds.version()

#%%
data = gds.run_cypher(
  """
    MATCH (i:IndustryCategory) RETURN i.name as industry
  """
)

# list the avilable graphs
gds.graph.list()
#%%
# Function to delete a specific graph by name
def delete_graph(graph_name):
    if gds.graph.exists(graph_name).exists:
        result = gds.graph.drop(graph_name)
        print(f"Graph '{graph_name}' deleted successfully.")
    else:
        print(f"Graph '{graph_name}' does not exist.")

# Example: Delete the graph named 'myGraph'
delete_graph('entireGraph')

#%%
def delete_node_and_relationships(node_id):
    query = """
    MATCH (n)
    WHERE id(n) = $node_id
    DETACH DELETE n
    """
    # Run the query using the gds.run_cypher method
    gds.run_cypher(query, params={"node_id": node_id})

# Example usage
node_id = 123  # Replace with the actual internal Neo4j node ID you want to delete
delete_node_and_relationships(node_id)

#%%
def delete_node_by_name(node_name):
    query = """
    MATCH (n {name: $node_name})
    DETACH DELETE n
    """
    gds.run_cypher(query, params={"node_name": node_name})


# Example usage
node_name = "Greeting"  # Replace with the actual node name you want to delete
delete_node_by_name(node_name)

#%%
def estimate_graph_memory():
    estimation = gds.graph.estimateFull('entireGraph', nodeProjection='*', relationshipProjection='*')
    print(f"Estimated Memory: {estimation['requiredMemory']}")

estimate_graph_memory()

#%%
def get_all_labels_and_relationship_types():
    # Query to get all node labels
    labels_query = "CALL db.labels() YIELD label"
    labels_result = gds.run_cypher(labels_query)
    node_labels = labels_result['label'].tolist()

    # Query to get all relationship types
    rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType"
    rel_types_result = gds.run_cypher(rel_types_query)
    rel_types = rel_types_result['relationshipType'].tolist()

    return node_labels, rel_types

# Define a function to project all nodes and relationships for analysis
def project_all_nodes_and_relationships():
    graph_name = "full_graph_projection"

    # Drop the graph if it already exists to avoid errors
    if gds.graph.exists(graph_name)["exists"]:
        gds.graph.drop(graph_name)

    # Get all node labels and relationship types
    node_labels, rel_types = get_all_labels_and_relationship_types()

    # Prepare the node and relationship specifications for projection
    node_spec = {label: {} for label in node_labels}
    rel_spec = {rel: {"type": rel} for rel in rel_types}

    # Create the projection of the entire graph
    gds.graph.project(graph_name, node_spec, rel_spec)

    print(f"Projected graph '{graph_name}' created successfully!")

# Example usage
project_all_nodes_and_relationships()


#%%
# Step 1: Create the graph projection
def create_organization_projection():
    graph_name = "organization_graph_projection"
    
    # Drop existing projection if exists
    if gds.graph.exists(graph_name)["exists"]:
        gds.graph.drop(graph_name)

    # Projecting only 'Organization' nodes and relevant relationships
    gds.graph.project(
        graph_name,
        {
            'Organization': {}  # Project all nodes with the 'Organization' label
        },
        {
            'HAS_INVESTOR': {'type': 'HAS_INVESTOR', 'orientation': 'NATURAL'},
            'HAS_SUBSIDIARY': {'type': 'HAS_SUBSIDIARY', 'orientation': 'NATURAL'},
            'HAS_SUPPLIER': {'type': 'HAS_SUPPLIER', 'orientation': 'NATURAL'}
        }
    )
    return graph_name

# Step 2: Run PageRank algorithm
def run_pagerank(graph_name):
    # Retrieve the Graph object using gds.graph.get
    graph = gds.graph.get(graph_name)
    
    # Run the PageRank algorithm
    result = gds.pageRank.stream(graph)
    return result

# Step 3: Extract and display top influential organizations
def analyze_pagerank_results(result, top_n=10):
    # Sorting by the PageRank score to get the most influential nodes
    sorted_result = result.sort_values(by="score", ascending=False)
    print(f"Top {top_n} influential organizations based on PageRank:")
    print(sorted_result.head(top_n))

# Main Execution
graph_name = create_organization_projection()
pagerank_result = run_pagerank(graph_name)
analyze_pagerank_results(pagerank_result)

# Close any open connections in the underlying Neo4j driver's connection pool
gds.close()

# %%
