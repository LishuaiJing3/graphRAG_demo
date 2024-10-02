from neo4j_service import GDSConnector, GraphOperations, NodeOperations
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configuration: Load Neo4j credentials from environment variables
URI: str = os.getenv('NEO4J_URI', 'neo4j+s://demo.neo4jlabs.com')
USER: str = os.getenv('NEO4J_USERNAME', 'companies')
PASSWORD: str = os.getenv('NEO4J_PASSWORD', 'companies')
DATABASE: str = os.getenv('NEO4J_DATABASE', 'companies')

def main() -> None:
    # Initialize the GDS Connector
    connector = GDSConnector(uri=URI, user=USER, password=PASSWORD, database=DATABASE)
    gds = connector.get_gds()

    # Initialize operations classes
    graph_ops = GraphOperations(gds)
    node_ops = NodeOperations(gds)

    # Example 1: Delete a specific graph by name
    graph_ops.delete_graph('entireGraph')

    # Example 2: Estimate memory for a specific graph
    graph_ops.estimate_graph_memory('entireGraph')

    # Example 3: Project all nodes and relationships
    graph_ops.project_all_nodes_and_relationships()

    # Example 4: Create an organization projection and run PageRank
    org_graph_name = graph_ops.create_organization_projection()
    pagerank_result = graph_ops.run_pagerank(org_graph_name)
    graph_ops.analyze_pagerank_results(pagerank_result)

    # Example 5: Delete a node by ID
    node_ops.delete_node_by_id(123)  # Replace 123 with the actual node ID to be deleted

    # Example 6: Delete a node by name
    node_ops.delete_node_by_name("Greeting")  # Replace "Greeting" with the actual node name to be deleted

    # Close the GDS connection
    connector.close()

if __name__ == "__main__":
    main()
