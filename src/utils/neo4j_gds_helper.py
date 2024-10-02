from graphdatascience import GraphDataScience
from typing import List, Tuple

class GDSConnector:
    def __init__(self, uri: str, user: str, password: str, database: str) -> None:
        self.gds = GraphDataScience(uri, auth=(user, password), database=database)
        print(f"GDS Version: {self.gds.version()}")
        assert self.gds.version()

    def close(self) -> None:
        self.gds.close()

    def get_gds(self) -> GraphDataScience:
        return self.gds

class GraphOperations:
    def __init__(self, gds: GraphDataScience) -> None:
        self.gds = gds

    def delete_graph(self, graph_name: str) -> None:
        if self.gds.graph.exists(graph_name).exists:
            self.gds.graph.drop(graph_name)
            print(f"Graph '{graph_name}' deleted successfully.")
        else:
            print(f"Graph '{graph_name}' does not exist.")

    def estimate_graph_memory(self, graph_name: str) -> None:
        estimation = self.gds.graph.estimateFull(graph_name, nodeProjection='*', relationshipProjection='*')
        print(f"Estimated Memory: {estimation['requiredMemory']}")

    def get_all_labels_and_relationship_types(self) -> Tuple[List[str], List[str]]:
        labels_query = "CALL db.labels() YIELD label"
        labels_result = self.gds.run_cypher(labels_query)
        node_labels = labels_result['label'].tolist()

        rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType"
        rel_types_result = self.gds.run_cypher(rel_types_query)
        rel_types = rel_types_result['relationshipType'].tolist()

        return node_labels, rel_types

    def project_all_nodes_and_relationships(self) -> None:
        graph_name = "full_graph_projection"

        if self.gds.graph.exists(graph_name)["exists"]:
            self.gds.graph.drop(graph_name)

        node_labels, rel_types = self.get_all_labels_and_relationship_types()

        node_spec = {label: {} for label in node_labels}
        rel_spec = {rel: {"type": rel} for rel in rel_types}

       
