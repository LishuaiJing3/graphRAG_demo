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
        ## needs to be fixes, need to pass exact node and relationship projection
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

        self.gds.graph.project(graph_name, node_spec, rel_spec)
        print(f"Projected graph '{graph_name}' created successfully!")

    def create_organization_projection(self) -> str:
        graph_name = "organization_graph_projection"
        
        if self.gds.graph.exists(graph_name)["exists"]:
            self.gds.graph.drop(graph_name)

        self.gds.graph.project(
            graph_name,
            {'Organization': {}},
            {
                'HAS_INVESTOR': {'type': 'HAS_INVESTOR', 'orientation': 'NATURAL'},
                'HAS_SUBSIDIARY': {'type': 'HAS_SUBSIDIARY', 'orientation': 'NATURAL'},
                'HAS_SUPPLIER': {'type': 'HAS_SUPPLIER', 'orientation': 'NATURAL'}
            }
        )
        return graph_name

    def run_pagerank(self, graph_name: str):
        graph = self.gds.graph.get(graph_name)
        result = self.gds.pageRank.stream(graph)
        return result

    def analyze_pagerank_results(self, result, top_n: int = 10) -> None:
        sorted_result = result.sort_values(by="score", ascending=False)
        print(f"Top {top_n} influential organizations based on PageRank:")
        print(sorted_result.head(top_n))

class NodeOperations:
    def __init__(self, gds: GraphDataScience) -> None:
        self.gds = gds

    def delete_node_by_id(self, node_id: int) -> None:
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        DETACH DELETE n
        """
        self.gds.run_cypher(query, params={"node_id": node_id})

    def delete_node_by_name(self, node_name: str) -> None:
        query = """
        MATCH (n {name: $node_name})
        DETACH DELETE n
        """
        self.gds.run_cypher(query, params={"node_name": node_name})
