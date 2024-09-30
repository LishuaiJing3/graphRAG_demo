from langchain.graphs import Neo4jGraph

def connect_to_graph(url, username, password, database_name):
    return Neo4jGraph(
        url=url,
        username=username,
        password=password,
        database=database_name
    )

def execute_query(graph, query):
    return graph.query(query)
