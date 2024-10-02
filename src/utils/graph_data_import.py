def import_graph_data_sample(graph):
    """import some sample data to build knowledge graph

    Args:
        graph (): neo4j grph object

    Returns:
        _type_: _description_
    """
    import requests
    from src.utils.graph_helpers import execute_query
    url = "https://gist.githubusercontent.com/tomasonjo/08dc8ba0e19d592c4c3cde40dd6abcc3/raw/da8882249af3e819a80debf3160ebbb3513ee962/microservices.json"
    try:
        import_query = requests.get(url).json()['query']
        return execute_query(graph, import_query)
    except requests.RequestException as e:
        print(f"Error fetching query: {e}")
        return None