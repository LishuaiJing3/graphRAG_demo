#%%
import os
from src.utils.amazon_kg_demo_helper import DataProcessor, GraphConstructor, VectorIndexManager, GraphOperations, EmbeddingModel, QueryGenerator, KnowledgeGraphConfig
from langchain.graphs import Neo4jGraph

# Configuration
entity_types = {
    "product": "Item detailed type, for example 'high waist pants', 'outdoor plant pot', 'chef kitchen knife'",
    "category": "Item category, for example 'home decoration', 'women clothing', 'office supply'",
    "characteristic": "if present, item characteristics, for example 'waterproof', 'adhesive', 'easy to use'",
    "measurement": "if present, dimensions of the item", 
    "brand": "if present, brand of the item",
    "color": "if present, color of the item",
    "age_group": "target age group for the product, one of 'babies', 'children', 'teenagers', 'adults'. If suitable for multiple age groups, pick the oldest."
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

FILE_PATH = 'data/amazon_product_kg.json'

#%%
def main(reconstruct_kg: bool = False):
    # Load data
    df, json_data = DataProcessor.load_data(FILE_PATH)


    if reconstruct_kg:
        # Construct the graph
        print("Reconstructing the knowledge graph...")
        graph_constructor = GraphConstructor()
        graph_constructor.add_products_to_db(json_data)

        # Embed entities
        print("Embedding entities...")
        vector_index_manager = VectorIndexManager()
        vector_index_manager.embed_entities(df)
    else:
        print("Skipping knowledge graph reconstruction...")

    # Create Knowledge Graph Config
    config = KnowledgeGraphConfig(entity_types, relation_types, entity_relationship_match)

    # Query using LLM to generate entities and Cypher query
    query_generator = QueryGenerator(config)
    prompt = "Which blue clothing items are suitable for adults?"
    entities = query_generator.define_query(prompt)
    
    if entities:
        graph = Neo4jGraph(
            url=os.getenv("NEO_DB_HOST_AURA"),
            username=os.getenv("NEO_DB_USER"),
            password=os.getenv("NEO_DB_PASSWORD_AURA")
        )
        graph_operations = GraphOperations(graph_connection=graph)

        # Query the graph
        print("Querying the graph based on LLM output...")
        similarity_thrshold = 0.2
        result = graph_operations.query_graph(entities, query_generator, similarity_thrshold)

        # Display results
        print("Query Results:")
        for r in result:
            print(f"Product ID: {r['p']['id']}, Name: {r['p']['name']}")


if __name__ == "__main__":
    import argparse

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Knowledge Graph Reconstruction and Query Tool")
    parser.add_argument(
        "--reconstruct_kg",
        action="store_true",
        help="Flag to indicate whether to reconstruct the knowledge graph"
    )
    args = parser.parse_args()

    # Run main with the flag for reconstructing the KG
    main(reconstruct_kg=args.reconstruct_kg)
