from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings


def load_vector_index(url, username, password, index_name='tasks'):
    """
    Load an existing vector index instead of creating a new one.
    """
    embedding_model = OpenAIEmbeddings()  # Create an embedding model instance
    return Neo4jVector.from_existing_index(
        embedding=embedding_model,  # Pass the actual embedding model object here
        url=url,
        username=username,
        password=password,
        index_name=index_name,
        node_label="Task",
        embedding_node_property='embedding',
    )

def create_vector_index(url, username, password, index_name='tasks'):
    return Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        url=url,
        username=username,
        password=password,
        index_name=index_name,
        node_label="Task",
        text_node_properties=['name', 'description', 'status'],
        embedding_node_property='embedding',
    )

def similarity_search(vector_index, query):
    return vector_index.similarity_search(query)
