#%%
# Load configuration
import os
from src.utils.graph_helpers import connect_to_graph, execute_query
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from neo4j.exceptions import ServiceUnavailable, DatabaseUnavailable
from neo4j import GraphDatabase
from google.cloud import bigquery

# Load configuration
load_dotenv()
# %%
# Create Neo4j driver to directly execute queries
driver = GraphDatabase.driver(
    os.getenv("NEO_DB_HOST_DEV"),
    auth=(os.getenv("NEO_DB_USER_DEV"), os.getenv("NEO_DB_PASSWORD_DEV")), database=os.getenv("NEO_DB_DATABASE_DEV")
)
#%%
# Set up BigQuery client
bigquery_client = bigquery.Client()

# Query BigQuery to get product data
query = """
    SELECT
        id,
        product_id,
        title,
        description,
        image_url,
        image_count,
        updated_at,
        price_amount,
        price_currency,
        brand_name,
        domain,
        identifier_id,
        identifier_type,
        country_code,
        embedding
    FROM
        `gowish-develop.poc_product_categorization.sampled_graph_poc_dk_5k`
"""

query_job = bigquery_client.query(query)
results = list(query_job.result())

#%%
# Iterate over BigQuery results and load data into Neo4j
with driver.session() as session:
    for row in results:
        # Create Product node
        query_product = """
        MERGE (p:Product {productId: $productId})
        SET p.title = $title,
            p.description = $description,
            p.imageUrl = $imageUrl,
            p.imageCount = $imageCount,
            p.updatedAt = $updatedAt,
            p.priceAmount = $priceAmount,
            p.priceCurrency = $priceCurrency,
            p.embedding = $embedding
        """
        session.run(query_product, parameters={
            "productId": row.product_id,
            "title": row.title if row.title else None,
            "description": row.description if row.description else None,
            "imageUrl": row.image_url if row.image_url else None,
            "imageCount": int(row.image_count) if row.image_count else None,
            "updatedAt": row.updated_at.isoformat() if row.updated_at else None,
            "priceAmount": float(row.price_amount) if row.price_amount else None,
            "priceCurrency": row.price_currency if row.price_currency else None,
            "embedding": row.embedding if row.embedding else None
        })

        # Create/merge Brand node and relationship
        if row.brand_name:
            query_brand = """
            MERGE (b:Brand {name: $brand_name})
            MERGE (p:Product {productId: $productId})
            MERGE (p)-[:BELONGS_TO_BRAND]->(b)
            """
            session.run(query_brand, parameters={
                "brand_name": row.brand_name,
                "productId": row.product_id
            })

        # Create/merge Identifier node and relationship
        if row.identifier_id:
            query_identifier = """
            MERGE (i:Identifier {identifierId: $identifierId})
            SET i.type = $identifierType
            MERGE (p:Product {productId: $productId})
            MERGE (p)-[:HAS_IDENTIFIER]->(i)
            """
            session.run(query_identifier, parameters={
                "identifierId": row.identifier_id,
                "identifierType": row.identifier_type if row.identifier_type else None,
                "productId": row.product_id
            })

        # Create/merge Webshop node and relationship
        if row.domain:
            query_webshop = """
            MERGE (w:Webshop {webshopId: $webshopId, name: $webshopName})
            MERGE (p:Product {productId: $productId})
            MERGE (p)-[:SOLD_BY]->(w)
            """
            session.run(query_webshop, parameters={
                "webshopId": row.domain,
                "webshopName": row.domain,
                "productId": row.product_id
            })

        # Create/merge Country node and relationship
        if row.country_code:
            query_country = """
            MERGE (c:Country {countryCode: $countryCode})
            MERGE (p:Product {productId: $productId})
            MERGE (p)-[:AVAILABLE_IN]->(c)
            """
            session.run(query_country, parameters={
                "countryCode": row.country_code,
                "productId": row.product_id
            })

print("Data successfully loaded into Neo4j.")
# %%
