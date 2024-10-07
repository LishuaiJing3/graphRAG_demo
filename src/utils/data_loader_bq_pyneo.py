from py2neo import Graph, Node, Relationship
import os

# Configure Neo4j client
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load data into Neo4j
graph.delete_all()  # Optional: Clears the database before loading new data

# Assuming data is already prepared in a suitable format for Neo4j
for row in results:
    # Create Product node
    product = Node("Product",
                   productId=row.product_id,
                   title=row.title if row.title else None,
                   description=row.description if row.description else None,
                   imageUrl=row.image_url if row.image_url else None,
                   imageCount=int(row.image_count) if row.image_count else None,
                   updatedAt=row.updated_at.isoformat() if row.updated_at else None,
                   priceAmount=float(row.price_amount) if row.price_amount else None,
                   priceCurrency=row.price_currency if row.price_currency else None,
                   embedding=row.embedding if row.embedding else None)
    graph.merge(product, "Product", "productId")

    # Create/merge Brand node and relationship
    if row.brand_name:
        brand = Node("Brand", name=row.brand_name)
        graph.merge(brand, "Brand", "name")
        belongs_to_brand = Relationship(product, "BELONGS_TO_BRAND", brand)
        graph.merge(belongs_to_brand)

    # Create/merge Identifier node and relationship
    if row.identifier_id:
        identifier = Node("Identifier", identifierId=row.identifier_id, type=row.identifier_type if row.identifier_type else None)
        graph.merge(identifier, "Identifier", "identifierId")
        has_identifier = Relationship(product, "HAS_IDENTIFIER", identifier)
        graph.merge(has_identifier)

    # Create/merge Webshop node and relationship
    if row.domain:
        webshop = Node("Webshop", webshopId=row.domain, name=row.domain)
        graph.merge(webshop, "Webshop", "webshopId")
        sold_by = Relationship(product, "SOLD_BY", webshop)
        graph.merge(sold_by)

    # Create/merge Country node and relationship
    if row.country_code:
        country = Node("Country", countryCode=row.country_code)
        graph.merge(country, "Country", "countryCode")
        available_in = Relationship(product, "AVAILABLE_IN", country)
        graph.merge(available_in)

print("Data successfully loaded into Neo4j.")