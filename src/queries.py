"""Every Cypher statement used in the project, in one place so the notebook can print and explain them."""

ALL_RELS = "VIEWED|ADDED_TO_CART|BOUGHT"

CONSTRAINTS = [
    "CREATE CONSTRAINT visitor_id IF NOT EXISTS FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE",
    "CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:Item) REQUIRE i.itemId IS UNIQUE",
]

DELETE_BATCH = "MATCH (n) WITH n LIMIT $size DETACH DELETE n RETURN count(*) AS deleted"

MERGE_NODES = "UNWIND $ids AS id MERGE (:{label} {{{key}: id}})"

CREATE_RELATIONSHIPS = {
    "VIEWED": """
        UNWIND $rows AS row
        MATCH (v:Visitor {visitorId: row.visitorId})
        MATCH (i:Item {itemId: row.itemId})
        CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)""",
    "ADDED_TO_CART": """
        UNWIND $rows AS row
        MATCH (v:Visitor {visitorId: row.visitorId})
        MATCH (i:Item {itemId: row.itemId})
        CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)""",
    "BOUGHT": """
        UNWIND $rows AS row
        MATCH (v:Visitor {visitorId: row.visitorId})
        MATCH (i:Item {itemId: row.itemId})
        CREATE (v)-[:BOUGHT {timestamp: row.timestamp, transactionId: row.transactionId}]->(i)""",
}

NODE_COUNTS = """
MATCH (v:Visitor) RETURN 'Visitor' AS label, count(v) AS total
UNION ALL
MATCH (i:Item) RETURN 'Item' AS label, count(i) AS total
"""

RELATIONSHIP_COUNTS = """
MATCH ()-[r:VIEWED]->() RETURN 'VIEWED' AS relationship, count(r) AS total
UNION ALL
MATCH ()-[r:ADDED_TO_CART]->() RETURN 'ADDED_TO_CART' AS relationship, count(r) AS total
UNION ALL
MATCH ()-[r:BOUGHT]->() RETURN 'BOUGHT' AS relationship, count(r) AS total
"""

INTERACTIONS = f"""
MATCH (v:Visitor)-[r:{ALL_RELS}]->(i:Item)
RETURN v.visitorId AS visitorId, i.itemId AS itemId, type(r) AS rel, count(r) AS n
"""

SIMILARITY_COUNT = "MATCH ()-[s:SIMILAR_TO]->() RETURN count(s) AS similar_to_relationships"
SIMILARITY_DELETE = "MATCH ()-[s:SIMILAR_TO]->() DELETE s"

COSINE_CROSS_CHECK = {
    "item": f"""
MATCH (ref:Item {{itemId: $ref}})<-[:{ALL_RELS}]-(v:Visitor)
WITH ref, collect(DISTINCT v) AS refUsers
WITH ref, refUsers, size(refUsers) AS nRef
UNWIND refUsers AS v
MATCH (v)-[:{ALL_RELS}]->(other:Item)
WHERE other <> ref
WITH nRef, other, count(DISTINCT v) AS common
MATCH (other)<-[:{ALL_RELS}]-(w:Visitor)
WITH nRef, other, common, count(DISTINCT w) AS nOther
RETURN other.itemId AS itemId, common, nOther,
       toFloat(common) / sqrt(toFloat(nRef) * nOther) AS cosine
ORDER BY cosine DESC, common DESC
LIMIT 5
""",
    "user": f"""
MATCH (ref:Visitor {{visitorId: $ref}})-[:{ALL_RELS}]->(i:Item)
WITH ref, collect(DISTINCT i) AS refItems
WITH ref, refItems, size(refItems) AS nRef
UNWIND refItems AS i
MATCH (i)<-[:{ALL_RELS}]-(other:Visitor)
WHERE other <> ref
WITH nRef, other, count(DISTINCT i) AS common
MATCH (other)-[:{ALL_RELS}]->(j:Item)
WITH nRef, other, common, count(DISTINCT j) AS nOther
RETURN other.visitorId AS visitorId, common, nOther,
       toFloat(common) / sqrt(toFloat(nRef) * nOther) AS cosine
ORDER BY cosine DESC, common DESC
LIMIT 5
""",
}


def write_similarity(label: str, key: str) -> str:
    """Store the Python-computed scores on the graph as (a)-[:SIMILAR_TO {score, rank}]->(b)."""
    return f"""
UNWIND $rows AS row
MATCH (a:{label} {{{key}: row.source}})
MATCH (b:{label} {{{key}: row.target}})
MERGE (a)-[s:SIMILAR_TO]->(b)
SET s.score = row.score, s.rank = row.rank, s.metric = 'cosine'
"""


def top_similar(label: str, key: str) -> str:
    """The five highest-scoring SIMILAR_TO neighbours of one reference node."""
    return f"""
MATCH (ref:{label} {{{key}: $ref}})-[s:SIMILAR_TO]->(other:{label})
RETURN other.{key} AS {key}, s.score AS similarity, s.rank AS rank
ORDER BY similarity DESC
LIMIT 5
"""


def similarity_subgraph(label: str, key: str) -> str:
    """Path form of the previous query, so Neo4j Browser draws it as a graph."""
    return f"""
MATCH p = (ref:{label} {{{key}: $ref}})-[s:SIMILAR_TO]->(other:{label})
RETURN p
ORDER BY s.score DESC
LIMIT 5
"""


def similarity_edges(label: str, key: str) -> str:
    """Tabular form of the SIMILAR_TO sub-graph, used to redraw it with networkx."""
    return f"""
MATCH (ref:{label} {{{key}: $ref}})-[s:SIMILAR_TO]->(other:{label})
RETURN ref.{key} AS source, other.{key} AS target, s.score AS score
ORDER BY score DESC
LIMIT 5
"""


BEFORE_ITEM = f"""
MATCH p = (v:Visitor)-[r:{ALL_RELS}]->(i:Item {{itemId: $ref}})
RETURN p
LIMIT 25
"""
BEFORE_USER = f"""
MATCH p = (v:Visitor {{visitorId: $ref}})-[r:{ALL_RELS}]->(i:Item)
RETURN p
LIMIT 25
"""

BEFORE_TABLE = {
    "item": f"""
MATCH (v:Visitor)-[r:{ALL_RELS}]->(i:Item {{itemId: $ref}})
RETURN v.visitorId AS other, type(r) AS rel
LIMIT 25
""",
    "user": f"""
MATCH (v:Visitor {{visitorId: $ref}})-[r:{ALL_RELS}]->(i:Item)
RETURN i.itemId AS other, type(r) AS rel
LIMIT 25
""",
}
