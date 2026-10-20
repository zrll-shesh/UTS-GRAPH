"""Neo4j access layer: connect, load the training fold, and read / write results."""
from __future__ import annotations

import argparse
import getpass
import sys
from typing import Iterable, Iterator

import pandas as pd
from neo4j import Driver, GraphDatabase, RoutingControl
from tqdm.auto import tqdm

from . import config, queries
from .data_prep import load_events


def connect(uri: str, user: str, password: str) -> Driver:
    """Open a driver and fail early, with a readable message, if the server is unreachable."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except Exception as exc:
        driver.close()
        raise ConnectionError(f"Cannot reach Neo4j at {uri}: {exc}") from exc
    return driver


def ensure_database(driver: Driver, name: str) -> None:
    """Create the database if it does not exist (needs a multi-database edition such as Desktop)."""
    if name in ("neo4j", "system"):
        return
    driver.execute_query(
        "CREATE DATABASE $name IF NOT EXISTS WAIT",
        parameters_={"name": name},
        database_="system",
    )


def run_df(driver: Driver, query: str, database: str, **params) -> pd.DataFrame:
    """Run a read query and return the result as a DataFrame."""
    records, _, keys = driver.execute_query(
        query, parameters_=params, database_=database, routing_=RoutingControl.READ
    )
    return pd.DataFrame([dict(r) for r in records], columns=list(keys))


def run_write(driver: Driver, query: str, database: str, **params) -> None:
    """Run a write query inside a managed transaction."""
    driver.execute_query(query, parameters_=params, database_=database)


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def create_constraints(driver: Driver, database: str) -> None:
    for statement in queries.CONSTRAINTS:
        run_write(driver, statement, database)


def reset_database(driver: Driver, database: str, size: int = 10_000) -> int:
    """Delete every node in small batches, so a big graph does not exhaust the transaction memory."""
    total = 0
    while True:
        records, _, _ = driver.execute_query(queries.DELETE_BATCH, parameters_={"size": size}, database_=database)
        deleted = records[0]["deleted"]
        total += deleted
        if deleted == 0:
            return total


def import_nodes(driver: Driver, database: str, visitor_ids: Iterable[int], item_ids: Iterable[int], batch_size: int) -> None:
    """Phase 1: create every Visitor and Item exactly once."""
    for label, key, ids in (("Visitor", "visitorId", visitor_ids), ("Item", "itemId", item_ids)):
        statement = queries.MERGE_NODES.format(label=label, key=key)
        ids = [int(i) for i in ids]
        for chunk in tqdm(list(_batches(ids, batch_size)), desc=f"Nodes :{label}"):
            run_write(driver, statement, database, ids=chunk)


def import_relationships(driver: Driver, database: str, events: pd.DataFrame, batch_size: int) -> None:
    """Phase 2: one relationship per event, grouped by relationship type."""
    for event_name, rel_type in config.REL_BY_EVENT.items():
        subset = events.loc[events["event"] == event_name]
        transaction_ids = [None if pd.isna(t) else int(t) for t in subset["transactionid"]]
        rows = [
            {"visitorId": v, "itemId": i, "timestamp": t, "transactionId": tx}
            for v, i, t, tx in zip(
                subset["visitorid"].tolist(), subset["itemid"].tolist(), subset["timestamp"].tolist(), transaction_ids
            )
        ]
        for chunk in tqdm(list(_batches(rows, batch_size)), desc=f"Rel  :{rel_type}"):
            run_write(driver, queries.CREATE_RELATIONSHIPS[rel_type], database, rows=chunk)


def load_training_graph(driver: Driver, database: str, events: pd.DataFrame, batch_size: int = config.BATCH_SIZE) -> None:
    """Build the whole training graph from a filtered events DataFrame (idempotent: it starts from empty)."""
    ensure_database(driver, database)
    create_constraints(driver, database)
    removed = reset_database(driver, database)
    if removed:
        print(f"Cleared {removed:,} existing nodes from '{database}'.")
    import_nodes(driver, database, events["visitorid"].unique(), events["itemid"].unique(), batch_size)
    import_relationships(driver, database, events, batch_size)


def fetch_composition(driver: Driver, database: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = run_df(driver, queries.NODE_COUNTS, database)
    rels = run_df(driver, queries.RELATIONSHIP_COUNTS, database)
    return nodes, rels


def fetch_interactions(driver: Driver, database: str) -> pd.DataFrame:
    return run_df(driver, queries.INTERACTIONS, database)


def write_similarity(driver: Driver, database: str, label: str, key: str, pairs: pd.DataFrame, batch_size: int = 5_000) -> int:
    """pairs needs the columns source, target, score, rank. Returns the number of rows written."""
    rows = pairs[["source", "target", "score", "rank"]].astype(
        {"source": "int64", "target": "int64", "score": "float64", "rank": "int64"}
    ).to_dict("records")
    statement = queries.write_similarity(label, key)
    for chunk in _batches(rows, batch_size):
        run_write(driver, statement, database, rows=chunk)
    return len(rows)


def clear_similarity(driver: Driver, database: str) -> None:
    run_write(driver, queries.SIMILARITY_DELETE, database)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the RetailRocket training fold (events_p1_train.csv) into Neo4j.")
    parser.add_argument("-u", "--user", default=config.NEO4J_USER, help="Neo4j username")
    parser.add_argument("-p", "--password", default=None, help="Neo4j password (prompted when omitted)")
    parser.add_argument("-b", "--bolt", default=config.NEO4J_URI, help="Neo4j URI")
    parser.add_argument("-d", "--database", default=config.NEO4J_DATABASE, help="Target database name")
    parser.add_argument("-s", "--source", default=str(config.TRAIN_CSV), help="CSV file to import")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    password = args.password or getpass.getpass("Neo4j password: ")
    try:
        events = load_events(args.source)
        driver = connect(args.bolt, args.user, password)
    except (FileNotFoundError, ConnectionError) as exc:
        sys.exit(f"Error: {exc}")
    with driver:
        load_training_graph(driver, args.database, events, args.batch_size)
    print("Import finished.")


if __name__ == "__main__":
    main()
