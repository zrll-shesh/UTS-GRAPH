"""Central configuration. Everything that is a "knob" lives here, not in the notebook."""
from pathlib import Path

STUDENT = {
    "name": "Nazril Ravi Pratama",
    "nim": "24031554129",
    "class": "International 24",
    "course": "Graf Sains Data",
}
SEED = 24031554129
SEED32 = SEED % 2**32

ROOT = Path(__file__).resolve().parents[1]
RAW_EVENTS = ROOT / "data" / "raw" / "events.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "events_p1_train.csv"
TEST_CSV = PROCESSED_DIR / "events_p2_test.csv"
FIG_DIR = ROOT / "figures"

PERIOD_P1 = ("2015-06-15 00:00:00", "2015-07-12 23:59:59")
PERIOD_P2 = ("2015-07-13 00:00:00", "2015-07-26 23:59:59")

REL_BY_EVENT = {"view": "VIEWED", "addtocart": "ADDED_TO_CART", "transaction": "BOUGHT"}
REL_WEIGHTS = {"VIEWED": 1.0, "ADDED_TO_CART": 3.0, "BOUGHT": 5.0}

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_DATABASE = "retailrocket"
BATCH_SIZE = 10_000

MAX_USER_ITEMS = 100
MIN_SUPPORT = 3
K_NEIGHBOURS = 5
N_ADVERTISED = 3
TOP_N = 5
N_EVAL_USERS = 300
