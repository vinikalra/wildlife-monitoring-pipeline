from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    airtable_api_key: str
    airtable_base_id: str
    airtable_table_name: str
    source_dir: Path
    birds_dir: Path
    animals_dir: Path
    rejected_dir: Path
    confidence_threshold: float

    @classmethod
    def load(cls) -> "Config":
        def resolve(env_var: str, default: str) -> Path:
            return (ROOT_DIR / os.getenv(env_var, default)).resolve()

        return cls(
            airtable_api_key=os.environ["AIRTABLE_API_KEY"],
            airtable_base_id=os.environ["AIRTABLE_BASE_ID"],
            airtable_table_name=os.getenv("AIRTABLE_TABLE_NAME", "Camera Trap Images"),
            source_dir=resolve("SOURCE_DIR", "data/download"),
            birds_dir=resolve("BIRDS_DIR", "data/classified/birds"),
            animals_dir=resolve("ANIMALS_DIR", "data/classified/animals"),
            rejected_dir=resolve("REJECTED_DIR", "data/rejected"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.3")),
        )
