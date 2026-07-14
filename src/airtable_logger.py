from datetime import datetime, timezone

from pyairtable import Api

from classifier import ClassificationResult
from metadata import ImageMetadata


class AirtableLogger:
    def __init__(self, api_key: str, base_id: str, table_name: str):
        self.table = Api(api_key).table(base_id, table_name)

    def log(
        self,
        metadata: ImageMetadata,
        result: ClassificationResult,
        original_path: str,
        new_path: str,
    ) -> None:
        self.table.create(
            {
                "File Name": metadata.filename,
                "Season": metadata.season,
                "Station": metadata.station,
                "Roll": metadata.roll,
                "Frame": metadata.frame,
                "Captured At": metadata.captured_at,
                "Category": result.category,
                "Predicted Label": result.label,
                "Confidence": round(result.confidence, 4),
                "Original Path": original_path,
                "New Path": new_path,
                "Processed At": datetime.now(timezone.utc).isoformat(),
            }
        )
