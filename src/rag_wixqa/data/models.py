from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KBDoc:
    doc_id: str
    title: Optional[str]
    text: str


@dataclass
class QAPair:
    question: str
    answer: str
    kb_ids: List[str]   # ids of relevant KB docs (adapt column name to real schema)
