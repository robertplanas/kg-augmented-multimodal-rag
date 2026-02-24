from pydantic import BaseModel
from typing import List, Optional, Field


class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


class UnstructuredRelation(BaseModel):
    # 1. Normalization & Identity
    head: str = Field(
        description="Normalized name of the subject (e.g., 'Apple Inc.' instead of 'Apple')"
    )
    head_type: str = Field(
        description="Upper-case entity label (e.g., ORGANIZATION, PERSON, GEO_POLITICAL_ENTITY)"
    )

    # 2. Relationship Predicate
    relation: str = Field(
        description="The verb/action connecting nodes, ideally in CONSTANT_CASE (e.g., WORKS_AT, ACQUIRED)"
    )

    # 3. Tail Node
    tail: str = Field(description="Normalized name of the object")
    tail_type: str = Field(description="Upper-case entity label of the object")

    # 4. Critical Metadata for Graphs
    confidence: float = Field(
        ge=0,
        le=1,
        description="The LLM's certainty score (0.0 to 1.0) regarding this specific fact.",
    )
    fact_provenance: str = Field(
        description="A direct quote or snippet from the text that proves this relationship."
    )
    properties: Optional[dict] = Field(
        default_factory=dict,
        description="Additional temporal or quantitative data (e.g., {'start_date': '2024', 'amount': '5B', 'role': 'CEO'})",
    )
