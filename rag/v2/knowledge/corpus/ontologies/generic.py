"""Default generic ontology — used when no corpus-specific template is set.

Extracts named entities and generic relationships without domain specifics.
Suitable for any corpus where graph extraction is enabled but no ontology
has been uploaded via POST /v1/corpus/{id}/ontology.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def edge(label: str, **kwargs: Any) -> Any:
    """Required helper — marks a field as a directed graph edge."""
    if "default" not in kwargs and "default_factory" not in kwargs:
        kwargs["default"] = ...
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


class GenericEntity(BaseModel):
    """A named entity extracted from the document."""

    model_config = ConfigDict()
    graph_id_fields: list[str] = ["name"]

    name: str = Field(
        description=(
            "Entity name — the most specific identifier available. "
            "LOOK FOR: Proper nouns, organisation names, people, dates, products. "
            "EXTRACT: As written in the document. "
            "EXAMPLES: 'Apple Inc', 'John Smith', 'ISO 27001', 'Q4 2025'"
        ),
        examples=["Apple Inc", "John Smith", "ISO 27001"],
    )
    entity_type: str = Field(
        description=(
            "Type of entity. "
            "EXAMPLES: 'Organization', 'Person', 'Location', 'Concept', "
            "'Date', 'Product', 'Regulation', 'Event'"
        ),
        examples=["Organization", "Person", "Location"],
    )
    description: str | None = Field(
        None,
        description="Brief description from the document context. Omit if not clear.",
    )
    related_to: list["GenericEntity"] = edge(
        label="RELATED_TO",
        default_factory=list,
        description=(
            "Other entities this entity is related to based on document context. "
            "Only include when the relationship is clearly stated."
        ),
    )


class GenericDocument(BaseModel):
    """Root entity: the document itself."""

    model_config = ConfigDict()
    graph_id_fields: list[str] = ["title"]

    title: str = Field(
        description=(
            "Document title or best identifying label. "
            "LOOK FOR: Title page, first heading, filename. "
            "EXTRACT: As written."
        ),
        examples=["Employee Handbook 2025", "Q4 Business Review"],
    )
    entities: list[GenericEntity] = edge(
        label="MENTIONS",
        default_factory=list,
        description="All named entities mentioned in the document.",
    )


GenericEntity.model_rebuild()
GenericDocument.model_rebuild()
