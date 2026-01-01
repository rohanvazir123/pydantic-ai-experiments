"""
Knowledge Graph Creation from Unstructured Data using Pydantic AI.

This module provides functionality to:
1. Create knowledge graphs from unstructured text using LLM-generated Cypher queries
2. Extract entities and relationships using structured output
3. Query the knowledge graph using natural language

Original LangChain implementation refactored to use Pydantic AI.

Usage:
    python -m rag.agent.kg_agent

Environment Variables (via .env or rag.config.settings):
    LLM_MODEL: Model name (default: llama3.1:8b)
    LLM_BASE_URL: API base URL (default: http://localhost:11434/v1)
    LLM_API_KEY: API key
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USERNAME: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password

Model Requirements:
    For reliable results with structured output and valid Cypher generation,
    use a capable LLM. Local models like Ollama llama3.1:8b may struggle with:
    - Generating complete, valid Cypher syntax
    - Producing properly formatted JSON for structured output
    - Following SCREAMING_SNAKE_CASE conventions for relationship types

    Recommended models:
    - OpenAI GPT-4 or GPT-4o:
        LLM_MODEL=gpt-4
        LLM_BASE_URL=https://api.openai.com/v1
        LLM_API_KEY=your-openai-key

    - Anthropic Claude:
        LLM_MODEL=claude-3-opus-20240229
        LLM_BASE_URL=https://api.anthropic.com/v1
        LLM_API_KEY=your-anthropic-key

    - Larger Ollama models:
        LLM_MODEL=llama3.1:70b
        # or
        LLM_MODEL=mixtral:8x7b
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from neo4j import AsyncDriver, AsyncGraphDatabase
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.config.settings import load_settings

# Load environment variables
load_dotenv(override=True)


# =============================================================================
# CONFIGURATION
# =============================================================================


def get_llm_model() -> OpenAIChatModel:
    """Get the LLM model configured from settings."""
    settings = load_settings()

    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    return OpenAIChatModel(settings.llm_model, provider=provider)


# =============================================================================
# NEO4J GRAPH STORE
# =============================================================================


class Neo4jStore:
    """Direct Neo4j connection without LangChain dependency."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        settings = load_settings()
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )
        # Verify connectivity
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return results."""
        if not self._driver:
            raise RuntimeError("Not connected. Call connect() first.")

        async with self._driver.session() as session:
            result = await session.run(cypher, parameters or {})
            records = await result.data()
            return records

    async def get_schema(self) -> str:
        """Get the database schema as a string."""
        if not self._driver:
            raise RuntimeError("Not connected. Call connect() first.")

        schema_parts = []

        # Get node labels and their properties
        node_query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        RETURN nodeType, collect({name: propertyName, types: propertyTypes}) as properties
        """
        try:
            nodes = await self.query(node_query)
            if nodes:
                schema_parts.append("Node Labels:")
                for node in nodes:
                    props = ", ".join(
                        [f"{p['name']}: {p['types']}" for p in node["properties"]]
                    )
                    schema_parts.append(f"  {node['nodeType']} ({props})")
        except Exception:
            # Fall back to simpler schema query
            labels = await self.query("CALL db.labels()")
            if labels:
                schema_parts.append(f"Node Labels: {[r['label'] for r in labels]}")

        # Get relationship types
        rel_query = "CALL db.relationshipTypes()"
        try:
            rels = await self.query(rel_query)
            if rels:
                rel_types = [
                    r.get("relationshipType", r.get("name", str(r))) for r in rels
                ]
                schema_parts.append(f"Relationship Types: {rel_types}")
        except Exception:
            pass

        return "\n".join(schema_parts) if schema_parts else "Empty database"


# =============================================================================
# STRUCTURED OUTPUT MODELS
# =============================================================================


class Entity(BaseModel):
    """Represents an entity extracted from text."""

    name: str = Field(description="Name of the entity")
    label: str = Field(
        description="Label/type of the entity (e.g., Person, Place, Company)"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Properties of the entity"
    )


class Relationship(BaseModel):
    """Represents a relationship between entities."""

    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    type: str = Field(description="Type of relationship (e.g., FRIENDS_WITH, WORKS_AT)")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Properties of the relationship"
    )


class GraphDocument(BaseModel):
    """A document converted to graph format with entities and relationships."""

    entities: list[Entity] = Field(
        default_factory=list, description="Extracted entities"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="Extracted relationships"
    )


class CypherQuery(BaseModel):
    """A Cypher query generated from text."""

    query: str = Field(description="The Cypher query to create the knowledge graph")


class AnswerResponse(BaseModel):
    """A response to a user query with context from the knowledge graph."""

    answer: str = Field(description="The answer to the user's question")
    cypher_used: str = Field(description="The Cypher query used to retrieve data")


# =============================================================================
# AGENT DEPENDENCIES
# =============================================================================


@dataclass
class KGAgentDeps:
    """Dependencies for knowledge graph agents."""

    neo4j_store: Neo4jStore
    schema: str = ""

    async def initialize(self) -> None:
        """Initialize the dependencies."""
        await self.neo4j_store.connect()
        self.schema = await self.neo4j_store.get_schema()

    async def close(self) -> None:
        """Close connections."""
        await self.neo4j_store.close()


# =============================================================================
# CYPHER GENERATION AGENT (Custom Method)
# =============================================================================

CYPHER_SYSTEM_PROMPT = """
You are a Cypher expert that creates knowledge graphs by generating valid Neo4j Cypher queries.

Task:
- Identify Entities, Relationships and Properties from the provided context
- Generate a complete, valid Cypher query to create the Knowledge Graph
- Extract ALL entities and relationships possible
- Always extract a person's Profession as a separate entity

CRITICAL RULES:
- Generate COMPLETE, VALID Cypher syntax - never truncate or leave incomplete
- All string property values MUST be in double quotes: {name: "John"}
- All relationship types MUST be in SCREAMING_SNAKE_CASE: [:WORKS_AT], [:FRIENDS_WITH]
- Variable names should be CamelCase without spaces: JohnSmith, TechCorp
- Each MERGE statement must be complete and valid
- Do not include explanations, only the Cypher query

Entity labels: Person, Organization, Place, Animal, Profession, Product, Concept

Example:
Context: Mary works at TechCorp. She is friends with John.

MERGE (Mary:Person {name: "Mary"})
MERGE (TechCorp:Organization {name: "TechCorp"})
MERGE (John:Person {name: "John"})
MERGE (Mary)-[:WORKS_AT]->(TechCorp)
MERGE (Mary)-[:FRIENDS_WITH]->(John)
"""

cypher_agent = Agent(
    get_llm_model(),
    system_prompt=CYPHER_SYSTEM_PROMPT,
    output_type=CypherQuery,
    retries=3,
)


# =============================================================================
# GRAPH TRANSFORMER AGENT (Structured Entity/Relationship Extraction)
# =============================================================================

TRANSFORMER_SYSTEM_PROMPT = """
You are an expert at extracting entities and relationships from unstructured text for knowledge graphs.

Task:
- Read the provided text carefully
- Extract ALL entities (people, places, organizations, concepts, etc.)
- Extract ALL relationships between entities
- Be thorough and extract as much information as possible

For each entity:
- name: The full name or identifier (e.g., "John Smith", "TechCorp")
- label: One of: Person, Organization, Place, Animal, Profession, Product, Concept
- properties: Any attributes mentioned (age, title, year, etc.)

For each relationship:
- source: The source entity name (must match an entity name exactly)
- target: The target entity name (must match an entity name exactly)
- type: MUST be in SCREAMING_SNAKE_CASE with underscores (e.g., WORKS_AT, FRIENDS_WITH, GRADUATED_FROM, LOCATED_IN, HAS_PET, FOUNDED_IN)
- properties: Any relationship attributes

CRITICAL: Relationship types MUST be SCREAMING_SNAKE_CASE. Examples:
- "works at" -> WORKS_AT
- "friends with" -> FRIENDS_WITH
- "graduated from" -> GRADUATED_FROM
- "has pet" -> HAS_PET
- "founded in" -> FOUNDED_IN
"""

graph_transformer_agent = Agent(
    get_llm_model(),
    system_prompt=TRANSFORMER_SYSTEM_PROMPT,
    output_type=GraphDocument,
    retries=3,
)


# =============================================================================
# GRAPH QA AGENT
# =============================================================================

CYPHER_GENERATION_PROMPT = """
You are a Cypher expert generating read queries to retrieve data from a Neo4j knowledge graph.

CRITICAL RULES:
1. Use ONLY node labels and relationship types that exist in the schema
2. Use case-insensitive matching with toLower() for name searches
3. Return complete, valid Cypher - no truncation
4. Return ONLY the Cypher query, no explanations
5. Use OPTIONAL MATCH when relationships might not exist
6. Always RETURN meaningful data

Common patterns:
- Find by name: MATCH (n) WHERE toLower(n.name) CONTAINS toLower("search term") RETURN n
- Find relationships: MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name
- Find all of type: MATCH (n:Person) RETURN n.name, n.title

Schema:
{schema}

Examples:
Q: Who works at TechCorp?
A: MATCH (p:Person)-[:WORKS_AT]->(o:Organization) WHERE toLower(o.name) CONTAINS toLower("techcorp") RETURN p.name as person, o.name as company

Q: What pets do people have?
A: MATCH (p:Person)-[:HAS_PET|OWNS]->(a:Animal) RETURN p.name as person, a.name as pet

Q: Where did John graduate from?
A: MATCH (p:Person)-[:GRADUATED_FROM]->(place) WHERE toLower(p.name) CONTAINS toLower("john") RETURN p.name as person, place.name as school
"""

QA_SYSTEM_PROMPT = """
You are a helpful assistant answering questions using data from a knowledge graph.

Instructions:
- Use only the provided context to answer the question
- If no relevant context is provided, explain that you couldn't find the information
- Do not fall back to pre-trained knowledge
- Do not mention "context" in your answer
- Be helpful and friendly
- Do not hallucinate

The context provided comes from an authoritative fact source.
"""


@dataclass
class QADeps:
    """Dependencies for the QA agent."""

    neo4j_store: Neo4jStore
    schema: str = ""


qa_agent = Agent(
    get_llm_model(),
    system_prompt=QA_SYSTEM_PROMPT,
    deps_type=QADeps,
)


# Cypher generation agent for QA
cypher_qa_agent = Agent(
    get_llm_model(),
    output_type=CypherQuery,
    retries=3,
)


@qa_agent.tool
async def query_knowledge_graph(
    ctx: RunContext[QADeps],
    question: str,
) -> str:
    """
    Query the knowledge graph to find information relevant to the question.

    Args:
        ctx: The run context with dependencies.
        question: The user's natural language question.

    Returns:
        Results from the knowledge graph query.
    """
    deps = ctx.deps

    # Generate Cypher query
    cypher_prompt = CYPHER_GENERATION_PROMPT.format(schema=deps.schema)
    cypher_result = await cypher_qa_agent.run(
        f"{cypher_prompt}\n\nQuestion: {question}",
    )

    cypher_query = cypher_result.output.query

    # Execute query
    try:
        results = await deps.neo4j_store.query(cypher_query)
        if results:
            return f"Query: {cypher_query}\n\nResults:\n{results}"
        return f"Query: {cypher_query}\n\nNo results found."
    except Exception as e:
        return f"Error executing query: {e}\nQuery was: {cypher_query}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def create_knowledge_graph_from_cypher(
    store: Neo4jStore, text: str
) -> tuple[str, str]:
    """
    Generate and execute a Cypher query to create a knowledge graph from text.

    Args:
        store: Neo4j store instance.
        text: The unstructured text to process.

    Returns:
        Tuple of (cypher_query, result_message).
    """
    # Generate Cypher query
    result = await cypher_agent.run(f"Context: {text}")
    cypher_query = result.output.query

    # Execute the query
    try:
        await store.query(cypher_query)
        return cypher_query, "Knowledge graph created successfully"
    except Exception as e:
        return cypher_query, f"Error creating graph: {e}"


async def extract_graph_document(text: str) -> GraphDocument:
    """
    Extract entities and relationships from text using structured output.

    Args:
        text: The unstructured text to process.

    Returns:
        GraphDocument with extracted entities and relationships.
    """
    try:
        result = await graph_transformer_agent.run(text)
        return result.output
    except Exception as e:
        # Fallback: try with a simpler prompt asking for JSON directly
        print(f"Structured output failed: {e}")
        print("Trying fallback extraction...")

        fallback_agent = Agent(get_llm_model())
        fallback_prompt = f"""Extract entities and relationships from this text as JSON.

Text: {text}

Return ONLY valid JSON in this exact format (no explanation):
{{"entities": [{{"name": "...", "label": "Person|Organization|Place|Animal|Concept", "properties": {{}}}}], "relationships": [{{"source": "...", "target": "...", "type": "RELATIONSHIP_TYPE", "properties": {{}}}}]}}
"""
        result = await fallback_agent.run(fallback_prompt)
        output = result.output

        # Try to parse JSON from the response
        import json
        import re

        # Find JSON in the response
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return GraphDocument(
                    entities=[Entity(**e) for e in data.get("entities", [])],
                    relationships=[
                        Relationship(**r) for r in data.get("relationships", [])
                    ],
                )
            except (json.JSONDecodeError, TypeError) as parse_error:
                print(f"JSON parsing failed: {parse_error}")

        # Return empty document if all else fails
        return GraphDocument(entities=[], relationships=[])


def graph_document_to_cypher(doc: GraphDocument) -> str:
    """
    Convert a GraphDocument to Cypher MERGE statements.

    Args:
        doc: The graph document with entities and relationships.

    Returns:
        Cypher query string.
    """
    lines = []
    entity_names = {entity.name for entity in doc.entities}
    entity_vars = {entity.name: entity.name.replace(" ", "") for entity in doc.entities}

    # Create entity MERGE statements
    for entity in doc.entities:
        var_name = entity_vars[entity.name]
        props = ", ".join(
            [
                f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
                for k, v in entity.properties.items()
            ]
        )
        if props:
            lines.append(
                f'MERGE ({var_name}:{entity.label} {{name: "{entity.name}", {props}}})'
            )
        else:
            lines.append(f'MERGE ({var_name}:{entity.label} {{name: "{entity.name}"}})')

    # Create relationship MERGE statements (only for known entities)
    for rel in doc.relationships:
        # Check if both source and target are known entities
        if rel.source not in entity_names:
            # Create a placeholder entity for unknown source
            var_name = rel.source.replace(" ", "")
            lines.append(f'MERGE ({var_name}:Concept {{name: "{rel.source}"}})')
            entity_names.add(rel.source)
            entity_vars[rel.source] = var_name

        if rel.target not in entity_names:
            # Create a placeholder entity for unknown target
            var_name = rel.target.replace(" ", "")
            lines.append(f'MERGE ({var_name}:Concept {{name: "{rel.target}"}})')
            entity_names.add(rel.target)
            entity_vars[rel.target] = var_name

        source_var = entity_vars[rel.source]
        target_var = entity_vars[rel.target]
        props = ", ".join(
            [
                f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
                for k, v in rel.properties.items()
            ]
        )
        if props:
            lines.append(
                f"MERGE ({source_var})-[:{rel.type} {{{props}}}]->({target_var})"
            )
        else:
            lines.append(f"MERGE ({source_var})-[:{rel.type}]->({target_var})")

    return "\n".join(lines)


async def add_graph_document(store: Neo4jStore, doc: GraphDocument) -> str:
    """
    Add a GraphDocument to the Neo4j database.

    Args:
        store: Neo4j store instance.
        doc: The graph document to add.

    Returns:
        Result message.
    """
    cypher = graph_document_to_cypher(doc)
    try:
        await store.query(cypher)
        return f"Added {len(doc.entities)} entities and {len(doc.relationships)} relationships"
    except Exception as e:
        return f"Error adding graph document: {e}"


async def query_graph(store: Neo4jStore, question: str) -> str:
    """
    Query the knowledge graph with a natural language question.

    Args:
        store: Neo4j store instance.
        question: The user's question.

    Returns:
        The answer from the QA agent.
    """
    schema = await store.get_schema()
    deps = QADeps(neo4j_store=store, schema=schema)

    result = await qa_agent.run(question, deps=deps)
    return result.output


# =============================================================================
# MAIN EXAMPLE
# =============================================================================


async def main():
    """Demonstrate knowledge graph creation and querying."""
    print("=" * 60)
    print("Knowledge Graph with Pydantic AI")
    print("=" * 60)

    # Initialize Neo4j store
    store = Neo4jStore()

    try:
        await store.connect()
        print("Connected to Neo4j")

        # Test LLM
        print("\n--- Testing LLM ---")
        test_agent = Agent(get_llm_model())
        result = await test_agent.run("What is Neo4j? Answer in one sentence.")
        print(f"LLM Response: {result.output}")

        # Get schema
        print("\n--- Database Schema ---")
        schema = await store.get_schema()
        print(schema)

        # Sample text (simple example for testing)
        content = """
        John works at TechCorp. Sarah also works at TechCorp.
        John is friends with Sarah. Sarah has a cat named Whiskers.
        John graduated from Stanford. TechCorp is located in San Francisco.
        """

        # Method 1: Custom Cypher Generation
        print("\n--- Method 1: Custom Cypher Generation ---")
        cypher, msg = await create_knowledge_graph_from_cypher(store, content)
        print(f"Generated Cypher:\n{cypher}")
        print(f"Result: {msg}")

        # Method 2: Structured Entity/Relationship Extraction
        print("\n--- Method 2: Graph Transformer ---")
        graph_doc = await extract_graph_document(content)
        print(f"Entities: {[e.name for e in graph_doc.entities]}")
        print(
            f"Relationships: {[(r.source, r.type, r.target) for r in graph_doc.relationships]}"
        )

        # Convert to Cypher and add to graph
        cypher2 = graph_document_to_cypher(graph_doc)
        print(f"\nGenerated Cypher:\n{cypher2}")
        result_msg = await add_graph_document(store, graph_doc)
        print(f"Result: {result_msg}")

        # Debug: Show what's in the database
        print("\n--- Database Contents ---")
        try:
            nodes = await store.query(
                "MATCH (n) RETURN labels(n) as labels, n.name as name LIMIT 20"
            )
            print(f"Nodes: {nodes}")
            rels = await store.query(
                "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
            )
            print(f"Relationships: {rels}")
        except Exception as e:
            print(f"Debug query error: {e}")

        # Query the graph
        print("\n--- Querying the Graph ---")
        questions = [
            "Who works at TechCorp?",
            "What pets do people have?",
            "Where did John graduate from?",
        ]

        for question in questions:
            print(f"\nQuestion: {question}")
            answer = await query_graph(store, question)
            print(f"Answer: {answer}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await store.close()
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
