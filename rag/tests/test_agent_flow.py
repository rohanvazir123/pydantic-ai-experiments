"""Test to verify agent_main flow execution with verbose debugging.

Run this test to see all Pydantic AI events during agent execution.

Usage:
    python -m pytest rag/tests/test_agent_flow.py -v -s

The -s flag shows verbose debug output including:
    - Node types (UserPromptNode, ModelRequestNode, CallToolsNode, End)
    - Streaming events (PartStartEvent, PartDeltaEvent, PartEndEvent)
    - Tool events (FunctionToolCallEvent, FunctionToolResultEvent)
"""

import pytest
from pydantic_ai.ag_ui import StateDeps

from rag.agent.agent_main import (
    set_verbose_debug,
    stream_agent_interaction,
)
from rag.agent.rag_agent import RAGState


class TestAgentFlow:
    """Test agent_main.py flow execution with verbose debugging."""

    @pytest.mark.asyncio
    async def test_agent_flow_verbose(self):
        """
        Test agent flow with verbose debugging enabled.

        This prints all Pydantic AI events from agent_main.py:
        - NODE #1: UserPromptNode
        - NODE #2: ModelRequestNode (PartStartEvent, PartDeltaEvent, etc.)
        - NODE #3: CallToolsNode (FunctionToolCallEvent, FunctionToolResultEvent)
        - NODE #4+: Additional model/tool nodes
        - End node

        Run with -s to see all events:
            python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_verbose -v -s
        """
        # Enable verbose debugging
        set_verbose_debug(True)

        # Create shared state with pre-initialized store (better performance)
        state = await RAGState.create()

        try:
            query = "What does NeuralFlow AI do?"
            deps = StateDeps(state)

            response, new_messages = await stream_agent_interaction(
                user_input=query,
                message_history=[],
                deps=deps,
            )

            # Basic assertions
            assert response, "Should return a response"
            assert isinstance(response, str), "Response should be a string"
            assert len(new_messages) > 0, "Should have new messages"

        finally:
            # Clean up resources
            await state.close()
            set_verbose_debug(False)

    @pytest.mark.asyncio
    async def test_agent_flow_with_tool_call(self):
        """
        Test agent flow that triggers a tool call (search_knowledge_base).

        Run with -s to see verbose output:
            python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_with_tool_call -v -s
        """
        set_verbose_debug(True)

        # Create shared state with pre-initialized store (better performance)
        state = await RAGState.create()

        try:
            query = "What is the PTO policy at NeuralFlow?"
            deps = StateDeps(state)

            response, new_messages = await stream_agent_interaction(
                user_input=query,
                message_history=[],
                deps=deps,
            )

            assert response, "Should return a response"
            # PTO query should trigger tool call and return relevant info
            response_lower = response.lower()
            assert any(
                term in response_lower
                for term in ["pto", "time off", "vacation", "leave", "days", "policy"]
            ), f"Response should mention PTO-related terms: {response[:200]}"

        finally:
            await state.close()
            set_verbose_debug(False)

    @pytest.mark.asyncio
    async def test_agent_flow_no_verbose(self):
        """
        Test agent flow without verbose debugging (normal mode).

        This verifies the agent works correctly without debug output.

        Run:
            python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_no_verbose -v -s
        """
        # Ensure verbose is disabled
        set_verbose_debug(False)

        # Create shared state with pre-initialized store (better performance)
        state = await RAGState.create()

        try:
            query = "Hello, how are you?"
            deps = StateDeps(state)

            response, new_messages = await stream_agent_interaction(
                user_input=query,
                message_history=[],
                deps=deps,
            )

            assert response is not None
            assert isinstance(response, str)

        finally:
            await state.close()
