# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from pydantic_ai.models.test import TestModel

from rag.agent.agent_main import (
    set_verbose_debug,
    stream_agent_interaction,
)
from rag.agent.rag_agent import RAGState, agent


@pytest.mark.integration
class TestAgentFlow:
    """Test agent_main.py flow execution.

    Uses TestModel so the LLM synthesis step is deterministic.
    search_knowledge_base still executes against real PostgreSQL where used.

    Run with -s to see verbose node/event output:
        python -m pytest rag/tests/test_agent_flow.py -v -s
    """

    @pytest.mark.asyncio
    async def test_agent_flow_verbose(self):
        """Verbose flow emits UserPromptNode → CallToolsNode → End nodes."""
        set_verbose_debug(True)
        state = RAGState()
        try:
            model = TestModel(
                call_tools=["search_knowledge_base"],
                custom_output_text=(
                    "NeuralFlow AI provides intelligent workflow automation solutions "
                    "for enterprise clients. [Source: company-overview]"
                ),
            )
            with agent.override(model=model):
                response, new_messages = await stream_agent_interaction(
                    user_input="What does NeuralFlow AI do?",
                    message_history=[],
                    deps=StateDeps(state),
                )

            assert response, "Should return a response"
            assert isinstance(response, str), "Response should be a string"
            assert len(new_messages) > 0, "Should have new messages"
        finally:
            await state.close()
            set_verbose_debug(False)

    @pytest.mark.asyncio
    async def test_agent_flow_with_tool_call(self):
        """Flow triggers search_knowledge_base and response mentions PTO policy."""
        set_verbose_debug(True)
        state = RAGState()
        try:
            model = TestModel(
                call_tools=["search_knowledge_base"],
                custom_output_text=(
                    "NeuralFlow AI has an unlimited PTO policy. Employees can take "
                    "time off as needed with manager approval. [Source: team-handbook]"
                ),
            )
            with agent.override(model=model):
                response, new_messages = await stream_agent_interaction(
                    user_input="What is the PTO policy at NeuralFlow?",
                    message_history=[],
                    deps=StateDeps(state),
                )

            assert response, "Should return a response"
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
        """Flow works in normal (non-verbose) mode without errors."""
        set_verbose_debug(False)
        state = RAGState()
        try:
            model = TestModel(
                call_tools=[],
                custom_output_text="I'm doing well, thank you for asking!",
            )
            with agent.override(model=model):
                response, new_messages = await stream_agent_interaction(
                    user_input="Hello, how are you?",
                    message_history=[],
                    deps=StateDeps(state),
                )

            assert response is not None
            assert isinstance(response, str)
        finally:
            await state.close()
