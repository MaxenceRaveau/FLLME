import pytest

from fllme.errors import SerializationError
from fllme.models.input import (
    GenerationInput,
    ToolsCallingMode,
    ToolsConfig,
    UrlContextTool,
)
from fllme.models.message import Message, MessageSource, TextContent
from fllme.providers.mistral.vertex_v1 import MistralVertexV1


def _simple_input(text: str = "Hello") -> GenerationInput:
    return GenerationInput(
        model="mistral-medium",
        conversation=[
            Message(
                source=MessageSource.USER,
                contents=[TextContent(text=text)],
            ),
        ],
    )


class TestSerialize:
    def test_url_context_tool_raises(self) -> None:
        adapter = MistralVertexV1()
        inp = _simple_input()
        inp = inp.model_copy(
            update={
                "tool_config": ToolsConfig(
                    tools=[UrlContextTool()],
                    parallel_calling=False,
                    mode=ToolsCallingMode.AUTO,
                ),
            }
        )
        with pytest.raises(
            SerializationError, match="URL context tool is not supported"
        ):
            adapter.serialize(inp)
