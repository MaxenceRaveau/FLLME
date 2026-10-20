import importlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from fllme.errors import ProviderError
from fllme.generate import configure, generate, get_service
from fllme.models.deployment import AdapterType, Deployment
from fllme.models.input import GenerationInput
from fllme.models.message import Message, MessageSource, TextContent
from fllme.models.model import LLMModel, Provider
from fllme.models.output import (
    FinishReason,
    GenerationOutput,
    Response,
    StreamDelta,
    TextDelta,
)
from fllme.service import DeploymentService


def _gen_mod() -> Any:
    return importlib.import_module("fllme.generate")


class TestConfigureAndGetService:
    def test_get_service_unconfigured(self) -> None:
        mod = _gen_mod()
        original = mod.DEFAULT_SERVICE
        mod.DEFAULT_SERVICE = None
        try:
            with pytest.raises(RuntimeError, match="No service configured"):
                get_service()
        finally:
            mod.DEFAULT_SERVICE = original

    @pytest.mark.asyncio
    async def test_configure(self) -> None:
        mod = _gen_mod()
        original = mod.DEFAULT_SERVICE
        try:
            svc = await DeploymentService.from_sqlite(":memory:")
            configure(svc)
            assert get_service() is svc
        finally:
            mod.DEFAULT_SERVICE = original


def _sse(*chunks: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode()


_GEMINI_SSE = _sse(
    {"candidates": [{"content": {"parts": [{"text": "Hello "}], "role": "model"}}]},
    {
        "candidates": [
            {
                "content": {"parts": [{"text": "world"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    },
)


def _input(*, stream: bool) -> GenerationInput:
    return GenerationInput(
        model="gemini",
        stream=stream,
        conversation=[
            Message(source=MessageSource.USER, contents=[TextContent(text="Hi")])
        ],
    )


def _text(output: GenerationOutput) -> str:
    return "".join(
        c.text for c in output.message.contents if isinstance(c, TextContent)
    )


def _as_stream(
    result: GenerationOutput | AsyncIterator[Response],
) -> AsyncIterator[Response]:
    assert not isinstance(result, GenerationOutput)
    return result


def _delta_text(response: Response) -> str:
    assert isinstance(response.content, StreamDelta)
    assert isinstance(response.content.delta, TextDelta)
    return response.content.delta.text


@pytest_asyncio.fixture
async def service() -> AsyncGenerator[DeploymentService]:
    svc = await DeploymentService.from_sqlite(":memory:")
    await svc.add_model(LLMModel(id="gemini", name="gemini", provider=Provider.GEMINI))
    await svc.add_deployment(
        Deployment(
            id="gemini-vertex",
            url="https://example.test/gemini:streamGenerateContent?alt=sse",
            model_id="gemini",
            adapter=AdapterType.GEMINI_VERTEX_V1,
            auth_id="google_adc",
        )
    )
    with patch.object(svc, "get_auth_headers", AsyncMock(return_value={})):
        yield svc


def _mock_client(
    status_code: int = 200, body: bytes = _GEMINI_SSE
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


class TestGenerateStreamArgument:
    @pytest.mark.asyncio
    async def test_stream_false_returns_generation_output(
        self, service: DeploymentService
    ) -> None:
        client, requests = _mock_client()
        with patch("fllme.generate.get_client", return_value=client):
            result = await generate(_input(stream=False), service=service)

        assert isinstance(result, GenerationOutput)
        assert _text(result) == "Hello world"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.total_tokens == 5
        assert len(requests) == 1

    @pytest.mark.asyncio
    async def test_stream_true_yields_deltas_then_output(
        self, service: DeploymentService
    ) -> None:
        client, _ = _mock_client()
        with patch("fllme.generate.get_client", return_value=client):
            stream = _as_stream(await generate(_input(stream=True), service=service))
            responses = [r async for r in stream]

        assert all(isinstance(r, Response) for r in responses)
        *deltas, last = responses
        assert [_delta_text(d) for d in deltas] == ["Hello ", "world"]
        assert isinstance(last.content, GenerationOutput)
        assert _text(last.content) == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_true_is_lazy(self, service: DeploymentService) -> None:
        client, requests = _mock_client()
        with patch("fllme.generate.get_client", return_value=client):
            stream = _as_stream(await generate(_input(stream=True), service=service))
            assert requests == []
            [r async for r in stream]
        assert len(requests) == 1

    @pytest.mark.asyncio
    async def test_stream_false_and_true_give_same_output(
        self, service: DeploymentService
    ) -> None:
        client, _ = _mock_client()
        with patch("fllme.generate.get_client", return_value=client):
            collected = await generate(_input(stream=False), service=service)
            stream = _as_stream(await generate(_input(stream=True), service=service))
            streamed = [r async for r in stream][-1].content

        assert isinstance(collected, GenerationOutput)
        assert isinstance(streamed, GenerationOutput)
        assert collected.model_dump(exclude={"id"}) == streamed.model_dump(
            exclude={"id"}
        )

    @pytest.mark.asyncio
    async def test_stream_false_raises_provider_error(
        self, service: DeploymentService
    ) -> None:
        client, _ = _mock_client(status_code=429, body=b"quota exceeded")
        with (
            patch("fllme.generate.get_client", return_value=client),
            pytest.raises(ProviderError, match="quota exceeded") as exc_info,
        ):
            await generate(_input(stream=False), service=service)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_stream_true_raises_provider_error_on_iteration(
        self, service: DeploymentService
    ) -> None:
        client, _ = _mock_client(status_code=500, body=b"boom")
        with patch("fllme.generate.get_client", return_value=client):
            stream = _as_stream(await generate(_input(stream=True), service=service))
            with pytest.raises(ProviderError, match="boom") as exc_info:
                [r async for r in stream]
        assert exc_info.value.status_code == 500
