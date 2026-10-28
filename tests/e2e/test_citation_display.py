import json

import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from app.config.settings import clear_settings_cache
from app.main import create_app
from app.repositories.session_repository import SessionRepository
from app.schemas.retrieval import Citation, RetrievedChunk


@pytest.fixture
def client(monkeypatch):
    # This SSE version caches its event globally; each TestClient owns a new loop.
    monkeypatch.setattr(AppStatus, "should_exit_event", None)
    clear_settings_cache()
    with TestClient(create_app()) as client:
        yield client


def _result(answer):
    return {
        "final_answer": answer,
        "retrieved_chunks": [
            RetrievedChunk(
                chunk_id="faq_1021b2da_01",
                document_id="faq_1021b2da",
                title="faq",
                content="电子发票在确认收货后 48 小时内开具。",
                score=1.0,
            ),
        ],
        "citations": [
            Citation(
                chunk_id="faq_1021b2da_01",
                document_id="faq_1021b2da",
                title="faq",
                section="发票",
            ),
        ],
    }


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("48 小时内开具。 [faq_1021b2da_01]", "48 小时内开具。"),
        ("48 小时内开具[faq_1021b2da_01]。", "48 小时内开具。"),
        ("48 小时内开具。 `faq_1021b2da_01`", "48 小时内开具。"),
        ("48 小时内开具。 faq_1021b2da_01", "48 小时内开具。"),
        ("See faq_1021b2da_01.", "See."),
        ("[faq_1021b2da_01]", "暂时无法回答，请稍后重试。"),
        (
            "48 小时内开具。 [faq_1021b2da_01, invoice_vat_d600f1a3_00]",
            "48 小时内开具。",
        ),
        ("48 小时内开具。【invoice_vat_d600f1a3_00】", "48 小时内开具。"),
        (
            "48 小时内开具。 [faq_1021b2da_01](#faq_1021b2da_01)",
            "48 小时内开具。",
        ),
        ("请查看[发票](#faq_1021b2da_01)。", "请查看发票。"),
        (
            "请查看 [订单详情](https://example.com/orders)。\n\n[1] 时限为 [48 小时]。",
            "请查看 [订单详情](https://example.com/orders)。\n\n[1] 时限为 [48 小时]。",
        ),
    ],
)
def test_response_removes_internal_ids_without_changing_sources(client, answer, expected):
    state = _result(answer)
    response = client.app.state.container.chat_service._to_response(
        state, session_id="citation-test", trace_id="trace-test"
    )
    assert response.answer == expected
    assert response.citations == state["citations"]
    assert state["final_answer"] == answer


def test_response_removes_known_nonstandard_id(client):
    state = _result("48 小时内开具。[legacy-chunk]")
    state["citations"][0].chunk_id = "legacy-chunk"
    response = client.app.state.container.chat_service._to_response(
        state, session_id="citation-test", trace_id="trace-test"
    )
    assert response.answer == "48 小时内开具。"
    assert response.citations[0].chunk_id == "legacy-chunk"


class _FinalAnswerGraph:
    """Deterministic graph output, including a citation leaked by a rewrite."""

    async def ainvoke(self, state):
        return {**state, **_result("48 小时内开具。[faq_1021b2da_01]")}

    async def astream(self, state, **kwargs):
        yield {"check_faithfulness": _result("48 小时内开具。[faq_1021b2da_01]")}


@pytest.mark.parametrize("stream", [False, True])
def test_chat_and_stream_publish_clean_answer_with_sources(client, stream):
    client.app.state.container.chat_service._graph = _FinalAnswerGraph()
    path = "/api/v1/chat/stream" if stream else "/api/v1/chat"
    response = client.post(path, json={"message": "电子发票多久开具？", "user_id": "citation-test"})
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[5:].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        final = next(event["data"] for event in events if event["event"] == "answer")
        tokens = "".join(event["text"] for event in events if event["event"] == "token")
        assert tokens == "48 小时内开具。"
    else:
        final = response.json()
    assert final["answer"] == "48 小时内开具。"
    assert final["citations"] == [
        {
            "chunk_id": "faq_1021b2da_01",
            "document_id": "faq_1021b2da",
            "title": "faq",
            "section": "发票",
        }
    ]

    async def read_saved_answer():
        async with client.app.state.container.session_factory() as session:
            messages = await SessionRepository(session).recent_messages(final["session_id"])
            return messages[-1].content

    assert client.portal.call(read_saved_answer) == "48 小时内开具。"
