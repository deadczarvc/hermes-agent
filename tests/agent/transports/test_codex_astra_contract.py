"""Wire-level contract tests for GPT-6 Astra on the Responses transport."""

import pytest
from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401

    return get_transport("codex_responses")


def _messages():
    return [{"role": "user", "content": "test"}]


def _final_wire(transport, **params):
    built = transport.build_kwargs(messages=_messages(), tools=[], **params)
    return transport.preflight_kwargs(built)


def test_astra_effort_contract_and_legacy_gpt55_remain_distinct(transport):
    astra = _final_wire(
        transport,
        model="gpt-6-astra",
        reasoning_config={"enabled": True, "effort": "max"},
    )
    assert astra["reasoning"]["effort"] == "max"

    for invalid in ("none", "minimal", "ultra", 3):
        with pytest.raises(ValueError, match="GPT-6 Astra.*reasoning effort"):
            transport.build_kwargs(
                model="gpt-6-astra",
                messages=_messages(),
                tools=[],
                reasoning_config={"enabled": True, "effort": invalid},
            )

    legacy_none = _final_wire(
        transport,
        model="gpt-5.5",
        reasoning_config={"enabled": True, "effort": "none"},
        request_overrides={"temperature": 0},
    )
    assert legacy_none["reasoning"]["effort"] == "none"
    assert legacy_none["temperature"] == 0.0
    assert _final_wire(
        transport,
        model="gpt-5.5",
        reasoning_config={"enabled": True, "effort": "max"},
    )["reasoning"]["effort"] == "xhigh"


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"temperature": 0}, "temperature"),
        ({"top_p": None}, "top_p"),
        ({"top_logprobs": False}, "top_logprobs"),
        ({"include": ["reasoning.encrypted_content", "message.output_text.logprobs"]}, "include"),
        ({"reasoning": {"effort": "none"}}, "reasoning.effort"),
        ({"extra_body": {"temperature": 0}}, "extra_body.temperature"),
        ({"extra_body": {"include": ["message.output_text.logprobs"]}}, "extra_body.include"),
        ({"extra_body": {"reasoning": {"effort": "none"}}}, "extra_body.reasoning.effort"),
    ],
)
def test_astra_final_preflight_rejects_forbidden_late_overrides(transport, override, field):
    built = transport.build_kwargs(
        model="gpt-6-astra",
        messages=_messages(),
        tools=[],
        request_overrides=override,
    )

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        transport.preflight_kwargs(built)


def test_astra_tool_conversion_is_sync_only_and_never_silently_drops_lifecycle_metadata(transport):
    synchronous = transport.convert_tools(
        [{
            "type": "function",
            "async": False,
            "function": {
                "name": "terminal",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
    )
    sync_wire_tool = transport.preflight_kwargs({
        "model": "gpt-6-astra",
        "instructions": "test",
        "input": _messages(),
        "tools": synchronous,
        "store": False,
    })["tools"][0]
    assert sync_wire_tool["name"] == "terminal"
    assert "async" not in sync_wire_tool

    with pytest.raises(ValueError, match="async tool lifecycle is unsupported"):
        transport.convert_tools(
            [{
                "type": "function",
                "async": True,
                "function": {
                    "name": "terminal",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]
        )
    with pytest.raises(ValueError, match="custom tool lifecycle is unsupported"):
        transport.convert_tools(
            [{"type": "custom", "name": "shell", "format": {"type": "text"}}]
        )


def test_output_cap_final_wire_is_sent_only_to_supported_responses_endpoints(transport):
    supported_wire = _final_wire(
        transport,
        model="gpt-6-astra",
        max_tokens=128000,
        is_codex_backend=False,
    )
    assert supported_wire["max_output_tokens"] == 128000

    codex_wire = _final_wire(
        transport,
        model="gpt-6-astra",
        max_tokens=128000,
        is_codex_backend=True,
    )
    assert "max_output_tokens" not in codex_wire

    with pytest.raises(ValueError, match="Codex backend.*max_output_tokens"):
        transport.build_kwargs(
            model="gpt-6-astra",
            messages=_messages(),
            tools=[],
            is_codex_backend=True,
            request_overrides={"max_output_tokens": 128000},
        )
    with pytest.raises(ValueError, match="Codex backend.*max_output_tokens"):
        transport.build_kwargs(
            model="gpt-6-astra",
            messages=_messages(),
            tools=[],
            is_codex_backend=True,
            request_overrides={"extra_body": {"max_output_tokens": 128000}},
        )


def test_astra_store_false_replays_reasoning_and_exact_call_ids(transport):
    call_id = "call_astra_exact"
    messages = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "sealed", "summary": []},
            ],
            "tool_calls": [
                {
                    "id": "fc_astra_exact",
                    "call_id": call_id,
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "ok"},
    ]

    wire = transport.preflight_kwargs(transport.build_kwargs(
        model="gpt-6-astra",
        messages=messages,
        tools=[],
        reasoning_config={"enabled": True, "effort": "low"},
    ))
    call = next(item for item in wire["input"] if item.get("type") == "function_call")
    output = next(item for item in wire["input"] if item.get("type") == "function_call_output")
    reasoning = next(item for item in wire["input"] if item.get("type") == "reasoning")
    assert wire["store"] is False
    assert wire["include"] == ["reasoning.encrypted_content"]
    assert reasoning["encrypted_content"] == "sealed"
    assert call["call_id"] == output["call_id"] == call_id
