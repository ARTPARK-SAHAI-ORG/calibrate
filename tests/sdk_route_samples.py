"""Minimal OpenAPI + Fern override shapes for SDK docs unit tests."""

SAMPLE_OVERRIDES: dict = {
    "paths": {
        "/agents": {
            "get": {
                "x-fern-sdk-group-name": "agents",
                "x-fern-sdk-method-name": "list",
            },
            "post": {
                "x-fern-sdk-group-name": "agents",
                "x-fern-sdk-method-name": "create",
            },
        },
        "/agent-tests/run": {
            "post": {
                "x-fern-sdk-group-name": "agent_tests",
                "x-fern-sdk-method-name": "run_batch",
            },
        },
        "/agents/{agent_uuid}": {
            "put": {
                "x-fern-sdk-group-name": "agents",
                "x-fern-sdk-method-name": "update",
            },
        },
    },
}

SAMPLE_OPENAPI: dict = {
    "paths": {
        "/agents": {
            "get": {"tags": ["agents"], "summary": "List agents"},
            "post": {"tags": ["agents"], "summary": "Create agent"},
        },
        "/agent-tests/run": {
            "post": {
                "tags": ["agent-tests"],
                "summary": "Run agent tests in batch",
            },
        },
        "/agents/{agent_uuid}": {
            "put": {"tags": ["agents"], "summary": "Update agent"},
        },
    },
}
