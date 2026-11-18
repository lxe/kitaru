#  Copyright (c) ZenML GmbH 2026. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Tests for the Orble ephemeral worker backend."""

import shlex
import uuid
from typing import Any

import pytest

pytest.importorskip("e2b")

from kitaru import images
from kitaru.server.adapters.ephemeral_workers import orble
from kitaru.server.adapters.ephemeral_workers.orble import OrbleEphemeralWorkers
from kitaru.server.application.models.worker import EphemeralWorkerSpec
from kitaru.server.ephemeral_worker_settings import (
    EphemeralWorkerBackend,
    EphemeralWorkerSettings,
    OrbleEphemeralWorkerSettings,
)


class _FakeCommands:
    """Records the commands run in a fake sandbox."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, cmd: str, **kwargs: Any) -> None:
        """Record a command and its keyword arguments."""
        self.calls.append((cmd, kwargs))


class _FakeSandbox:
    """Fake e2b sandbox with a command recorder."""

    def __init__(self) -> None:
        self.commands = _FakeCommands()


class _FakeAsyncSandbox:
    """Fake e2b AsyncSandbox class recording create calls."""

    def __init__(self) -> None:
        self.creates: list[dict[str, Any]] = []
        self.sandbox = _FakeSandbox()

    async def create(self, **kwargs: Any) -> _FakeSandbox:
        """Record a create call and return the fake sandbox."""
        self.creates.append(kwargs)
        return self.sandbox


@pytest.fixture
def fake_e2b(monkeypatch: pytest.MonkeyPatch) -> _FakeAsyncSandbox:
    """Replace the e2b SDK's AsyncSandbox with a fake."""
    fake = _FakeAsyncSandbox()
    monkeypatch.setattr(orble, "AsyncSandbox", fake)
    return fake


def _settings(
    image: str | None = "zenmldocker/kitaru-worker:1.0.0",
    command: str = "python -m kitaru.worker",
    env: dict[str, str] | None = None,
    **orble_settings: Any,
) -> EphemeralWorkerSettings:
    return EphemeralWorkerSettings(
        backend=EphemeralWorkerBackend.ORBLE,
        image=image,
        command=command,
        timeout_seconds=120,
        env=env or {},
        orble=OrbleEphemeralWorkerSettings(api_key="orble-key", **orble_settings),
    )


def _spec() -> EphemeralWorkerSpec:
    job_id = uuid.uuid4()
    return EphemeralWorkerSpec(
        worker_id=uuid.uuid4(),
        name=f"job-{job_id}",
        worker_token="worker-token",
        server_url="https://kitaru.example.com",
        job_id=job_id,
        tags={"kitaru/job_id": str(job_id)},
    )


async def test_start_creates_a_box_and_runs_the_worker_image(
    fake_e2b: _FakeAsyncSandbox,
) -> None:
    """Create a box with the key, domain and timeout, and run the image in it."""
    ephemeral_workers = OrbleEphemeralWorkers(_settings())
    spec = _spec()

    await ephemeral_workers.start(spec)

    assert fake_e2b.creates == [
        {
            "template": "base",
            "timeout": 120,
            "metadata": {**spec.tags, "kitaru/name": spec.name},
            "api_key": "orble-key",
            "domain": "orble.tech",
        }
    ]
    [(cmd, kwargs)] = fake_e2b.sandbox.commands.calls
    assert shlex.split(cmd) == [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--env",
        "KITARU_API_URL",
        "--env",
        "KITARU_API_TOKEN",
        "--env",
        "KITARU_WORKER_ID",
        "--env",
        "KITARU_WORKER_TIMEOUT",
        "zenmldocker/kitaru-worker:1.0.0",
        "python",
        "-m",
        "kitaru.worker",
    ]
    assert kwargs == {
        "background": True,
        "envs": {
            "KITARU_API_URL": spec.server_url,
            "KITARU_API_TOKEN": "worker-token",
            "KITARU_WORKER_ID": str(spec.worker_id),
            "KITARU_WORKER_TIMEOUT": "120",
        },
        "user": "root",
    }


async def test_start_keeps_secrets_off_the_command_line(
    fake_e2b: _FakeAsyncSandbox,
) -> None:
    """Pass the configured env by name, with the contract variables winning."""
    ephemeral_workers = OrbleEphemeralWorkers(
        _settings(
            env={"MY_API_KEY": "secret-value", "KITARU_API_URL": "https://evil.example"}
        )
    )
    spec = _spec()

    await ephemeral_workers.start(spec)

    [(cmd, kwargs)] = fake_e2b.sandbox.commands.calls
    assert "secret-value" not in cmd
    assert "worker-token" not in cmd
    assert shlex.split(cmd).count("MY_API_KEY") == 1
    assert kwargs["envs"] == {
        "MY_API_KEY": "secret-value",
        "KITARU_API_URL": spec.server_url,
        "KITARU_API_TOKEN": "worker-token",
        "KITARU_WORKER_ID": str(spec.worker_id),
        "KITARU_WORKER_TIMEOUT": "120",
    }


async def test_start_uses_the_configured_domain_and_template(
    fake_e2b: _FakeAsyncSandbox,
) -> None:
    """Create the box on the configured domain, in the configured size."""
    ephemeral_workers = OrbleEphemeralWorkers(
        _settings(domain="staging.orble.tech", template="l")
    )

    await ephemeral_workers.start(_spec())

    [create] = fake_e2b.creates
    assert create["domain"] == "staging.orble.tech"
    assert create["template"] == "l"


async def test_start_splits_a_configured_command(fake_e2b: _FakeAsyncSandbox) -> None:
    """Run a configured command after the image, as its shell-split arguments."""
    ephemeral_workers = OrbleEphemeralWorkers(
        _settings(command="/app/.venv/bin/python -m kitaru.worker --log-level debug")
    )

    await ephemeral_workers.start(_spec())

    [(cmd, _)] = fake_e2b.sandbox.commands.calls
    assert shlex.split(cmd)[-6:] == [
        "zenmldocker/kitaru-worker:1.0.0",
        "/app/.venv/bin/python",
        "-m",
        "kitaru.worker",
        "--log-level",
        "debug",
    ]


async def test_start_defaults_to_the_published_worker_image(
    fake_e2b: _FakeAsyncSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the published worker image at the installed version when unset."""
    monkeypatch.setattr(images, "version", lambda name: "0.25.0")
    ephemeral_workers = OrbleEphemeralWorkers(_settings(image=None))

    await ephemeral_workers.start(_spec())

    [(cmd, _)] = fake_e2b.sandbox.commands.calls
    assert "zenmldocker/kitaru-worker:0.25.0" in shlex.split(cmd)
