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
"""Ephemeral workers backed by Orble boxes.

Orble serves the E2B API, so the backend uses the e2b SDK with Orble's
domain. An Orble box is a full VM with Docker, so the worker image runs in a
container inside it.
"""

import shlex

from e2b import AsyncSandbox

from kitaru.server.application.models.worker import EphemeralWorkerSpec
from kitaru.server.ephemeral_worker_settings import EphemeralWorkerSettings


class OrbleEphemeralWorkers:
    """Ephemeral workers starting one Orble box per worker."""

    def __init__(self, settings: EphemeralWorkerSettings) -> None:
        """Initialize the backend.

        Args:
            settings: Ephemeral worker settings.
        """
        assert settings.orble is not None
        self._orble_settings = settings.orble
        self._image = settings.get_image()
        self._command = shlex.split(settings.command)
        self._timeout_seconds = settings.timeout_seconds
        self._env = {
            key: value.get_secret_value() for key, value in settings.env.items()
        }

    async def start(self, spec: EphemeralWorkerSpec) -> None:
        """Start a worker for a job.

        Args:
            spec: Ephemeral worker spec.
        """
        sandbox = await AsyncSandbox.create(
            template=self._orble_settings.template,
            timeout=self._timeout_seconds,
            metadata={**spec.tags, "kitaru/name": spec.name},
            api_key=self._orble_settings.api_key.get_secret_value(),
            domain=self._orble_settings.domain,
        )
        # The contract variables go last so a configured value cannot
        # override them.
        env = {
            **self._env,
            "KITARU_API_URL": spec.server_url,
            "KITARU_API_TOKEN": spec.worker_token.get_secret_value(),
            "KITARU_WORKER_ID": str(spec.worker_id),
            "KITARU_WORKER_TIMEOUT": str(self._timeout_seconds),
        }
        # Docker takes each value from the command's own environment, so no
        # value appears on a command line.
        docker = ["docker", "run", "--rm", "--network", "host"]
        for key in env:
            docker += ["--env", key]
        await sandbox.commands.run(
            shlex.join([*docker, self._image, *self._command]),
            background=True,
            envs=env,
            user="root",
        )
