import logging
import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt

LOGGER = logging.getLogger(__name__)


class VLLMModel(InferenceAPIModel):
    """Client for a local vLLM OpenAI-compatible server."""

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=300.0)

    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        response = await self._make_api_call(prompt, model_ids[0], time.time(), **kwargs)
        if print_prompt_and_response:
            print(prompt)
            print(response[0].completion)
        return response

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadError, httpx.HTTPStatusError)),
        wait=wait_fixed(5),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    async def _make_api_call(self, prompt: Prompt, model_id: str, start_time: float, **params) -> list[LLMResponse]:
        port = params.get("vllm_port", 8000)
        url = f"http://localhost:{port}/v1/chat/completions"

        new_params = {k: v for k, v in params.items() if k not in ("seed", "cais_path", "logprobs", "vllm_port")}

        # vLLM rejects top_p=0.0; clamp to a small value for near-greedy sampling
        if "top_p" in new_params and new_params["top_p"] is not None and new_params["top_p"] <= 0.0:
            new_params["top_p"] = 0.01

        body = {
            "model": model_id,
            "messages": prompt.openai_format(),
            "stream": False,
            **new_params,
        }

        api_start = time.time()
        resp = await self._client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        api_duration = time.time() - api_start
        duration = time.time() - start_time

        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice["message"]["content"],
                stop_reason=choice.get("finish_reason", "unknown"),
                api_duration=api_duration,
                duration=duration,
                cost=0,
                logprobs=None,
            )
            for choice in data["choices"]
        ]
        assert len(responses) >= 1, f"No choices returned from vLLM: {data}"
        return responses
