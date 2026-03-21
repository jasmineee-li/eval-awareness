import logging
import os
import time

import openai
import requests
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from evals.apis.inference.openai.base import OpenAIModel
from evals.apis.inference.openai.utils import count_tokens, price_per_token
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt
from evals.utils import GPT_CHAT_MODELS

LOGGER = logging.getLogger(__name__)


class OpenAIChatModel(OpenAIModel):
    def _assert_valid_id(self, model_id: str):
        if "ft:" in model_id:
            model_id = model_id.split(":")[1]
        assert model_id in GPT_CHAT_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
            "OpenAI-Organization": self.organization,
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 1"}],
        }
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            print("⚠️ Failed to get dummy response from header—adding defaults")
            # these are the ones I get for GPT3.5 normally
            # TODO remove this hack once OpenAI fixes their API
            response.headers["x-ratelimit-limit-tokens"] = "2000000"
            response.headers["x-ratelimit-limit-requests"] = "10000"
            response.headers["x-ratelimit-remaining-tokens"] = "1999999"
            response.headers["x-ratelimit-remaining-requests"] = "9999"
            response.headers["x-ratelimit-reset-requests"] = "6ms"
            response.headers["x-ratelimit-reset-tokens"] = "0s"
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: Prompt, **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        num_tokens = 0
        for message in prompt.messages:
            num_tokens += 1
            num_tokens += len(message.content) / 4

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER) + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    @staticmethod
    def convert_top_logprobs(data):
        # Initialize the new structure with only top_logprobs
        # Supports both dict (old SDK) and Pydantic object (new SDK) formats
        top_logprobs = []
        content = data.content if hasattr(data, "content") else data["content"]
        if content is None:
            return []

        for item in content:
            top_logprob_dict = {}
            item_top = item.top_logprobs if hasattr(item, "top_logprobs") else item["top_logprobs"]
            for top_logprob in item_top:
                token = top_logprob.token if hasattr(top_logprob, "token") else top_logprob["token"]
                logprob = top_logprob.logprob if hasattr(top_logprob, "logprob") else top_logprob["logprob"]
                top_logprob_dict[token] = logprob

            top_logprobs.append(top_logprob_dict)

        return top_logprobs

    @retry(
        # retry 5 times
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIError, openai.APIConnectionError)
        ),
        wait=wait_fixed(5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **params) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call with {self.organization}")

        if "logprobs" in params:
            params["top_logprobs"] = params["logprobs"]
            params["logprobs"] = True

        base_url = os.environ.get("OPENAI_BASE_URL")
        client = AsyncOpenAI(
            api_key=openai.api_key,
            organization=self.organization,
            base_url=base_url,
        )

        prompt_file = self.create_prompt_history_file(prompt.openai_format(), model_id, self.prompt_history_dir)  # type: ignore
        api_start = time.time()
        try:
            api_response = await client.chat.completions.create(
                messages=prompt.openai_format(), model=model_id, **params
            )
        except Exception as e:
            LOGGER.error(
                f"Error when getting response from OpenAI: {e}\nRetrying once before failing.\nThe prompt was: {prompt.openai_format()}"
            )
            time.sleep(1)
            api_response = await client.chat.completions.create(
                messages=prompt.openai_format(), model=model_id, **params
            )
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = api_response.usage.prompt_tokens * context_token_cost
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + count_tokens(choice.message.content) * completion_token_cost,
                logprobs=self.convert_top_logprobs(choice.logprobs) if choice.logprobs is not None else None,
            )
            for choice in api_response.choices
        ]
        self.add_response_to_prompt_file(prompt_file, responses)
        return responses
