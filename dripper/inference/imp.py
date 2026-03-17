from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, pipeline
else:
    from dripper.utils.lazy_import import lazy_from

    AutoModelForCausalLM = lazy_from('transformers', 'AutoModelForCausalLM')
    pipeline = lazy_from('transformers', 'pipeline')


@dataclass
class ModelResponse:
    generated_text: str
    generated_token_ids: list[int] | None = None
    prompt: str | None = None
    prompt_token_ids: list[int] | None = None


class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        pass


class AsyncInferenceBackend(ABC):
    """Abstract base class for async inference backends."""

    @abstractmethod
    async def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        pass


try:
    from vllm import LLM, SamplingParams

    class VLLMInferenceBackend(InferenceBackend):
        def __init__(
            self, model_path: str, model_init_kwargs: dict[str, Any] = {}, model_gen_kwargs: dict[str, Any] = {}
        ):
            tensor_parallel_size = model_init_kwargs.pop('tensor_parallel_size', 1)
            self.sample_paras = SamplingParams(
                top_k=model_gen_kwargs.pop('top_k', 1),
                top_p=model_gen_kwargs.pop('top_p', 0.95),
                temperature=model_gen_kwargs.pop('temperature', 0),
                max_tokens=model_gen_kwargs.pop('max_tokens', 8 * 1024),
            )
            self.gen_config = model_gen_kwargs

            self._llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, **model_init_kwargs)

        @override
        def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
            model_output = self._llm.generate(prompt_list, sampling_params=self.sample_paras, **self.gen_config)
            response = []
            for res in model_output:
                response.append(
                    ModelResponse(
                        prompt=res.prompt,
                        prompt_token_ids=res.prompt_token_ids,
                        generated_text=res.outputs[0].text,
                        generated_token_ids=res.outputs[0].token_ids,
                    )
                )
            return response
except Exception:
    # Dummy implementation for import safety when vLLM is not installed
    class VLLMInferenceBackend(InferenceBackend):
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                'vLLM is not installed. Please install it (e.g., `pip install vllm`) to use VLLMInferenceBackend.'
            )

        @override
        def generate(self, prompt_list: list[str]) -> list['ModelResponse']:
            raise ImportError('vLLM is not installed. Cannot generate responses.')


class TransformersInferenceBackend(InferenceBackend):
    def __init__(
        self,
        model_path: str,
        tokenizer: Any,
        model_init_kwargs: dict[str, Any] = {},
        model_gen_kwargs: dict[str, Any] = {},
    ):
        device_map = model_init_kwargs.pop('device_map', 'auto')
        dtype = model_init_kwargs.pop('dtype', 'auto')
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map=device_map, dtype=dtype, **model_init_kwargs
        )
        self._llm = pipeline('text-generation', model=model, tokenizer=tokenizer, device_map=device_map)

        self.gen_config = {
            'top_k': model_gen_kwargs.pop('top_k', 1),
            'top_p': model_gen_kwargs.pop('top_p', 0.95),
            'temperature': model_gen_kwargs.pop('temperature', 0),
            'max_new_tokens': model_gen_kwargs.pop('max_new_tokens', 8 * 1024),
            'do_sample': model_gen_kwargs.pop('do_sample', False),
            'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'return_full_text': model_gen_kwargs.pop('return_full_text', False),
            **model_gen_kwargs,
        }

    @override
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        model_output = self._llm(prompt_list, **self.gen_config)
        response = []
        for res in model_output:
            response.append(ModelResponse(generated_text=res[0]['generated_text']))
        return response


class SyncVLLMInferenceBackend(InferenceBackend):
    """Synchronous inference backend using OpenAI-compatible API for remote vLLM server.

    This backend connects to a remote vLLM server via its OpenAI-compatible API
    and uses synchronous HTTP requests for inference.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        model_gen_kwargs: dict[str, Any] = {},
        tokenizer_url: str | None = None,
        tokenizer_api_key: str | None = None,
    ):
        """Initialize SyncVLLMInferenceBackend.

        Args:
            api_base: Base URL of the vLLM OpenAI API (e.g., "http://localhost:8000")
            model_name: Name of the model served by vLLM
            model_gen_kwargs: Additional generation parameters
            tokenizer_url: Base URL for tokenization API (defaults to api_base if not provided)
            tokenizer_api_key: API key for tokenization (defaults to api_key if not provided)
        """
        # Store original base URL for tokenize endpoint
        self._api_base = api_base.rstrip('/')

        # Tokenizer URL and API key - can be different from inference
        self._tokenizer_url = tokenizer_url.rstrip('/') if tokenizer_url else self._api_base
        self._tokenizer_api_key = tokenizer_api_key if tokenizer_api_key is not None else api_key

        # Ensure base_url ends with /v1 for OpenAI-compatible API
        api_base_v1 = api_base.rstrip('/') + '/v1'
        self.client = OpenAI(api_key=api_key, base_url=api_base_v1)
        self.model_name = model_name
        # Note: OpenAI Completions API doesn't support top_k
        # Only max_tokens, temperature, top_p are supported
        self.gen_kwargs = {
            'max_tokens': model_gen_kwargs.pop('max_tokens', 8 * 1024),
            'temperature': model_gen_kwargs.pop('temperature', 0),
            'top_p': model_gen_kwargs.pop('top_p', 0.95),
        }

    @override
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        """Generate responses for multiple prompts synchronously.

        Args:
            prompt_list: List of prompt strings

        Returns:
            List of ModelResponse objects
        """
        response = []
        for prompt in prompt_list:
            try:
                result = self.client.completions.create(model=self.model_name, prompt=prompt, **self.gen_kwargs)
                response.append(
                    ModelResponse(
                        prompt=prompt,
                        generated_text=result.choices[0].text,
                    )
                )
            except Exception as e:
                raise RuntimeError(f"Sync inference failed: {str(e)}") from e
        return response

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text using the remote vLLM server's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        import httpx

        try:
            # Use httpx directly since /tokenize is not part of OpenAI API
            # Use tokenizer_url (without /v1) for vLLM native endpoints
            headers = {}
            if self._tokenizer_api_key:
                headers['Authorization'] = f"Bearer {self._tokenizer_api_key}"
            response = httpx.post(
                f"{self._tokenizer_url}/tokenize",
                json={
                    'model': self.model_name,
                    'prompt': text,
                },
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get('prompt_token_ids', [])
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {str(e)}") from e


class AsyncVLLMInferenceBackend(AsyncInferenceBackend):
    """Async inference backend using OpenAI-compatible API for remote vLLM server.

    This backend connects to a remote vLLM server via its OpenAI-compatible API
    and uses async HTTP requests for inference, allowing better concurrency
    when serving multiple requests.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        model_gen_kwargs: dict[str, Any] = {},
        tokenizer_url: str | None = None,
        tokenizer_api_key: str | None = None,
    ):
        """Initialize AsyncVLLMInferenceBackend.

        Args:
            api_base: Base URL of the vLLM OpenAI API (e.g., "http://localhost:8000")
            model_name: Name of the model served by vLLM
            model_gen_kwargs: Additional generation parameters
            tokenizer_url: Base URL for tokenization API (defaults to api_base if not provided)
            tokenizer_api_key: API key for tokenization (defaults to api_key if not provided)
        """
        # Store original base URL for tokenize endpoint
        self._api_base = api_base.rstrip('/')

        # Tokenizer URL and API key - can be different from inference
        self._tokenizer_url = tokenizer_url.rstrip('/') if tokenizer_url else self._api_base
        self._tokenizer_api_key = tokenizer_api_key if tokenizer_api_key is not None else api_key

        # Ensure base_url ends with /v1 for OpenAI-compatible API
        api_base_v1 = api_base.rstrip('/') + '/v1'
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base_v1)
        self.model_name = model_name
        # Note: OpenAI Completions API doesn't support top_k
        # Only max_tokens, temperature, top_p are supported
        self.gen_kwargs = {
            'max_tokens': model_gen_kwargs.pop('max_tokens', 8 * 1024),
            'temperature': model_gen_kwargs.pop('temperature', 0),
            'top_p': model_gen_kwargs.pop('top_p', 0.95),
        }

    async def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        """Generate responses for multiple prompts asynchronously.

        Args:
            prompt_list: List of prompt strings

        Returns:
            List of ModelResponse objects
        """
        import asyncio

        # Create async tasks for all prompts
        tasks = [self._generate_single(prompt) for prompt in prompt_list]
        return await asyncio.gather(*tasks)

    async def _generate_single(self, prompt: str) -> ModelResponse:
        """Generate response for a single prompt."""
        try:
            response = await self.client.completions.create(model=self.model_name, prompt=prompt, **self.gen_kwargs)
            return ModelResponse(
                prompt=prompt,
                generated_text=response.choices[0].text,
            )
        except Exception as e:
            raise RuntimeError(f"Async inference failed: {str(e)}") from e

    async def tokenize(self, text: str) -> list[int]:
        """Tokenize text using the remote vLLM server's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        import httpx

        try:
            # Use httpx directly since /tokenize is not part of OpenAI API
            # Use tokenizer_url (without /v1) for vLLM native endpoints
            headers = {}
            if self._tokenizer_api_key:
                headers['Authorization'] = f"Bearer {self._tokenizer_api_key}"
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._tokenizer_url}/tokenize",
                    json={
                        'model': self.model_name,
                        'prompt': text,
                    },
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                return data.get('prompt_token_ids', [])
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {str(e)}") from e

    def tokenize_sync(self, text: str) -> list[int]:
        """Synchronously tokenize text using the remote vLLM server's tokenizer.

        This is a sync wrapper around the async tokenize method.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(self.tokenize(text))

        # There's a running loop, use run_in_executor to avoid blocking
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self.tokenize(text))
            return future.result()
