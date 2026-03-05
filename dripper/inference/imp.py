from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing import override

from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, pipeline
from vllm import LLM, SamplingParams


@dataclass
class ModelResponse:
    generated_text: str
    generated_token_ids: Optional[list[int]] = None
    prompt: Optional[str] = None
    prompt_token_ids: Optional[list[int]] = None


class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        pass


class AsyncInferenceBackend(ABC):
    """Abstract base class for async inference backends."""

    @abstractmethod
    async def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        pass


class VLLMInferenceBackend(InferenceBackend):
    def __init__(self, model_path:str, model_init_kwargs: Dict[str, Any] = {}, model_gen_kwargs:Dict[str, Any] = {}):
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
                    generated_token_ids=res.outputs[0].token_ids
                )
            )
        return response


class TransformersInferenceBackend(InferenceBackend):
    def __init__(self, model_path: str, tokenizer: Any, model_init_kwargs: Dict[str, Any] = {}, model_gen_kwargs: Dict[str, Any] = {}):
        device_map = model_init_kwargs.pop('device_map', 'auto')
        dtype = model_init_kwargs.pop('dtype', 'auto')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device_map,
            dtype=dtype,
            **model_init_kwargs
        )
        self._llm = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device_map=device_map
        )

        self.gen_config = {
            'top_k': model_gen_kwargs.pop('top_k', 1),
            'top_p': model_gen_kwargs.pop('top_p', 0.95),
            'temperature': model_gen_kwargs.pop('temperature', 0),
            'max_new_tokens': model_gen_kwargs.pop('max_new_tokens', 8 * 1024),
            'do_sample': model_gen_kwargs.pop('do_sample', False),
            'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'return_full_text': model_gen_kwargs.pop('return_full_text', False),
            **model_gen_kwargs
        }

    @override
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        model_output = self._llm(prompt_list,**self.gen_config)
        response = []
        for res in model_output:
            response.append(
                ModelResponse(
                    generated_text=res[0]['generated_text']
                )
            )
        return response


class AsyncVLLMInferenceBackend(AsyncInferenceBackend):
    """
    Async inference backend using OpenAI-compatible API for remote vLLM server.

    This backend connects to a remote vLLM server via its OpenAI-compatible API
    and uses async HTTP requests for inference, allowing better concurrency
    when serving multiple requests.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        model_gen_kwargs: Dict[str, Any] = {},
    ):
        """
        Initialize AsyncVLLMInferenceBackend.

        Args:
            api_base: Base URL of the vLLM OpenAI API (e.g., "http://localhost:8000")
            model_name: Name of the model served by vLLM
            model_gen_kwargs: Additional generation parameters
        """
        self.client = AsyncOpenAI(api_key="dummy", base_url=api_base)
        self.model_name = model_name
        self.gen_kwargs = {
            'max_tokens': model_gen_kwargs.pop('max_tokens', 8 * 1024),
            'temperature': model_gen_kwargs.pop('temperature', 0),
            'top_p': model_gen_kwargs.pop('top_p', 0.95),
            'top_k': model_gen_kwargs.pop('top_k', -1),
            **model_gen_kwargs
        }

    async def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        """
        Generate responses for multiple prompts asynchronously.

        Args:
            prompt_list: List of prompt strings

        Returns:
            List of ModelResponse objects
        """
        import asyncio

        # Create async tasks for all prompts
        tasks = [
            self._generate_single(prompt)
            for prompt in prompt_list
        ]
        return await asyncio.gather(*tasks)

    async def _generate_single(self, prompt: str) -> ModelResponse:
        """Generate response for a single prompt."""
        try:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                **self.gen_kwargs
            )
            return ModelResponse(
                prompt=prompt,
                generated_text=response.choices[0].text,
            )
        except Exception as e:
            raise RuntimeError(f"Async inference failed: {str(e)}") from e

    async def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text using the remote vLLM server's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        try:
            response = await self.client.post(
                "/tokenize",
                json={
                    "model": self.model_name,
                    "prompt": text,
                }
            )
            data = response.json()
            return data.get("prompt_token_ids", [])
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {str(e)}") from e

    def tokenize_sync(self, text: str) -> list[int]:
        """
        Synchronously tokenize text using the remote vLLM server's tokenizer.

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
            future = pool.submit(
                asyncio.run, self.tokenize(text)
            )
            return future.result()
