"""LLM inference utilities for structured generation.

This module provides functions to generate structured outputs using vLLM
with optional state machine-based logits processing.
"""

import copy

from dripper.base import (DripperGenerateInput, DripperGenerateOutput,
                          check_and_find_max_item_id)
from dripper.exceptions import DripperTypeError
from dripper.inference.imp import AsyncInferenceBackend, InferenceBackend
from dripper.inference.logits import build_token_state_machine


def generate(
    llm: InferenceBackend,
    input: DripperGenerateInput | list[DripperGenerateInput] | str | list[str],
    use_state_machine: str = 'v1',
) -> list[DripperGenerateOutput]:
    """Generate structured outputs using vLLM with optional state machine.

    Performs batch inference on input data, optionally using a state machine
    to guide token generation for structured JSON output.

    Args:
        llm: LLM instance for inference, should be a specific type in practical cases.
        input: Input data in various formats:
               - Single DripperGenerateInput
               - List of DripperGenerateInput
               - Single HTML string (will be converted to DripperGenerateInput)
               - List of HTML strings
        use_state_machine: State machine version to use ('v1', 'v2', or None/'' to disable)

    Returns:
        List of DripperGenerateOutput objects containing generated responses

    Raises:
        DripperTypeError: If input type is not supported
    """
    base_gen_config = None
    try:
        from vllm import SamplingParams

        # Base generation configuration
        base_gen_config = SamplingParams(top_k=1, top_p=0.95, temperature=0, max_tokens=8 * 1024)
    except Exception:
        pass
    # Normalize input to list of DripperGenerateInput
    if isinstance(input, list):
        input_list = []
        for p in input:
            if isinstance(p, str):
                # Convert string to DripperGenerateInput with identity prompt
                input_list.append(DripperGenerateInput(alg_html=p, prompt=lambda x: x))
            elif isinstance(p, DripperGenerateInput):
                input_list.append(p)
            else:
                raise DripperTypeError(f"Unsupported input type: {type(p)}, {p}")
    elif isinstance(input, str):
        # Convert single string to list
        input_list = [DripperGenerateInput(alg_html=input, prompt=lambda x: x)]
    elif isinstance(input, DripperGenerateInput):
        # Convert single DripperGenerateInput to list
        input_list = [input]
    else:
        raise DripperTypeError(f"Unsupported input type: {type(input)}, {input}")

    # Extract prompts from input data
    prompt_list = [data.full_prompt for data in input_list]

    if use_state_machine and base_gen_config is not None:
        # If use_state_machine is not None/empty AND vLLM is available
        # Set state machine to sampling_params_arg
        state_machines = [
            build_token_state_machine(
                check_and_find_max_item_id(data.alg_html),
                llm.get_tokenizer(),
                version=use_state_machine,
            )
            for data in input_list
        ]
        sampling_params_arg = []
        for state_machine in state_machines:
            # Create a copy of base config and add logits processor
            sampling_params = copy.deepcopy(base_gen_config)
            sampling_params.logits_processors = [state_machine.process_logit]
            sampling_params_arg.append(sampling_params)
    else:
        # If use_state_machine is None/empty, or vLLM not available (e.g., sync_vllm/async_vllm)
        # Just pass prompts directly to the backend - it handles its own generation
        sampling_params_arg = None
    # Perform batch generation
    res_list = llm.generate(prompt_list)

    # Convert results to DripperGenerateOutput objects
    output_list = []
    for input_data, res_data in zip(input_list, res_list):
        case_id = input_data.case_id
        output_list.append(DripperGenerateOutput(case_id=case_id, response=res_data.generated_text))
    return output_list


async def generate_async(
    llm: AsyncInferenceBackend,
    input: DripperGenerateInput | list[DripperGenerateInput] | str | list[str],
    use_state_machine: str = None,
) -> list[DripperGenerateOutput]:
    """Generate structured outputs asynchronously using OpenAI-compatible API.

    Performs async batch inference on input data. Note: State machine is not
    supported for async backend since it requires local tokenizer access.

    Args:
        llm: Async LLM instance for inference
        input: Input data in various formats:
               - Single DripperGenerateInput
               - List of DripperGenerateInput
               - Single HTML string (will be converted to DripperGenerateInput)
               - List of HTML strings
        use_state_machine: State machine version (NOT SUPPORTED for async, will be ignored)

    Returns:
        List of DripperGenerateOutput objects containing generated responses

    Raises:
        DripperTypeError: If input type is not supported
    """
    # Normalize input to list of DripperGenerateInput
    if isinstance(input, list):
        input_list = []
        for p in input:
            if isinstance(p, str):
                input_list.append(DripperGenerateInput(alg_html=p, prompt=lambda x: x))
            elif isinstance(p, DripperGenerateInput):
                input_list.append(p)
            else:
                raise DripperTypeError(f"Unsupported input type: {type(p)}, {p}")
    elif isinstance(input, str):
        input_list = [DripperGenerateInput(alg_html=input, prompt=lambda x: x)]
    elif isinstance(input, DripperGenerateInput):
        input_list = [input]
    else:
        raise DripperTypeError(f"Unsupported input type: {type(input)}, {input}")

    # Extract prompts from input data
    prompt_list = [data.full_prompt for data in input_list]

    # Note: State machine is not supported for async backend
    # because it requires local tokenizer access for logits processing
    if use_state_machine:
        from dripper.utils import logger

        logger.warning(
            'State machine is not supported for async inference backend. Ignoring use_state_machine parameter.'
        )

    # Perform async batch generation
    res_list = await llm.generate(prompt_list)

    # Convert results to DripperGenerateOutput objects
    output_list = []
    for input_data, res_data in zip(input_list, res_list):
        case_id = input_data.case_id
        output_list.append(DripperGenerateOutput(case_id=case_id, response=res_data.generated_text))
    return output_list
