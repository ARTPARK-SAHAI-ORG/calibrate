import asyncio
import argparse
import sys
from typing import List
from loguru import logger
import os
from os.path import join, exists, splitext, basename
import json

from calibrate.utils import configure_print_logger, log_and_print, build_tools_schema
from pipecat.frames.frames import (
    TranscriptionFrame,
    LLMRunFrame,
    EndFrame,
    EndTaskFrame,
    LLMFullResponseEndFrame,
    CancelFrame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    TextFrame,
    FunctionCallInProgressFrame,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from calibrate.llm.metrics import test_response_llm_judge


class Processor(FrameProcessor):
    """Processor that captures LLM text output."""

    def __init__(
        self,
        chat_history: List[dict[str, str]],
    ):
        super().__init__(enable_direct_mode=True, name="Processor")
        self._current_response = ""
        self._collecting_response = False
        self._tool_calls = []
        self._ready = False
        self._chat_history = chat_history

    def set_task(self, task: "PipelineTask"):
        """Set the task reference after task creation."""
        self._task = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        logger.info(f"text output processor frame: {frame}")

        if not self._ready:
            self._ready = True
            if self._task:
                await self._task.queue_frames(
                    [
                        LLMMessagesAppendFrame(self._chat_history, run_llm=True),
                    ]
                )

        # Capture text frames from LLM
        if isinstance(frame, TextFrame):
            text = frame.text
            if text:
                self._collecting_response = True
                self._current_response += text
                logger.info(f"Received text chunk: {text}")

        if isinstance(frame, FunctionCallInProgressFrame):
            log_and_print(f"Function call in progress: {frame.function_name}")
            log_and_print(f"Arguments: {frame.arguments}")
            self._tool_calls.append(
                {
                    "tool": frame.function_name,
                    "arguments": frame.arguments,
                }
            )

        # When we get an EndFrame after collecting text, save the complete response
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._task:
                await self._task.queue_frames([EndFrame()])

        await self.push_frame(frame, direction)


async def run_inference(
    chat_history: List[dict[str, str]],
    system_prompt: str,
    model: str,
    provider: str,
    tools: List[dict[str, str]],
) -> List[str]:
    """Runs a text-only bot that processes text inputs through an LLM and returns text outputs."""
    # Create LLM service
    if provider == "openrouter":
        llm = OpenRouterLLMService(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=model,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
        )

    # Create context with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    end_call_tool = FunctionSchema(
        name="end_call",
        description="End the current call when the conversation is complete.",
        properties={
            "reason": {
                "type": "string",
                "description": "Optional explanation for why the call should end.",
            }
        },
        required=[],
    )

    # Build tool schemas using common utility
    tool_schemas, webhook_tool_configs = build_tools_schema(tools)
    standard_tools = [end_call_tool] + tool_schemas

    tools_schema = ToolsSchema(standard_tools=standard_tools)

    async def generic_tool_call(params: FunctionCallParams):
        logger.info(f"tool call: {params}")
        await params.result_callback(
            None, properties=FunctionCallResultProperties(run_llm=False)
        )
        return

    def create_webhook_tool_call(webhook_config: dict):
        async def webhook_tool_call(params: FunctionCallParams):
            logger.info(
                f"webhook tool call: {params.function_name}\n"
                f"  method: {webhook_config['method']}\n"
                f"  url: {webhook_config['url']}\n"
                f"  headers: {webhook_config['headers']}\n"
                f"  query: {params.arguments.get('query', {})}\n"
                f"  body: {params.arguments.get('body', {})}"
            )
            await params.result_callback(
                None, properties=FunctionCallResultProperties(run_llm=False)
            )
            return

        return webhook_tool_call

    # Register appropriate handler for each tool
    for tool_schema in standard_tools:
        if tool_schema.name in webhook_tool_configs:
            llm.register_function(
                tool_schema.name,
                create_webhook_tool_call(webhook_tool_configs[tool_schema.name]),
            )
        else:
            llm.register_function(tool_schema.name, generic_tool_call)

    context = LLMContext(messages, tools=tools_schema)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create processors (text_output needs to be created first for reference)
    processor = Processor(chat_history)
    # text_input = TextInputProcessor(text_inputs, text_output)

    # Build pipeline with all processors
    pipeline = Pipeline(
        [
            # text_input,
            context_aggregator.user(),
            llm,
            processor,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LLMLogObserver()],
        # idle_timeout_secs=5,
    )

    # Set task reference for text_input processor
    processor.set_task(task)

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    return {
        "response": processor._current_response,
        "tool_calls": processor._tool_calls,
    }


def sort_tool_calls(tool_calls):
    return sorted(tool_calls, key=lambda val: val["tool"])


def get_webhook_tool_names(tools: List[dict]) -> set:
    """
    Extract names of webhook tools from the tools configuration.

    Args:
        tools: List of tool definition dicts

    Returns:
        Set of webhook tool names
    """
    return {tool["name"] for tool in tools if tool.get("type") == "webhook"}


def preprocess_conversation_history(
    chat_history: List[dict], tools: List[dict]
) -> List[dict]:
    """
    Preprocess conversation history to add tool responses for non-webhook tools.

    For non-webhook tools that have tool calls but no corresponding tool response,
    this function inserts a default tool response with {"status": "received"}.

    For non-webhook tools that already have a tool response, this function raises
    an error since tool responses should not be manually added for non-webhook tools.

    Args:
        chat_history: The conversation history to preprocess
        tools: List of tool definition dicts

    Returns:
        Preprocessed conversation history with tool responses inserted

    Raises:
        ValueError: If a non-webhook tool has a manually added tool response
    """
    webhook_tool_names = get_webhook_tool_names(tools)

    # Build a set of existing tool response IDs
    existing_tool_response_ids = set()
    for message in chat_history:
        if message.get("role") == "tool" and "tool_call_id" in message:
            existing_tool_response_ids.add(message["tool_call_id"])

    # Process conversation history and collect tool calls that need responses
    processed_history = []
    for message in chat_history:
        processed_history.append(message)

        # Check for assistant messages with tool calls
        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("function", {}).get("name")

                # Skip webhook tools - they handle their own responses
                if tool_name in webhook_tool_names:
                    continue

                # Check if this non-webhook tool already has a response
                if tool_call_id in existing_tool_response_ids:
                    raise ValueError(
                        f"Structured output tool '{tool_name}' (tool_call_id: {tool_call_id}) "
                        f"should not have a manually added tool response in the conversation history. "
                    )

                # Insert a default tool response for non-webhook tools
                tool_response = {
                    "role": "tool",
                    "content": '{"status": "received"}',
                    "tool_call_id": tool_call_id,
                }
                processed_history.append(tool_response)
                logger.info(
                    f"Inserted tool response for structured output tool '{tool_name}' "
                    f"(tool_call_id: {tool_call_id})"
                )

    return processed_history


def evaluate_tool_calls(output_tool_calls, evaluation_tool_calls):
    if not output_tool_calls:
        return {"passed": False, "reasoning": "No tool calls were generated by the LLM"}

    output_tool_calls = sort_tool_calls(output_tool_calls)
    evaluation_tool_calls = sort_tool_calls(evaluation_tool_calls)

    for output_tool_call, evaluation_tool_call in zip(
        output_tool_calls, evaluation_tool_calls
    ):
        if output_tool_call["tool"] != evaluation_tool_call["tool"]:
            return {
                "passed": False,
                "reasoning": f"Tool call mismatch - expected tool call: {evaluation_tool_call['tool']} but got: {output_tool_call['tool']}",
            }

        # if the "arguments" key is not present in the evaluation_tool_call, then we don't need to check the arguments
        if "arguments" not in evaluation_tool_call:
            continue

        # if the "arguments" key is present in the evaluation_tool_call, then we need to check the arguments
        if (
            evaluation_tool_call["arguments"] is not None
            and output_tool_call["arguments"] != evaluation_tool_call["arguments"]
        ):
            return {
                "passed": False,
                "reasoning": f"Tool call arguments mismatch - expected arguments: {evaluation_tool_call['arguments']} but got: {output_tool_call['arguments']}",
            }

    return {
        "passed": True,
    }


async def run_test(
    chat_history: List[dict[str, str]],
    evaluation: dict[str, str],
    system_prompt: str,
    model: str,
    provider: str,
    tools: List[dict[str, str]] = [],
):
    output = await run_inference(
        chat_history=chat_history,
        system_prompt=system_prompt,
        model=model,
        provider=provider,
        tools=tools,
    )
    metrics = {"passed": False}

    if evaluation["type"] == "tool_call":
        metrics = evaluate_tool_calls(output["tool_calls"], evaluation["tool_calls"])
    elif evaluation["type"] == "response":
        if output["response"]:
            result = await test_response_llm_judge(
                conversation=chat_history,
                response=output["response"],
                criteria=evaluation["criteria"],
            )
            metrics["passed"] = result["match"]
            metrics["reasoning"] = result["reasoning"]
        else:
            if output["tool_calls"]:
                metrics["reasoning"] = (
                    f"The LLM generated tool calls: {output['tool_calls']}, but no reply was generated"
                )
            else:
                metrics["reasoning"] = "No reply was generated by the LLM"
    else:
        raise ValueError(f"Invalid evaluation type: {evaluation['type']}")

    return {
        "output": output,
        "metrics": metrics,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Text-only Pipecat bot that processes text through an LLM"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="examples/tests.json",
        help="Path to the JSON configuration file for the tests",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model to use for the evaluation",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["openai", "openrouter"],
        default="openrouter",
        help="LLM provider to use (openai or openrouter)",
    )

    args = parser.parse_args()

    print(
        f"\033[91mRunning tests defined in the config at {args.config} for model: {args.model}\033[0m"
    )

    config = json.load(open(args.config))

    config_name = splitext(basename(args.config))[0]

    save_folder_name = (
        f"{args.provider}/{args.model}"
        if args.provider == "openai"
        else f"{args.model}"
    )

    save_folder_name = save_folder_name.replace("/", "__")
    output_dir = join(args.output_dir, config_name, save_folder_name)

    if not exists(output_dir):
        os.makedirs(output_dir)

    log_save_path = join(output_dir, "logs")

    if exists(log_save_path):
        os.remove(log_save_path)

    logger.remove()
    logger.add(log_save_path)

    print_log_save_path = join(output_dir, "results.log")

    if exists(print_log_save_path):
        os.remove(print_log_save_path)

    configure_print_logger(print_log_save_path)

    command = " ".join(sys.argv)
    log_and_print(f"\033[33mRunning command\033[0m: {command}")

    results = []
    results_file_path = join(output_dir, "results.json")
    for test_case_index, test_case in enumerate(config["test_cases"]):
        agent_language = test_case.get("settings", {}).get("language", "english")

        # Preprocess conversation history to add tool responses for non-webhook tools
        preprocessed_history = preprocess_conversation_history(
            test_case["history"], config["tools"]
        )

        result = await run_test(
            chat_history=preprocessed_history,
            evaluation=test_case["evaluation"],
            system_prompt=config["system_prompt"]
            + f"\n\nYou must always speak in {agent_language}.",
            model=args.model,
            provider=args.provider,
            tools=config["tools"],
        )

        if result["metrics"]["passed"]:
            log_and_print(f"✅ Test case {test_case_index + 1} passed")
        else:
            log_and_print(f"❌ Test case {test_case_index + 1} failed")
            if "reasoning" in result["metrics"]:
                log_and_print(result["metrics"]["reasoning"])

        result["test_case"] = test_case
        results.append(result)

        # Save intermediate results after each test case
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)

        log_and_print("-" * 40)

    total_passed = sum(1 for result in results if result["metrics"]["passed"])
    total_tests = len(results)

    passed_count = total_passed
    failed_count = total_tests - total_passed

    if passed_count == total_tests:
        log_and_print("🎉 All tests passed!")
    elif failed_count == total_tests:
        log_and_print("❌ All tests failed!")
    else:
        log_and_print(
            f"✅ Total Passed: {passed_count}/{total_tests} ({(passed_count/total_tests)*100:.1f}%)"
        )
        log_and_print(
            f"❌ Total Failed: {failed_count}/{total_tests} ({(failed_count/total_tests)*100:.1f}%)"
        )

    with open(join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    metrics = {
        "total": total_tests,
        "passed": passed_count,
    }

    with open(join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    if failed_count:
        log_and_print("Failed test cases:")
        log_and_print("=" * 40)
        for result in results:
            if result["metrics"]["passed"]:
                continue

            log_and_print("History:\n\n")
            log_and_print(
                "\n".join(
                    [
                        f"{message['role']}: {message['content']}"
                        for message in result["test_case"]["history"]
                        if "content" in message
                    ]
                )
            )
            log_and_print("\n\nExpected output:")
            log_and_print(result["test_case"]["evaluation"])
            log_and_print("-" * 40)
            log_and_print("Output:")
            log_and_print(result["output"])
            log_and_print("-" * 40)
            log_and_print("Metrics:")
            log_and_print(result["metrics"])
            log_and_print("-" * 40)
            log_and_print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
