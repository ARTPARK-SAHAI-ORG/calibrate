import asyncio
import argparse
from typing import List, Optional
from loguru import logger
from dotenv import load_dotenv
import os
from os.path import join, exists, splitext, basename
import json
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
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from metrics import test_response_llm_judge

load_dotenv(".env", override=True)


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
    tools: List[dict[str, str]] = [],
    model: str = "gpt-4.1",
) -> List[str]:
    """Runs a text-only bot that processes text inputs through an LLM and returns text outputs."""
    # Create LLM service
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
    standard_tools = [end_call_tool]

    for tool in tools:
        properties = {}
        required = []
        for parameter in tool["parameters"]:
            # if parameter["required"]:
            # required.append(parameter["id"])

            prop = {
                "type": parameter["type"],
                "description": parameter["description"],
            }

            if "items" in parameter:
                prop["items"] = parameter["items"]

            if "enum" in parameter:
                prop["enum"] = parameter["enum"]

            properties[parameter["id"]] = prop

        tool_function = FunctionSchema(
            name=tool["name"],
            description=tool["description"],
            properties=properties,
            required=[],
        )
        standard_tools.append(tool_function)

    tools = ToolsSchema(standard_tools=standard_tools)

    async def tool_call(params: FunctionCallParams):
        logger.info(f"tool call: {params}")
        await params.result_callback(
            None, properties=FunctionCallResultProperties(run_llm=False)
        )
        return

    for tool in standard_tools:
        llm.register_function(tool.name, tool_call)

    context = LLMContext(messages, tools=tools)
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


async def run_test(
    chat_history: List[dict[str, str]],
    evaluation: dict[str, str],
    system_prompt: str,
    tools: List[dict[str, str]] = [],
    model: str = "gpt-4.1",
):
    output = await run_inference(
        chat_history=chat_history,
        system_prompt=system_prompt,
        tools=tools,
        model=model,
    )
    metrics = {"passed": True}
    if evaluation["type"] == "tool_call":
        if output["tool_calls"] != evaluation["tool_calls"]:
            metrics["passed"] = False
    elif evaluation["type"] == "response":
        result = await test_response_llm_judge(
            conversation=chat_history,
            response=output["response"],
            criteria=evaluation["criteria"],
        )
        metrics["passed"] = result["match"]
        metrics["reasoning"] = result["reasoning"]
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
        "-d",
        "--debug",
        action="store_true",
        help="Run the evaluation on the first 5 audio files",
    )

    args = parser.parse_args()

    config = json.load(open(args.config))

    config_name = splitext(basename(args.config))[0]

    output_dir = join(args.output_dir, config_name, args.model)

    if not exists(output_dir):
        os.makedirs(output_dir)

    log_save_path = join(output_dir, "logs")

    if exists(log_save_path):
        os.remove(log_save_path)

    logger.remove()
    logger.add(log_save_path)

    results = []
    for test_case_index, test_case in enumerate(config["test_cases"]):
        result = await run_test(
            chat_history=test_case["history"],
            evaluation=test_case["evaluation"],
            system_prompt=config["system_prompt"],
            tools=config["tools"],
            model=args.model,
        )

        if result["metrics"]["passed"]:
            print(f"✅ Test case {test_case_index + 1} passed")
        else:
            print(f"❌ Test case {test_case_index + 1} failed")
            if "reasoning" in result["metrics"]:
                print(result["metrics"]["reasoning"])

        result["test_case"] = test_case
        results.append(result)

        print("-" * 40)

    total_passed = sum(1 for result in results if result["metrics"]["passed"])
    total_tests = len(results)

    passed_count = total_passed
    failed_count = total_tests - total_passed

    if passed_count == total_tests:
        print("🎉 All tests passed!")
    elif failed_count == total_tests:
        print("❌ All tests failed!")
    else:
        print(
            f"✅ Total Passed: {passed_count}/{total_tests} ({(passed_count/total_tests)*100:.1f}%)"
        )
        print(
            f"❌ Total Failed: {failed_count}/{total_tests} ({(failed_count/total_tests)*100:.1f}%)"
        )

    output = {
        "total": total_tests,
        "passed": passed_count,
    }

    with open(join(output_dir, "results.json"), "w") as f:
        json.dump(output, f)

    if failed_count:
        print("Failed test cases:")
        print("=" * 40)
        for result in results:
            if result["metrics"]["passed"]:
                continue

            print("History:\n\n")
            print(
                "\n".join(
                    [
                        f"{message['role']}: {message['content']}"
                        for message in result["test_case"]["history"]
                        if "content" in message
                    ]
                )
            )
            print("\n\nExpected output:")
            print(result["test_case"]["evaluation"])
            print("-" * 40)
            print("Output:")
            print(result["output"])
            print("-" * 40)
            print("Metrics:")
            print(result["metrics"])
            print("-" * 40)
            print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
