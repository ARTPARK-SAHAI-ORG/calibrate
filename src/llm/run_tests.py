import asyncio
import argparse
from typing import List, Optional
from loguru import logger
from dotenv import load_dotenv
import os

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


async def run_text_bot(
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


async def test_text_bot():
    system_prompt = """You are a helpful nurse speaking to a pregnant mother who has come for an Antenatal visit (ANC visit).

    You are helping her with filling the ANC visit form which has the following questions:

    1: Name of patient (string)
    2: Address (string)
    3: Telephone (number)
    4: Age (number)
    5: Previous stillbirth or neonatal loss?
    6: History of 3 or more consecutive spontaneous abortions
    7: Birth weight of last baby (string)
    8: Last pregnancy: hospital admission for hypertension or pre-eclampsia/eclampsia?
    9: Previous surgery on reproductive tract (Caesarean section, myomectomy, cone biopsy, cervical cerclage,)
    10: Diagnosed or suspected multiple pregnancy
    11: Isoimmunisation Rh (-) in current or previous pregnancy
    12: Vaginal bleeding
    13: Pelvic mass
    14: Diastolic blood pressure 90 mmHg or more at booking
    15: Diabetes mellitus on insulin or oral hypoglycaemic treatment
    16: Cardiac disease
    17: Renal disease
    18: Epilepsy
    19: Asthmatic on medication
    20: Tuberculosis
    21: Known ‘substance’ abuse (including heavy alcohol drinking)
    22: Any other severe medical disease or condition (string)

    You always speak in {language}.

    Your goal is to get the answer for all these questions from the user. Ask the questions sequentially, one at a time. Make sure to always speak out the next question for the user. They don't know what the next question will be. It is your responsibility to ask them the next question.
    
    Do not repeat the user's answer back to them.
    
    Except for the questions where a type has been explicitly given beside it, the rest of the questions are boolean questions (yes/no).

    If the user gives an answer that is not valid for a question, then, don't call the `plan_next_question` tool at all.

    Only if the user gives a valid answer to a question, then, call the `plan_next_question` tool.

    Once all the questions have been answered, end the call by calling the `end_call` tool immediately without asking or saying anything else to the user.

    # Important instructions
    For each field, you must make sure that you get valid and complete answers from the user.

    Name: the full name including both the first name and last name (users might have initials as part of their name - don't ask them to expand it)
    Address: the full address of the user's specific place of residence
    Telephone: must be a valid 10-digit phone number (don't bug the user to say the number in 10 digits or without spaces or without hyphens or to repeat it as long you can infer the number from your message, irrespective of whatever format it may have been given in - asking for repeating information that is already clear is not a good user experience)
    Age: must be a valid age value
    Birth weight of last baby: must be a valid weight value with the appropriate units (except if this is their first baby, in which case this should be null

    Until you get a valid and complete response for a specific question from the user, keep probing for more details until you have extracted all the details required to meet the criteria of completeness for that question.

    For the boolean questions, just look for whether the user has that symptom or not. If the user is not able to give a definitive answer to any the boolean questions, try probing once to help them arrive at a definitive answer. If they are still not able to give a definitive true/false answer, mark that response as null.

    # Skipping already answered questions
    It is possible that even though you have asked a particular question, the user's response may end up answering some of the future questions you were going to ask. If the user's response already definitively answers those questions, then, don't repeat those questions again and skip them to move to the next unanswered question. 

    After every valid and complete response given by the user to a question, call the `plan_next_question` tool with the indices of the questions in the list of questions above that the user's response has already answered and the index of the next unanswered question that you are going to ask.

    If the user's response only answers the exact question you asked, then, call `plan_next_question` with `questions_answered` as `[<index_of_that_question>]` and `next_unanswered_question_index` as the index of the next unanswered question based on the conversation history so far.   

    If the user's response answers more questions beyond the question you asked, then, call `plan_next_question` with `questions_answered` as `[<index_of_the_first_question_answered>, <index_of_the_second_question_answered>, ....]` and `next_unanswered_question_index` as the index of the next unanswered question based on the conversation history so far.   

    If the user's response answers some other question from the form but not the question you asked, then, call `plan_next_question` with `questions_answered` as `[<index_of_the_question_answered>]` and `next_unanswered_question_index` as the index of the question you had originally asked but the user did not answer.

    If the user's response is completely irrelevant to any of the questions in the form, don't call this tool and steer the user back to the conversation.

    Use the same index values as mentioned in the list above: from 1-22. So, 1-indexed. Not, 0-indexed.
    
    # Ensuring completion of all questions
    - If the user insists on skipping any question when you asked it, make sure to ask it again later. Before ending the call, make sure to ask all the questions that you asked. Only when all the questions have been asked and answered, then, call the `end_call` tool.

    # Style
    - Keep the conversation clear, warm, friendly and empathetic. It should feel like a natural and realistic conversation between a nurse and a mother who is pregnant. Make sure each question that you ask is concise and to the point.
    - Speak in an empathetic, friendly tone like a nurse at a government hospital. 
    - Always stay gender neutral and don't say anything that might indicate any gender or name for you or anthropomorphise you in any way."""

    system_prompt = system_prompt.format(language="english")

    tools = [
        {
            "type": "client",
            "name": "plan_next_question",
            "description": "Optional, only call this tool if the user gave a valid answer to a valid question that you asked; if yes, then which questions have been answered and which question needs to be asked next; if not, don't call this tool at all",
            "disable_interruptions": False,
            "force_pre_tool_speech": "auto",
            "assignments": [],
            "tool_call_sound": None,
            "tool_call_sound_behavior": "auto",
            "execution_mode": "immediate",
            "expects_response": False,
            "response_timeout_secs": 1,
            "parameters": [
                {
                    "id": "next_unanswered_question_index",
                    "type": "integer",
                    "value_type": "llm_prompt",
                    "description": "the index of the next unanswered question that should be asked to the user next. don't repeat questions that have been asked before already in the previous responses.",
                    "dynamic_variable": "",
                    "constant_value": "",
                    # "enum": None,
                    "required": True,
                },
                {
                    "id": "questions_answered",
                    "type": "array",
                    "description": "Optional, the indices of the questions in the full list of questions that have been answered by the user's last response",
                    "items": {
                        "type": "integer",
                        "description": "the index of each question that got answered by the current response of the user",
                        # "constant_value": "",
                        # "enum": None,
                        # "required": False,
                    },
                    "required": True,
                    "value_type": "llm_prompt",
                },
            ],
            "dynamic_variables": {"dynamic_variable_placeholders": {}},
        }
    ]

    model = "gpt-4.1"

    test_cases = [
        {
            "history": [
                {
                    "role": "assistant",
                    "content": "Hello! Let's fill out your ANC form. What is your name?",
                },
                {
                    "role": "user",
                    "content": "Aman Dalmia",
                },
            ],
        }
    ]

    logger.info("Starting text bot test...")

    outputs = []

    for test_case in test_cases:
        logger.info(f"Input: {test_case['history']}")

        output = await run_text_bot(
            chat_history=test_case["history"],
            system_prompt=system_prompt,
            tools=tools,
            model=model,
        )

        outputs.append(output)

    logger.info("\n" + "=" * 50)
    logger.info("RESULTS:")
    logger.info("=" * 50)

    for i, (input_text, output_text) in enumerate(zip(test_cases, outputs), 1):
        logger.info(f"\nTurn {i}:")
        logger.info(f"  Input:  {input_text}")
        logger.info(
            f"  Output:\n\nResponse: {output_text['response']}\nTool Calls: {output_text['tool_calls']}"
        )

    return outputs


async def main():
    parser = argparse.ArgumentParser(
        description="Text-only Pipecat bot that processes text through an LLM"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a simple test with predefined inputs",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Text inputs to process (space-separated)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for the LLM",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )

    args = parser.parse_args()

    if args.test:
        await test_text_bot()
    elif args.input:
        logger.info("Processing text inputs...")
        outputs = await run_text_bot(
            text_inputs=args.input,
            system_prompt=args.system_prompt,
            model=args.model,
        )
        logger.info("\n" + "=" * 50)
        logger.info("RESULTS:")
        logger.info("=" * 50)
        for i, (input_text, output_text) in enumerate(zip(args.input, outputs), 1):
            logger.info(f"\nTurn {i}:")
            logger.info(f"  Input:  {input_text}")
            logger.info(f"  Output: {output_text}")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
