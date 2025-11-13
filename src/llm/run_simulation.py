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
    LLMMessagesAppendFrame,
    TextFrame,
)
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


class ConversationState:
    """Tracks conversation turns and coordinates termination across pipelines."""

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.turn_count = 0
        self.finished = False
        self._lock = asyncio.Lock()
        self._finish_notified = False

    async def record_turn(self) -> bool:
        """Register a completed message turn.

        Returns:
            True if the conversation should continue, False if it reached the limit.
        """
        async with self._lock:
            if self.finished:
                return False

            self.turn_count += 1

            if self.turn_count >= self.max_turns:
                self.finished = True
                return False

            return True

    async def mark_finished(self) -> bool:
        """Mark the conversation as finished and ensure it's done only once.

        Returns:
            True if this call marked the conversation finished, False otherwise.
        """
        async with self._lock:
            if self._finish_notified:
                return False

            self._finish_notified = True
            self.finished = True
            return True


class Processor(FrameProcessor):
    """Processor that captures LLM text output."""

    def __init__(
        self,
        speaks_first: bool,
        *,
        conversation_state: "ConversationState",
        name: str = "Processor",
    ):
        super().__init__(enable_direct_mode=True, name=name)
        self._current_response = ""
        self._collecting_response = False
        self._ready = False
        self._speaks_first = speaks_first
        self._conversation_state = conversation_state
        self._partner_task: Optional["PipelineTask"] = None
        self._self_end_sent = False

    def set_task(self, task: "PipelineTask"):
        """Set the task reference after task creation."""
        self._task = task

    def set_partner(self, task: "PipelineTask"):
        """Set the partner task to exchange messages with."""
        self._partner_task = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        logger.info(f"text output processor frame: {frame}")

        if not self._ready:
            self._ready = True
            if self._task and self._speaks_first:
                await self._task.queue_frames(
                    [
                        LLMRunFrame(),
                    ]
                )

        # Capture text frames from LLM
        if isinstance(frame, TextFrame):
            text = frame.text
            if text:
                self._collecting_response = True
                self._current_response += text
                logger.info(f"Received text chunk: {text}")

        # When we get an EndFrame after collecting text, save the complete response
        if isinstance(frame, LLMFullResponseEndFrame):
            response = self._current_response.strip()
            self._current_response = ""
            self._collecting_response = False

            if response:
                await self._handle_completed_response(response)
            elif self._conversation_state and self._conversation_state.finished:
                await self._end_conversation()

        await self.push_frame(frame, direction)

    async def _handle_completed_response(self, response: str):
        should_continue = True

        if self._conversation_state:
            should_continue = await self._conversation_state.record_turn()

        await self._forward_to_partner(response, run_partner=should_continue)

        if self._conversation_state and not should_continue:
            await self._end_conversation()

    async def _forward_to_partner(self, response: str, *, run_partner: bool):
        if not self._partner_task or not response:
            return

        frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": response}],
            run_llm=run_partner,
        )

        await self._partner_task.queue_frames([frame])

    async def _end_conversation(self):
        notify_partner = False

        if self._conversation_state:
            notify_partner = await self._conversation_state.mark_finished()

        if notify_partner and self._partner_task:
            await self._partner_task.queue_frames([EndFrame()])

        if self._task and not self._self_end_sent:
            self._self_end_sent = True
            await self._task.queue_frames([EndFrame()])


async def run_simulation(
    bot_system_prompt: str,
    user_system_prompt: str,
    bot_model: str = "gpt-4.1",
    user_model: str = "gpt-4.1",
    bot_speaks_first: bool = True,
    max_turns: int = 10,
) -> List[str]:
    """Runs a text-only bot that processes text inputs through an LLM and returns text outputs."""
    # Create LLM service

    bot_llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=bot_model,
    )

    user_llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=user_model,
    )

    # Create context with system prompt
    messages = [{"role": "system", "content": bot_system_prompt}]
    bot_context = LLMContext(messages)
    bot_context_aggregator = LLMContextAggregatorPair(bot_context)

    messages = [{"role": "system", "content": user_system_prompt}]
    user_context = LLMContext(messages)
    user_context_aggregator = LLMContextAggregatorPair(user_context)

    conversation_state = ConversationState(max_turns=max_turns)

    # Create processors (text_output needs to be created first for reference)
    bot_processor = Processor(
        speaks_first=bot_speaks_first,
        conversation_state=conversation_state,
        name="BotProcessor",
    )
    user_processor = Processor(
        speaks_first=not bot_speaks_first,
        conversation_state=conversation_state,
        name="UserProcessor",
    )

    # Build pipeline with all processors
    bot_pipeline = Pipeline(
        [
            # text_input,
            bot_context_aggregator.user(),
            bot_llm,
            bot_processor,
            bot_context_aggregator.assistant(),
        ]
    )

    user_pipeline = Pipeline(
        [
            # text_input,
            user_context_aggregator.user(),
            user_llm,
            user_processor,
            user_context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        bot_pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LLMLogObserver()],
    )

    user_task = PipelineTask(
        user_pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LLMLogObserver()],
    )

    # Set task reference for text_input processor
    bot_processor.set_task(task)
    user_processor.set_task(user_task)

    bot_processor.set_partner(user_task)
    user_processor.set_partner(task)

    runner = PipelineRunner(handle_sigint=False)
    user_runner = PipelineRunner(handle_sigint=False)

    try:
        await asyncio.gather(
            runner.run(task),
            user_runner.run(user_task),
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    return bot_context.get_messages()


async def test_text_bot():
    """Simple test function to demonstrate the text bot."""

    bot_system_prompt = (
        "You are a helpful and friendly AI assistant. Keep your responses concise."
    )

    user_system_prompt = "You are a mother. This is your persona:\n\n{user_persona}\n\nYou are speaking to an agent acting as a nurse. The agent will ask you questions and you need to answer them based on your persona. The following scenario will be played out: {simulation_scenario}. Make sure to respond to the agent to match the given scenario."

    model = "gpt-4.1"

    simulation_configs = [
        {
            "user_persona": "Name: R. Lakshmi, Address: Flat 302, Sri Venkateshwara Nilaya, near Hanuman Temple, Indiranagar; Phone number: +919843322940. Never a stillbirth or a baby who died soon after birth; one baby had died on the third day after delivery; never had three or more miscarriages in a row; in her last pregnancy, wasn’t admitted for high blood pressure or eclampsia; carrying only one baby as per the scan; age: 39 years old; blood group is O positive, so there are no Rh-related issues; experienced light vaginal spotting once during this pregnancy but otherwise no pelvic mass or complications; uncertain about her exact blood-pressure reading, but recalls that it was around 150/95 mmHg at booking; does not have diabetes, heart disease, kidney disease, or epilepsy; does does have asthma and uses an inhaler daily; never had tuberculosis or any other serious medical condition; she no longer drinks alcohol or uses substances, having stopped completely after learning about pregnancy; very shy, reserved and uses short answers to questions",
            "simulation_scenario": "the mother hesitates in directly answering some questions and wants to skip answering them at first but answers later on further probing",
        }
    ]

    logger.info("Starting simulations...")

    outputs = []

    for simulation_config in simulation_configs:
        logger.info(f"Simulation config: {simulation_config}")

        output = await run_simulation(
            bot_system_prompt=bot_system_prompt,
            user_system_prompt=user_system_prompt.format(
                user_persona=simulation_config["user_persona"],
                simulation_scenario=simulation_config["simulation_scenario"],
            ),
            bot_model=model,
            user_model=model,
        )

        outputs.append(output)

    logger.info("\n" + "=" * 50)
    logger.info("RESULTS:")
    logger.info("=" * 50)

    for i, (simulation_config, conversation) in enumerate(
        zip(simulation_configs, outputs), 1
    ):
        logger.info(f"\nSimulation {i}:")
        logger.info(f"  User persona:  {simulation_config['user_persona']}")
        logger.info(
            f"  Simulation scenario: {simulation_config['simulation_scenario']}"
        )
        logger.info(f"  Conversation:\n\n{conversation}")

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
        outputs = await run_simulation(
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
