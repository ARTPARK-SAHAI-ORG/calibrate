from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    from langfuse import get_client
    from langfuse.media import LangfuseMedia

    langfuse = get_client()
    langfuse.auth_check()
    from langfuse.openai import AsyncOpenAI
    from langfuse import observe

    langfuse_enabled = True
except Exception:
    from openai import AsyncOpenAI

    LangfuseMedia = None
    langfuse = None
    langfuse_enabled = False

    def observe(**kwargs):
        """No-op decorator when langfuse is not available."""

        def decorator(func):
            return func

        return decorator


def test_langfuse_connection():
    try:
        from langfuse import get_client

        client = get_client()
        client.auth_check()
        print("Langfuse client is authenticated and ready!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def create_langfuse_audio_media(audio_path: str):
    if not langfuse_enabled or LangfuseMedia is None:
        return None

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        logger.warning(f"Failed to read audio file at {audio_path}: {e}")
        return None

    return LangfuseMedia(content_bytes=audio_bytes, content_type="audio/wav")
