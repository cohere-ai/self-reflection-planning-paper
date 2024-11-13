import backoff
from mistralai import exceptions as mistral_exceptions
from mistralai.client import MistralClient

client = MistralClient(max_retries=0)


@backoff.on_exception(
    backoff.expo,
    (mistral_exceptions.MistralAPIStatusException, mistral_exceptions.MistralConnectionException),
    factor=4,
    max_tries=5,
)
def chat_with_backoff(model, messages, **generation_params):
    """Exponential backoff: (2 ** (current_try - 1)) * 4. E.g. 1st retry: 4 seconds. 5th retry: 64 seconds."""
    response = client.chat(
        model=model,
        messages=messages,
        **generation_params,
    )
    return response
