from anthropic import Anthropic, AnthropicBedrock

client = AnthropicBedrock(max_retries=0)


def _messages_complete(messages, max_tokens_to_sample, model, temperature):
    # messages = convert_completion_to_messages(prompt)
    completion = client.messages.create(
        model=model,
        max_tokens=max_tokens_to_sample,
        temperature=temperature,
        # stop_sequences=["</function_calls>", "\n\nHuman:"],
        messages=messages,
    )

    return completion