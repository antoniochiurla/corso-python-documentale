import anthropic

client = anthropic.Anthropic()  # legge ANTHROPIC_API_KEY dall'ambiente

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="Sei un assistente conciso.",
    messages=[
        {"role": "user", "content": "Spiega cos'è un'API in due frasi."}
    ],
)

print(message.content[0].text)
