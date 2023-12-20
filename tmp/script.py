
from litellm import completion
import litellm
async def async_ollama():
    response = await litellm.acompletion(
        model="ollama/meditron", 
        messages=[{ "content": "what is cancer?" ,"role": "user"}], 
        api_base="http://localhost:11434", 
        stream=True
    )
    async for chunk in response:
        print(chunk)

# call async_ollama
import asyncio
asyncio.run(async_ollama())
