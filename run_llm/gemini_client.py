import os
from typing import List, Optional
from google import genai
from google.genai import types
from functools import lru_cache

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client with an optional API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided either through environment variable or constructor")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash-preview-04-17"
        self.config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="text/plain",
        )

    @lru_cache(maxsize=1)
    def get_model(self):
        """Get the Gemini model instance with caching."""
        return self.client.models.get_model(self.model)

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using the Gemini model.
        
        Args:
            prompt (str): The input prompt for the model
            
        Returns:
            str: The generated response
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.config,
        )
        
        return response.text

    def generate_content_stream(self, prompt: str):
        """
        Generate content using the Gemini model with streaming.
        
        Args:
            prompt (str): The input prompt for the model
            
        Yields:
            str: Chunks of the generated response
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=self.config,
        ):
            yield chunk.text

# Example usage
if __name__ == "__main__":
    client = GeminiClient()
    
    # Example of non-streaming generation
    response = client.generate_content("What is the capital of France?")
    print("Non-streaming response:", response)
    
    # Example of streaming generation
    print("\nStreaming response:")
    for chunk in client.generate_content_stream("Tell me a short story about a cat."):
        print(chunk, end="") 