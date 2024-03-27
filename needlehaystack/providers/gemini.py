import os
import google.generativeai as genai
from operator import itemgetter
from typing import Optional

import json
import tiktoken

from .model import ModelProvider

class GeminiPRO(ModelProvider):
    """
    A wrapper class for interacting with Gemini's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the Gemini model to use for evaluations and interactions.
        model (GeminiPRO): An instance of the GeminiPRO client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(candidate_count = 1,
                                      max_output_tokens  = 300,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "gemini-1.0-pro-latest",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        Initializes the GeminiPRO model provider with a specific model.

        Args:
            model_name (str): The name of the GeminiPRO model to use. Defaults to 'gemini-1.0-pro-latest'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """
        api_key = os.getenv('NIAH_MODEL_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs

        genai.configure(api_key=api_key)
        self.generation_config = genai.types.GenerationConfig(**model_kwargs)
        self.model = genai.GenerativeModel(model_name)
        #self.model = AsyncOpenAI(api_key=self.api_key)
        # TODO: convert to GeminiPRO tokenizer
        self.tokenizer = tiktoken.encoding_for_model('gpt-4-0125-preview')
    
    def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """

        print('Nr tokens', len(self.tokenizer.encode(json.dumps(prompt))))
        response = self.model.generate_content(
            contents = prompt,
            generation_config=self.generation_config)
        return response.candidates[0].content.parts[0].text
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return [{
                "role": "user",
                "parts": ["You are a helpful AI bot that answers questions for a user based on the below text. \
                          Keep your response short and direct. Don't give information outside the document or repeat your findings.\n",
                          "Begin Content \n",
                            context,
                            "End content\n"
                            f"Question: {retrieval_question}\n",
                             "Answer: "]
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])

