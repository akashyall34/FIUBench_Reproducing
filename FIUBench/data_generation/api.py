"""
API evaluators for GPT and Gemini models.

Provides interfaces to generate answers using OpenAI GPT and Google Gemini APIs,
used for factual consistency evaluation in the unlearning benchmark.
"""

import os
from typing import Dict, Optional, List, Any


class GPTEvaluator:
    """Evaluator using OpenAI GPT models."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 20):
        """
        Initialize GPT evaluator.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o-mini", "gpt-4")
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

    def generate_answer(self, question: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate answer using GPT.

        Args:
            question: Dict with keys:
                - prompted_system_content: System prompt
                - prompted_content: User message
                - image_list: List of images (not supported in this implementation)

        Returns:
            Dict with 'prediction' key containing the model's response
        """
        system_content = question.get("prompted_system_content", "")
        user_content = question.get("prompted_content", "")

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            prediction = response.choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            prediction = "0.0"

        return {"prediction": prediction}


class GeminiEvaluator:
    """Evaluator using Google Gemini models."""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        """
        Initialize Gemini evaluator.

        Args:
            api_key: Google API key for Gemini
            model: Model name (e.g., "gemini-pro", "gemini-1.5-pro")
        """
        self.api_key = api_key
        self.model = model

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")

    def generate_answer(self, question: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate answer using Gemini.

        Args:
            question: Dict with keys:
                - prompted_system_content: System prompt
                - prompted_content: User message
                - image_list: List of images (not supported in this implementation)

        Returns:
            Dict with 'prediction' key containing the model's response
        """
        system_content = question.get("prompted_system_content", "")
        user_content = question.get("prompted_content", "")

        try:
            prompt = user_content
            if system_content:
                prompt = f"{system_content}\n\n{user_content}"

            response = self.client.generate_content(prompt)
            prediction = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            prediction = "0.0"

        return {"prediction": prediction}


# Message templates for data generation (used in illustration_generate.py)
system_message = "You are a helpful assistant."
user_message = "Please provide a response."

# Placeholder for job management (if needed for async processing)
jobs = []
