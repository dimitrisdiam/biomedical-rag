# wrapper around the llm call. uses openai if there's a key, otherwise falls
# back to just returning the top passage so I can still run everything offline

import os


class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None
        if self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

    @property
    def online(self) -> bool:
        return self._client is not None

    def complete(self, system: str, user: str) -> str:
        if not self.online:
            return self._extractive_fallback(user)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _extractive_fallback(user: str) -> str:
        # no key, so just hand back the first context block. keeps retrieval
        # eval working without paying for tokens
        marker = "Context:"
        if marker in user:
            context = user.split(marker, 1)[1]
            first = context.strip().split("\n\n")[0]
            return f"[offline mode, no LLM] Most relevant passage: {first}"
        return "[offline mode, no LLM]"
