import openai # type: ignore

class LocalGeneralizationAgent:
    """
    Agent for generating local generalizations (summaries) of text chunks using OpenAI models.
    """

    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model

    def local_generalise(self, text: str) -> str:
        """
        Use OpenAI's SOTA model to summarize the input text.
        """
        if not text.strip():
            return ""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize the following text in 1-2 sentences."},
                    {"role": "user", "content": text[:3000]}
                ],
                max_tokens=120,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI summary error: {e}")
            return ""