import asyncio
from typing import Any, Dict, Optional
import httpx

# Import the LMLMDetector built previously
from lmlm_detector import LMLMDetector, Language

# 1. Define Language-Specific System Prompts mapped by ISO 639-1 codes
SYSTEM_PROMPTS: Dict[str, str] = {
    "en": (
        "You are a helpful AI assistant. Answer the user's request accurately and "
        "concisely in English."
    ),
    "fr": (
        "Vous êtes un assistant IA attentif. Répondez à la demande de l'utilisateur "
        "de manière précise et concise en français."
    ),
    "de": (
        "Sie sind ein hilfreicher KI-Assistent. Beantworten Sie die Anfrage des "
        "Benutzers präzise und prägnant auf Deutsch."
    ),
    "es": (
        "Eres un asistente de IA atento. Responde a la solicitud del usuario "
        "de forma precisa y concisa en español."
    ),
}

DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["en"]


class AsyncOllamaLanguageRouter:
    """Routes incoming prompts to Ollama with language-appropriate system prompts."""

    def __init__(
        self,
        detector: LMLMDetector,
        model_name: str = "llama3",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.detector = detector
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")

    def get_system_prompt(self, iso_code: str) -> str:
        """Retrieves system prompt for language ISO code or falls back to default."""
        return SYSTEM_PROMPTS.get(iso_code.lower(), DEFAULT_SYSTEM_PROMPT)

    async def generate_response(
        self,
        user_prompt: str,
        temperature: float = 0.7,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """Detects language, builds request payload, and queries the local Ollama instance."""
        # 1. Detect language ISO code and confidence
        scores = self.detector.get_confidence_scores(user_prompt)
        top_score = scores[0] if scores else None

        detected_iso = top_score.iso_code_639_1 if top_score else "en"
        confidence = top_score.score if top_score else 0.0

        # 2. Select system prompt
        system_prompt = self.get_system_prompt(detected_iso)

        # 3. Build Ollama Chat API Payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": temperature},
            "stream": False,
        }

        # 4. Execute Async Request to Ollama
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=60.0)
            close_client = True

        try:
            response = await client.post(
                f"{self.ollama_base_url}/api/chat", json=payload
            )
            response.raise_for_status()
            ollama_data = response.json()

            return {
                "prompt": user_prompt,
                "detected_language": top_score.language if top_score else "ENGLISH",
                "iso_code": detected_iso,
                "confidence": confidence,
                "system_prompt_used": system_prompt,
                "response": ollama_data.get("message", {}).get("content", ""),
            }
        finally:
            if close_client:
                await client.aclose()


# --- Execution Example ---

async def main():
    # Initialize the LMLM Detector
    detector = (
        LMLMDetector.builder()
        .from_languages("ENGLISH", "FRENCH", "GERMAN", "SPANISH")
        .set_minimum_confidence(0.20)
        .build()
    )

    router = AsyncOllamaLanguageRouter(detector=detector, model_name="llama3")

    test_prompts = [
        "What are three key benefits of local AI models?",
        "Quels sont les avantages d'utiliser des modèles d'IA locaux?",
        "Was sind die Vorteile von lokalen KI-Modellen?",
        "¿Cuáles son las ventajas de los modelos de IA locales?",
    ]

    print("--- Processing Prompts through LMLM + Ollama Pipeline ---\n")
    
    # Process prompts concurrently using httpx session reuse
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [router.generate_response(prompt, client=client) for prompt in test_prompts]
        results = await asyncio.gather(*tasks)

    for res in results:
        print(f"PROMPT    : {res['prompt']}")
        print(f"LANG      : {res['detected_language']} [{res['iso_code'].upper()}] (Confidence: {res['confidence']})")
        print(f"RESPONSE  : {res['response'].strip()}\n")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
