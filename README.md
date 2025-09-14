

# 🚀🧬 Clinical Trials Inclusion/Exclusion Agent: The AI Supercollider for Medical Research

<p align="center">
    <img src="https://img.shields.io/badge/AI-Powered-blueviolet?style=for-the-badge" alt="AI-Powered"/>
    <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License"/>
    <img src="https://img.shields.io/badge/RAG-End--to--End-orange?style=for-the-badge" alt="RAG"/>
    <img src="https://img.shields.io/badge/Clinical%20Trials-Next%20Gen-blue?style=for-the-badge" alt="Clinical Trials"/>
</p>

<p align="center">
    <b>"Where clinical science meets AI wizardry, and eligibility criteria bow before your queries."</b>
</p>

---

## � Why This Project? (A Love Letter to Science)

Clinical research is a labyrinth. Eligibility criteria are riddles. Data is a galaxy, but answers are black holes. <br>
**This repo is your AI-powered starship.**

---

## 🏆 What Makes It Legendary?

### 🧠 End-to-End RAG Pipeline
- **Retrieval-Augmented Generation**: Not just LLMs, but LLMs with brains *and* memory, and a sixth sense for context.
- **Local & Cloud Data**: Fetches, chunks, and vectorizes clinical trial data from anywhere in the universe.
- **Query Enhancement**: Your question, but supercharged—AI-enriched for maximum relevance.
- **Inclusion/Exclusion Extraction**: Pinpoints eligibility like a laser-guided scalpel.
- **OpenAI-Powered Reasoning**: Final answers are structured, cited, and ready for peer review.
- **Batch & Async**: Handles one question or a thousand, all at once, with the grace of a quantum computer.
- **Plug & Play**: Modular, extensible, and ready for your next moonshot.
- **Logging & Monitoring**: See the magic, debug the chaos, audit the brilliance.
- **Vectorizer & Chunker**: Turns documents into AI-digestible bites, like a chef for neural networks.

---

## 🏗️ Architecture: The AI Engine Room

```ascii
┌───────────────────────────────────────────────────────────────┐
│                        User Query                            │
└───────────────┬───────────────────────────────┬───────────────┘
                                │                               │
                                ▼                               ▼
     [ComprehensiveRAGPipelineAgent]      [RAGPipelineManager]
                                │                               │
                ┌───────┴────────┐              ┌───────┴────────┐
                ▼                ▼              ▼                ▼
 [DocumentRetriever] [InclusionExclusionAgent]   [Batch/Async]
                │                │              │                │
                ▼                ▼              ▼                ▼
 [Vectorizer/Chunker] [Criteria Extraction]   [OpenAI Model]
                │                │              │                │
                └───────┬────────┴───────┬──────┴────────────────┘
                                ▼                ▼
                 [Final Structured Response]
                                │
                                ▼
     [JSON Output + Citations + Confidence]
```

**Core Components:**
- `DocumentRetriever`: Finds the needles in your data haystack.
- `ClinicalTrialInclusionExclusionAgent`: Extracts eligibility with surgical precision.
- `ComprehensiveRAGPipelineAgent`: The conductor of this AI orchestra.
- `RAGPipelineManager`: Batch, async, and multi-pipeline magic.
- `Vectorizer & Chunker`: Turns documents into AI-digestible bites.
- `Logging & Monitoring`: See the magic, debug the chaos.

---

## 🌠 Real-World Scenarios: Where This Shines

- **Clinical Trial Design**: Accelerate protocol review and eligibility checks. No more manual slogging.
- **Screening Automation**: Power up patient recruitment. Find the right fit, instantly.
- **Evidence Synthesis**: For research, regulatory, and beyond. Summarize, cite, and conquer.
- **Education**: Teach the next generation of clinical AI engineers. Inspire awe.
- **Moonshot Projects**: Build the next big thing in digital health.

---

## 🚀 Quickstart: From Zero to AI Hero

### 1. Clone the Repo
```sh
git clone https://github.com/yourusername/clinical-trials-inclusion-exclusion-agent.git
cd clinical-trials-inclusion-exclusion-agent
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Configure Your Secrets
- Copy `.env.example` to `.env` and fill in your OpenAI API key and other secrets.
- Place credential files (like `service_account_credentials.json`) in the root. **Never commit them!**

### 4. Run the Pipeline
```python
from clinical_trials_inclusion_exclusion_agent.src.pipeline import ComprehensiveRAGPipelineAgent
import asyncio

async def main():
        agent = ComprehensiveRAGPipelineAgent()
        result = await agent.process_comprehensive_query(
                user_query="What are the age requirements for cancer clinical trials?",
                local_model_id="your_local_model_id"
        )
        print(result)

if __name__ == "__main__":
        asyncio.run(main())
```

### 5. Batch Mode: AI at Scale
```python
from clinical_trials_inclusion_exclusion_agent.src.pipeline import RAGPipelineManager
import asyncio

async def main():
        manager = RAGPipelineManager()
        queries = [
                {"user_query": "What are the age requirements for cancer clinical trials?", "local_model_id": "model1"},
                {"user_query": "Common exclusion criteria for cardiovascular disease trials?", "local_model_id": "model2"}
        ]
        results = await manager.batch_process_queries(queries)
        print(results)

if __name__ == "__main__":
        asyncio.run(main())
```

---

## 🧩 Configuration: Your AI, Your Rules

- `.env` for secrets (see `.env.example`).
- `requirements.txt` for dependencies.
- `config.py` for advanced tuning.
- **Environment Variables:**
    - `OPENAI_API_KEY`: Your OpenAI API key
    - `MODEL_ID_GPT5`: (Optional) Model name for OpenAI (default: gpt-5-2025-08-07)
    - Others as needed for your data sources

---

## 📝 Output: Science, Structured

- **JSON**: User query, pipeline steps, final response, processing time, error (if any).
- **Citations**: Every claim, every number, every time.
- **Confidence Scores**: Know what the AI knows (and what it doesn't).
- **Bullet Points**: For humans. For machines. For science.

---

## 🛡️ Security & Best Practices

- **Never commit credentials or secrets.**
- `.gitignore` is your friend—use it.
- Log responsibly. Audit regularly.
- Respect privacy, always.

---

## 🧠 Extending the Magic

- Add new retrievers in `src/retrieval/`.
- Supercharge criteria extraction in `src/inclusion_exclusion/`.
- Plug in new LLMs or APIs—just update the pipeline agent.
- Build your own agents and managers. The framework is yours.

---

## 🤝 Contributing: Join the AI Renaissance

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your branch and open a Pull Request
5. Add a meme to your PR (optional, but encouraged)

---

## 🧑‍🚀 Call to Action: Be a Clinical AI Pioneer

This is not just a repo. It's a launchpad. It's a movement. It's your chance to:
- Change how clinical research is done
- Build tools that save lives
- Inspire the next generation
- Leave your mark on science

---

## 📄 License

MIT. Free as in science. Free as in speech. Free as in "let's build the future together."

---

## 🙏 Acknowledgements

- OpenAI for LLM APIs
- ClinicalTrials.gov for public datasets
- All contributors, dreamers, and doers
- The AI community for pushing the boundaries

---

## 💬 Questions? Bugs? Ideas?

Open an issue, start a discussion, or send a carrier pigeon. (Carrier pigeons not guaranteed to be AI-powered.)

---

> **Note:** This project is for research and educational purposes. It *is* for building the future. Join us.
