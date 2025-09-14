

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

## 🖥️ Frontend Integration: Extracting the Magic

> **No backend changes needed!** Here’s how your frontend should extract and render the Inclusion/Exclusion/Similarity blocks, citations, and all viz-friendly metadata from the API response.

### 📦 What’s in the API Response?

**Use these JSONPaths from the root of the API response:**

- **Parsed payload (use this first if present):**
        - Inclusion: `$.data.final_response.comprehensive_answer_parsed.Inclusion` *(string, markdown)*
        - Exclusion: `$.data.final_response.comprehensive_answer_parsed.Exclusion` *(string, markdown)*
        - Similarity: `$.data.final_response.comprehensive_answer_parsed.Similarity` *(array of {CT_ID, percentage_similarity})*
- **If parsed block is missing, fall back to model’s JSON string:**
        - Raw JSON string: `$.data.final_response.comprehensive_answer` *(stringified JSON)*
                - Parse it and read the same three keys.
- **Citations (for hover/collapsible details):**
        - Citation text list: `$.data.pipeline_steps.criteria_analysis.citations` *(array of strings — already formatted, keep “as-is”)*
        - Optional local doc context for tooltips: `$.data.pipeline_steps.local_data_retrieval.results` *(array with `pdf_name`, `page_number`, `page_summary`, `similarity_score`)*
- **Useful meta for UI:**
        - User query: `$.data.user_query`
        - Confidence (criteria agent): `$.data.pipeline_steps.criteria_analysis.confidence`
        - Chunks used: `$.data.final_response.context_used.local_data_chunks`
        - Total citations count: `$.data.final_response.context_used.total_citations`
        - Model id: `$.data.final_response.model_used`
        - Processing time: `$.data.processing_time`

---

### 📝 TypeScript Data Contracts

```ts
export type SimilarityItem = {
        CT_ID: string;
        percentage_similarity: number; // 0–100
};

export type IePayload = {
        Inclusion: string;  // markdown bullets (may contain sub-bullets)
        Exclusion: string;  // markdown bullets (may contain sub-bullets)
        Similarity: SimilarityItem[];
};

export type ApiEnvelope = {
        success: boolean;
        data: any; // your full server payload
};
```

---

### 🧩 Extraction Logic (Robust, Zero Backend Changes)

```ts
function extractIePayload(api: ApiEnvelope): {
        ie: IePayload | null;
        citations: string[];
        meta: {
                userQuery?: string;
                confidence?: number;
                chunks?: number;
                totalCitations?: number;
                modelUsed?: string;
                processingTime?: number;
        };
        errors: string[];
} {
        const errors: string[] = [];
        const d = api?.data ?? {};
        const fr = d.final_response ?? {};
        let Inclusion: string | undefined;
        let Exclusion: string | undefined;
        let Similarity: SimilarityItem[] | undefined;

        // 1) Prefer parsed
        const parsed = fr.comprehensive_answer_parsed;
        if (parsed?.Inclusion && parsed?.Exclusion && parsed?.Similarity) {
                Inclusion = String(parsed.Inclusion);
                Exclusion = String(parsed.Exclusion);
                Similarity = parsed.Similarity as SimilarityItem[];
        } else {
                // 2) Fallback: parse the JSON string the model returned
                const raw = fr.comprehensive_answer;
                if (typeof raw === "string") {
                        try {
                                const obj = JSON.parse(raw);
                                Inclusion = obj?.Inclusion;
                                Exclusion = obj?.Exclusion;
                                Similarity = obj?.Similarity;
                        } catch (e) {
                                errors.push("Failed to parse comprehensive_answer JSON string.");
                        }
                }
        }

        // Validate minimal structure
        const ie: IePayload | null =
                Inclusion && Exclusion && Array.isArray(Similarity)
                        ? { Inclusion, Exclusion, Similarity: Similarity! }
                        : null;

        // Citations (render exactly as provided)
        const citations: string[] =
                d?.pipeline_steps?.criteria_analysis?.citations ?? [];

        // Meta for viz / headers
        const meta = {
                userQuery: d?.user_query,
                confidence: d?.pipeline_steps?.criteria_analysis?.confidence,
                chunks: fr?.context_used?.local_data_chunks,
                totalCitations: fr?.context_used?.total_citations,
                modelUsed: fr?.model_used,
                processingTime: d?.processing_time,
        };

        return { ie, citations, meta, errors };
}
```

---

### 🎨 Rendering Guidance

- **Inclusion/Exclusion**: Render the strings as **markdown**. They already contain bullets, sub-bullets, and inline NCT citations like `[NCT06108414]`. Don’t alter the text; just pass into your markdown renderer.
- **CITATION_TEXT**: Show `citations[]` as a collapsible “Citations” panel or inline “footnotes”. Keep them **verbatim** (already formatted).
- **Similarity Viz**:
        - Use `ie.Similarity` to render bars or chips:
                - Label: `CT_ID`
                - Value: `percentage_similarity` (0–100)
        - Sort descending before rendering.

Example bar-prep:
```ts
const bars = (ie?.Similarity ?? [])
        .slice()
        .sort((a,b) => b.percentage_similarity - a.percentage_similarity)
        .map(s => ({ label: s.CT_ID, value: s.percentage_similarity }));
```

- **Tooltips**: Map `CT_ID` to any matching citation strings (simple `.find(str => str.includes(CT_ID))`) or show all citations in a modal.

---

### ⚠️ Edge Cases & Fallbacks

1. **Parsed missing, raw parse fails**
         - Show a graceful error banner using `errors[]`, and display the top-level `fr.comprehensive_answer` as plain text so users still see something.
2. **No Similarity array**
         - Render Inclusion/Exclusion normally; hide the viz panel.
3. **Empty citations**
         - Hide the “Citations” section.
4. **Markdown safety**
         - Use a markdown component that supports lists and code blocks; do not sanitize away square brackets (they are the inline NCT cites).

---

### ✅ Quick Checklist for Your FE Dev

- [ ] Read `IePayload` via `comprehensive_answer_parsed`, else parse `comprehensive_answer`.
- [ ] Render `Inclusion` & `Exclusion` as markdown (no mutation).
- [ ] Show `citations` exactly as provided.
- [ ] Visualize `Similarity` as a sorted bar chart/chips.
- [ ] Use `meta` to display small header: model, confidence, chunks, time.
- [ ] Handle fallbacks (errors array).

That’s it — no backend edits required. Your frontend is now ready to render clinical trial insights like a pro!

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

> **Note:** This project is for research and educational purposes. Not for clinical decision-making or patient care. But it *is* for building the future. Join us.
