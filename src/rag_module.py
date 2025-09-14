# rag_module.py
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RAGModule:
    """RAG system for generating answers from retrieved documents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """Initialize the RAG module."""
        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up OpenAI API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_answer(self, query: str, contexts: List[Document]) -> Dict[str, Any]:
        """Generate an evidence-based medical research answer."""
        try:
            # Prepare the prompt with context
            prompt = self._prepare_prompt(query, contexts)
            
            # Call OpenAI API using chat completions
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert medical research assistant specializing in evidence-based analysis of scientific literature."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Prepare citations from contexts with metadata
            citations = []
            for i, doc in enumerate(contexts):
                citation = {
                    "id": f"[{i+1}]",
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                citations.append(citation)
            
            return {
                "answer": answer,
                "contexts": contexts,
                "citations": citations
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _prepare_prompt(self, query: str, contexts: List[Document]) -> str:
        """Prepare the medical research analysis prompt with query and context."""
        # Format contexts with citations
        context_texts = []
        citations_text = []
        
        for i, doc in enumerate(contexts):
            # Add formatted context
            context_texts.append(f"[{i+1}] {doc.page_content}\n")
            
            # Add citation with metadata
            citation = f"[{i+1}] "
            if "title" in doc.metadata:
                citation += f"{doc.metadata['title']}. "
            if "authors" in doc.metadata:
                citation += f"{doc.metadata['authors']}. "
            if "journal" in doc.metadata:
                citation += f"{doc.metadata['journal']}. "
            if "year" in doc.metadata:
                citation += f"({doc.metadata['year']})"
            citations_text.append(citation)

        # Join all contexts and citations
        context = "\n\n".join(context_texts)
        citations = "\n".join(citations_text)

        prompt = f"""You are an expert medical research assistant analyzing scientific literature. Your task is to provide comprehensive, evidence-based insights with precise citations.

        USER QUESTION: {query}

        CONTEXT FROM MEDICAL PAPERS:
        {context}

        RELEVANT PAPERS:
        {citations}

        GUIDELINES FOR YOUR RESPONSE:
        1. Prioritize information directly present in the provided papers
        2. Extract and synthesize findings across multiple sources when available
        3. Present a nuanced analysis that acknowledges:
           - Strength and consistency of evidence
           - Methodology quality (study design, sample size, controls)
           - Statistical significance of findings (p-values, confidence intervals)
           - Clinical relevance versus statistical significance
        4. When evaluating evidence quality:
           - Clearly identify study designs (RCT, meta-analysis, cohort, case-control, etc.)
           - Note sample characteristics (size, demographics, inclusion/exclusion criteria)
           - Address potential limitations or biases in methodology
           - Indicate levels of evidence using recognized frameworks (e.g., GRADE)
        5. For numerical data:
           - Include specific effect sizes, risk ratios, hazard ratios, or odds ratios
           - Provide confidence intervals and p-values when available
           - Contextualize percentages with absolute numbers
           - Compare findings across studies when possible
        6. For comparative analyses:
           - Present data from each intervention/group side-by-side
           - Highlight statistical and clinical significance of differences
           - Note heterogeneity in methods or populations that might affect comparability

        FORMAT YOUR RESPONSE AS:
        1. Executive Summary 
        2. Key Findings 
        3. Supporting Evidence 
        4. Clinical Implications 
        5. Evidence Quality Assessment
        6. References (use citation to create the references section with pdf name, page number and title/topic *DO NOT INCLUDE TEXT*)

        Remember to:
        - Use bracketed numbers [1], [2], etc. for citations
        - Include all relevant sources when multiple papers address the same point
        - Focus on evidence-supported findings
        - Maintain scientific objectivity
        - Provide actionable insights based on available evidence
        """
        
        return prompt