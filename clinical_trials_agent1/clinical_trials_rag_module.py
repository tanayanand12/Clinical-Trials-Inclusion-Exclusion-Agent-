# clinical_trials_rag_module.py
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv # type: ignore
import time

try:
    from openai import OpenAI # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


INCLUSION_EXCLUSION_PROMPT = r"""
You are an inclusion/exclusion criteria synthesis assistant. Using the user question and the CLINICAL TRIAL DATA CONTEXT, produce a SINGLE JSON object in the exact schema below. Your output must be valid JSON (no extra keys, no comments, no code fences).

========================
INPUTS
- USER QUESTION:
$question

- CLINICAL TRIAL DATA CONTEXT (retrieved snippets; may be partial or noisy):
$context
========================

========================
OUTPUT SCHEMA (STRICT)
{
  "Inclusion": "<markdown-formatted bullet list>",
  "Exclusion": "<markdown-formatted bullet list>",
  "Similarity": [
    {
      "CT_ID": "NCT########",
      "percentage_similarity": <float 0.0–100.0 with one decimal>
    }
    // zero or more additional entries
  ]
}
========================

REQUIREMENTS
1) GENERAL
   - Return ONLY the JSON object matching the schema above. No prose, no explanations, no markdown fences.
   - Keys must be exactly: "Inclusion", "Exclusion", "Similarity".
   - All numbers must be JSON numbers (not strings).
   - If a required field has no evidence, emit a single bullet: "- Not reported in cited records".

2) "Inclusion" STRING (markdown bullets)
   - Write concise, clinical-grade inclusion criteria synthesized from the CONTEXT.
   - Use markdown bullets with hyphens:
     - "- Age ≥ 18 years"
     - "- Undergoing transradial coronary angiography/PCI"
   - When a criterion is supported by specific trials, cite inline using square brackets with NCT IDs (no commas inside the brackets), e.g.:
     - "- Reverse Barbeau test documented [NCT01234567 NCT08976543]"
   - Merge duplicates across trials into a single, generalizable bullet where appropriate.

3) "Exclusion" STRING (markdown bullets)
   - Same formatting rules as Inclusion.
   - Include typical contraindications or trial-specific exclusions present in the CONTEXT (e.g., ongoing anticoagulation above thresholds, Barbeau Class D, prior ipsilateral RA access, thrombocytopenia).
   - Cite supporting NCT IDs inline as above.

4) "Similarity" ARRAY (computed by you)
   - PURPOSE: Provide a per-trial similarity score indicating how well each cited trial aligns with the synthesized Inclusion/Exclusion profile and overall eligibility framing in the USER QUESTION.
   - COVERAGE RULE:
     a) INCLUDE EVERY trial ID that appears in the CONTEXT (all NCT IDs you can reliably extract), deduplicated.
     b) You MAY add additional NCT IDs from your parametric knowledge ONLY if they are highly relevant to radial access hemostasis/eligibility; if unsure, do not add extras.
   - FORMAT: Each entry must be an object:
     { "CT_ID": "NCT########", "percentage_similarity": <float with one decimal> }
   - ORDERING: Sort entries by descending percentage_similarity.
   - SCORING METHOD (fabricate transparently and consistently):
     • Define three subscores on a 0–10 scale for each trial based on the CONTEXT:
       - InterventionMatch (IM): How closely the device/strategy/population matches the USER QUESTION and synthesized criteria.
       - EndpointEligibilityFit (EEF): How well the trial’s stated inclusion/exclusion structure aligns with the synthesized criteria (overlap, clarity, timing).
       - DesignComparability (DC): How comparable the design is for eligibility benchmarking (randomization, population, setting).
     • Compute the similarity percentage as a weighted sum:
       Similarity% = 100 * [ 0.4*(IM/10) + 0.4*(EEF/10) + 0.2*(DC/10) ]
     • Rounding: round to one decimal place.
     • Thresholding / conservative defaults:
       - When information is partial but directionally consistent, use conservative mid-range values (e.g., IM=6–7, EEF=6–7, DC=5–6).
       - If a trial plainly mismatches the inclusion/exclusion framing, reduce IM and/or EEF accordingly (e.g., 2–4).
       - Keep all final scores within [0.0, 100.0].
   - VALIDATION:
     • Ensure every CT_ID conforms to the pattern NCT[0-9]{8}.
     • No duplicate CT_IDs.
     • If subscores cannot be inferred, assign IM=5, EEF=5, DC=5 (neutral), yielding ~50.0%.

5) CONSISTENCY & CITATIONS
   - The Inclusion/Exclusion bullets should collectively reference (via inline [NCT######## ...]) the same set of trials that appear in "Similarity".
   - If you mention an NCT ID in Inclusion/Exclusion citations, that NCT ID MUST also appear in the "Similarity" array.

QUALITY BAR
- Be concise, clinically precise, and avoid speculative claims.
- Prefer standardized wording (e.g., “Barbeau’s test”, “duplex ultrasound”, “warfarin therapy”, “eGFR < 30 ml/min”).
- Use ranges or representative thresholds only when present or strongly implied in the CONTEXT.

NOW PRODUCE THE JSON OBJECT ONLY.

"""

class ClinicalTrialsRAGModule:
    """Module for generating answers using clinical trial context via RAG."""
    
    def __init__(self, model_name: str = "gpt-4-turbo"):
        """
        Initialize the RAG module.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        load_dotenv()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized ClinicalTrialsRAGModule with model: {self.model_name}")
    
    def create_system_prompt(self) -> str:
        """
        Create the system prompt for clinical trials analysis.
        
        Returns:
            System prompt string
        """
        return """You are an expert clinical trials eligibility specialist focusing exclusively on analyzing and interpreting inclusion and exclusion criteria from ClinicalTrials.gov. Your role is to provide comprehensive, evidence-based insights about patient eligibility requirements using current clinical trial data.

CORE CAPABILITIES:
- Analyze eligibility criteria patterns across clinical trials
- Interpret inclusion requirements for specific patient populations
- Evaluate exclusion criteria and safety considerations
- Assess age, gender, and demographic eligibility requirements
- Compare eligibility patterns across different medical conditions
- Identify common screening requirements and biomarker criteria

RESPONSE GUIDELINES:
1. Always base responses on eligibility criteria from provided clinical trial data
2. Clearly categorize different types of eligibility information:
   - Inclusion criteria requirements and patient characteristics
   - Exclusion criteria and safety restrictions
   - Age and demographic eligibility parameters
   - Medical history and condition-specific requirements
   - Laboratory values and biomarker criteria
   - Prior treatment and medication restrictions

3. When analyzing eligibility across multiple studies:
   - Compare inclusion criteria patterns between trials
   - Highlight common exclusion criteria themes
   - Note variations in age and demographic requirements
   - Identify trends in eligibility strictness or flexibility

4. For eligibility analysis and interpretation:
   - Reference specific inclusion/exclusion text from studies
   - Note enrollment numbers and eligibility impact
   - Describe screening complexity and patient burden
   - Mention eligibility criteria evolution across study phases

5. Present eligibility information in structured format including:
   - Clear sections for inclusion vs exclusion criteria
   - Specific citations using NCT IDs for eligibility examples
   - Quantitative eligibility details (age ranges, lab values)
   - Practical implications for patient screening and recruitment

IMPORTANT LIMITATIONS:
- Extensively interpret and analyze eligibility criteria stated in clinical trial data
- Reason with documented eligibility requirements and screening parameters
- Clearly indicate reasoning process for eligibility pattern analysis
- Focus on factual eligibility criteria rather than clinical recommendations

Remember that eligibility criteria represent research screening protocols and should not be interpreted as clinical guidance.

HARD LIMITATIONS:
- Do NOT use phrases like "None of the identified trials explicitly mention eligibility" or "No specific inclusion criteria were found" or "eligibility requirements not directly available" or anything that reduces response CONFIDENCE and RELEVANCE.
- Extract and synthesize all relevant eligibility information from provided clinical trial data.
- Answer confidently about inclusion/exclusion patterns based on available trial records.
- State reasoning clearly and provide detailed eligibility analysis from clinical trial data.
- Avoid low confidence statements about eligibility requirements.
- Your task is to provide thorough analysis of eligibility criteria, not general clinical advice.
"""
    
#     """You are an expert clinical trials research assistant specializing in analyzing and interpreting clinical trial data from ClinicalTrials.gov. Your role is to provide comprehensive, evidence-based insights using the most current clinical trial information.

# CORE CAPABILITIES:
# - Analyze clinical trial protocols, designs, and methodologies
# - Interpret eligibility criteria, inclusion/exclusion requirements
# - Evaluate study endpoints, outcome measures, and statistical plans
# - Assess study status, recruitment, and timeline information
# - Compare interventions, treatments, and study designs across trials
# - Identify patterns in sponsor types, geographic distribution, and study characteristics

# RESPONSE GUIDELINES:
# 1. Always base responses on the provided clinical trial data
# 2. Clearly distinguish between different types of clinical trial information:
#    - Study design and methodology details
#    - Eligibility criteria and patient populations
#    - Intervention protocols and treatment arms
#    - Primary and secondary endpoints
#    - Study status and recruitment information
#    - Sponsor and location data

# 3. When analyzing multiple studies:
#    - Compare and contrast key study characteristics
#    - Highlight similarities and differences in design approaches
#    - Note variations in patient populations and inclusion criteria
#    - Identify trends in intervention strategies

# 4. For statistical and methodological information:
#    - Reference specific study designs (RCT, observational, etc.)
#    - Note sample sizes and statistical power considerations
#    - Describe randomization and blinding strategies when available
#    - Mention study phases for interventional trials

# 5. Present information in a structured, accessible format that includes:
#    - Clear section headers for different aspects
#    - Specific citations to relevant studies using NCT IDs
#    - Quantitative details when available (enrollment numbers, timeframes)
#    - Practical implications for patients and researchers

# IMPORTANT LIMITATIONS:
# - Extensively interpret reason and provide information that is stated in the clinical trial data
# - Reason with the Documented records and metadata from ClinicalTrials.gov
# - Clearly indicate your reasoning process and how you arrived at conclusions
# - Focus on factual trial characteristics rather than clinical advice

# Remember that clinical trial data represents research protocols and should not be interpreted as clinical recommendations.

# HARD LIMITATIONS:
# - Do NOT use phrases like "None of the identified clinical trials explicitly mention" or "No specific studies were found or "not directly available" or "did not explicitly provide" or anything that reduces you response CONFIDENCE and RELEVANCE.
# - Try to extract and synthesize all relevant information from the provided clinical trial data.
# - Answer confidently and comprehensively based on the available clinical trial records.
# - State your reasoning clearly and provide detailed information from the clinical trial data.
# - Avoid making low confidence statements or vague assertions.
# - Your task is to provide a thorough analysis of the clinical trial data, not to provide answers to personal questions other than the like `how is the weather today?` OR `What your name?` OR `who made you?` etc.
# """

    def create_user_prompt(self, query: str, context: str, studies: list[dict]) -> str:
        # Build citation text exactly like before
        citations = []
        for i, s in enumerate(studies):
            citations.append(f"[{i+1}] {s['title']} (NCT ID: {s['study_id']}, Relevance: {s['similarity_score']:.3f})")
        citations_text = "\n".join(citations) if citations else "No specific studies identified."

        # Compact, JSON-only prompt with fabrication rules for Similarity
        prompt = (
            "USER QUESTION:\n"
            f"{query}\n\n"
            "CLINICAL TRIAL DATA CONTEXT:\n"
            f"{context}\n\n"
            "RELEVANT STUDIES IDENTIFIED:\n"
            f"{citations_text}\n\n"
            "OUTPUT FORMAT (STRICT JSON ONLY — NO PROSE, NO BACKTICKS):\n"
            "{\n"
            '  "Inclusion": "<markdown bullets; cite trials inline like [NCT######## NCT########]>",\n'
            '  "Exclusion": "<markdown bullets; cite trials inline like [NCT########]>",\n'
            '  "Similarity": [{"CT_ID":"NCT########","percentage_similarity": 0.0}]\n'
            "}\n\n"
            "RULES:\n"
            "- Return ONLY the JSON above. Do not add any other keys or text.\n"
            "- Inclusion/Exclusion: extract ONLY eligibility criteria from the CONTEXT and the CITATIONS; bullets only.\n"
            "- Merge duplicates; append inline NCT IDs for supporting trials in square brackets with spaces between IDs.\n"
            "- If a section is missing in records, output a single bullet: \"- Not reported in cited records\".\n"
            "- Similarity must include EVERY distinct NCT ID found in the context OR in the citations; no duplicates; no invented IDs.\n"
            "- Compute percentage_similarity by this transparent method (0–100, 1 decimal):\n"
            "  * Define subscores on 0–10: IM (InterventionMatch), EEF (Endpoint/Eligibility Fit), DC (Design Comparability).\n"
            "  * Similarity% = 100 * (0.4*(IM/10) + 0.4*(EEF/10) + 0.2*(DC/10)). Round to 1 decimal.\n"
            "  * Use conservative defaults (IM=6–7, EEF=6–7, DC=5–6) when details are partial; penalize obvious mismatch.\n"
            "- Validate CT_ID format: ^NCT[0-9]{8}$; sort Similarity by descending percentage.\n"
            "- Ensure every NCT cited in Inclusion/Exclusion appears in Similarity.\n"
            "- JSON numbers for percentages (not strings).\n"
            "NOW OUTPUT THE JSON OBJECT ONLY."
        )
        return prompt


#     def create_user_prompt(self, query: str, context: str, studies: List[Dict[str, Any]]) -> str:
#         """
#         Create the user prompt with query and context.
        
#         Args:
#             query: User query
#             context: Extracted context from clinical trials
#             studies: List of relevant studies metadata
            
#         Returns:
#             Formatted user prompt
#         """
#         # Format study citations
#         citations = []
#         for i, study in enumerate(studies):
#             citation = f"[{i+1}] {study['title']} (NCT ID: {study['study_id']}, Relevance: {study['similarity_score']:.3f})"
#             citations.append(citation)
        
#         citations_text = "\n".join(citations) if citations else "No specific studies identified."
        
#         prompt = f"""
# USER QUESTION: {query}

# CLINICAL TRIAL DATA CONTEXT:
# {context}

# RELEVANT STUDIES IDENTIFIED:
# {citations_text}

# Please provide a comprehensive analysis based on the clinical trial information above. Structure your response as follows:

# 1. **Executive Summary** (50-75 words)
#    - Direct answer to the user's question
#    - Key findings from the clinical trial data

# 2. **Detailed Analysis** 
#    - Comprehensive synthesis of relevant trial information
#    - Specific details from individual studies with NCT ID references
#    - Comparison across multiple trials when applicable

# 3. **Study Characteristics**
#    - Overview of study designs, phases, and methodologies
#    - Patient populations and eligibility criteria
#    - Intervention details and treatment protocols

# 4. **Key Findings and Patterns**
#    - Notable trends or patterns across the identified trials
#    - Variations in approach or methodology
#    - Geographic distribution and sponsor types if relevant

# 5. **Practical Implications**
#    - What this means for patients seeking relevant trials
#    - Considerations for researchers in this field
#    - Current state of research in this area

# 6. **Data Limitations**
#    - What information was not available in the trial records
#    - Areas where additional research or data might be needed

# Use NCT IDs [NCT########] when referencing specific trials. If multiple studies address the same point, cite all relevant NCT IDs.

# Focus on factual information from the trial records and avoid clinical recommendations or medical advice.
# """ 
#         prompt += """

# OUTPUT FORMAT (STRICT):
# Return a SINGLE valid JSON object with EXACTLY these keys and nothing else:

# {
#   "Inclusion": "<markdown bullets listing inclusion criteria synthesized from the CLINICAL TRIAL DATA CONTEXT above; cite trials inline like [NCT########]>",
#   "Exclusion": "<markdown bullets listing exclusion criteria synthesized from the CLINICAL TRIAL DATA CONTEXT above; cite trials inline like [NCT########]>",
#   "Similarity": [{"CT_ID":"...", "percentage_similarity": 00.0}, ...]
# }

# Rules:
# - Do NOT add any top-level keys besides Inclusion, Exclusion, Similarity.
# - Inclusion/Exclusion must be markdown-formatted bullet points only.
# - If a criterion is implied across multiple trials, merge and cite all relevant NCT IDs.
# - If a criterion is absent, write a single bullet: "- Not reported in cited records".
# - Do NOT include prose outside the JSON.
# """

#         return prompt
    
    def generate_answer(self, query: str, context: str, studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer using OpenAI with clinical trial context.
        
        Args:
            query: User query
            context: Extracted context from clinical trials
            studies: List of relevant studies
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            system_prompt = self.create_system_prompt()
            user_prompt = self.create_user_prompt(query, context, studies)
            
            # Generate response using OpenAI
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=0.2,  # Lower temperature for more consistent, factual responses
                # max_tokens=2000,
                # max_completion_tokens=2000,
                # top_p=0.9,
                # frequency_penalty=0.1,
                # presence_penalty=0.1
                reasoning_effort= "minimal"
            )
            
            generation_time = time.time() - start_time
            answer = response.choices[0].message.content
            
            # Format citations for display
            formatted_citations = []
            for i, study in enumerate(studies):
                citation = f"[{i+1}] {study['title']} - NCT ID: {study['study_id']} (Relevance Score: {study['similarity_score']:.3f})"
                formatted_citations.append(citation)
            
            # Calculate usage statistics
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            return {
                "answer": answer,
                "citations": formatted_citations,
                "studies": studies,
                "metadata": {
                    "model_used": self.model_name,
                    "generation_time": generation_time,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "studies_analyzed": len(studies),
                    "context_length": len(context)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while analyzing the clinical trial data: {str(e)}. Please try rephrasing your question or check if the clinical trial data is available.",
                "citations": [],
                "studies": studies,
                "metadata": {
                    "error": str(e),
                    "model_used": self.model_name,
                    "studies_analyzed": len(studies)
                }
            }
    
    def validate_response_quality(self, response: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Validate the quality of the generated response.
        
        Args:
            response: Generated response dictionary
            query: Original user query
            
        Returns:
            Response with quality assessment
        """
        quality_metrics = {
            "has_answer": bool(response.get("answer")),
            "has_citations": len(response.get("citations", [])) > 0,
            "answer_length": len(response.get("answer", "")),
            "studies_referenced": len(response.get("studies", [])),
            "contains_nct_ids": "NCT" in response.get("answer", ""),
            "structured_response": any(marker in response.get("answer", "") for marker in ["**", "##", "1.", "2.", "3."])
        }
        
        # Calculate overall quality score
        quality_score = sum([
            quality_metrics["has_answer"] * 0.3,
            quality_metrics["has_citations"] * 0.2,
            (quality_metrics["answer_length"] > 200) * 0.2,
            (quality_metrics["studies_referenced"] > 0) * 0.15,
            quality_metrics["contains_nct_ids"] * 0.1,
            quality_metrics["structured_response"] * 0.05
        ])
        
        response["quality_assessment"] = {
            "metrics": quality_metrics,
            "overall_score": quality_score,
            "quality_level": "high" if quality_score > 0.8 else "medium" if quality_score > 0.5 else "low"
        }
        
        logger.info(f"Response quality assessment: {quality_score:.2f} ({response['quality_assessment']['quality_level']})")
        return response