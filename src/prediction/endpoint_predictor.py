# endpoint_predictor.py
import logging
import json
from typing import List, Dict, Any, Optional
import openai # type: ignore
from dotenv import load_dotenv # type: ignore
import os

from src.retrieval.timepoint_parser import TimepointData

logger = logging.getLogger(__name__)

class EndpointPredictor:
    """LLM orchestration for clinical trial endpoint timing prediction."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize endpoint predictor.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        load_dotenv()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def prepare_evidence(
        self,
        docs_results: List[Dict[str, Any]],
        csv_results: List[Dict[str, Any]],
        parsed_timepoints: Dict[str, List[TimepointData]],
        max_trials: int = 6
    ) -> Dict[str, Any]:
        """
        Prepare evidence for LLM prediction.
        
        Args:
            docs_results: Document retrieval results
            csv_results: CSV retrieval results
            parsed_timepoints: Parsed timepoint data
            max_trials: Maximum trials to include
            
        Returns:
            Structured evidence
        """
        # Select top trials by similarity score
        top_csv_trials = sorted(
            csv_results, 
            key=lambda x: x.get('similarity_score', 0), 
            reverse=True
        )[:max_trials]
        
        # Build trial summaries
        trial_summaries = []
        for trial in top_csv_trials:
            nct = trial.get('nct_number', '')
            timepoints = parsed_timepoints.get(nct, [])
            
            summary = {
                'nct_number': nct,
                'primary_outcomes': trial.get('primary_outcomes', ''),
                'secondary_outcomes': trial.get('secondary_outcomes', ''),
                'similarity_score': trial.get('similarity_score', 0),
                'timepoints': [
                    {
                        'days': tp.days,
                        'unit': tp.unit,
                        'value': tp.value,
                        'outcome_type': getattr(tp, 'outcome_type', 'unknown'),
                        'text': tp.text
                    } for tp in timepoints
                ]
            }
            trial_summaries.append(summary)
        
        # Extract key insights from docs
        doc_insights = []
        for doc in docs_results[:3]:  # Top 3 docs
            insight = {
                'text': doc.get('text', '')[:500],  # Truncate long texts
                'metadata': doc.get('metadata', {}),
                'similarity_score': doc.get('similarity_score', 0)
            }
            doc_insights.append(insight)
        
        # Calculate timepoint statistics
        all_primary_days = []
        all_secondary_days = []
        
        for timepoints in parsed_timepoints.values():
            for tp in timepoints:
                if hasattr(tp, 'outcome_type'):
                    if tp.outcome_type == 'primary':
                        all_primary_days.append(tp.days)
                    elif tp.outcome_type == 'secondary':
                        all_secondary_days.append(tp.days)
        
        timepoint_stats = {
            'primary': {
                'count': len(all_primary_days),
                'mean': sum(all_primary_days) / len(all_primary_days) if all_primary_days else 0,
                'median': sorted(all_primary_days)[len(all_primary_days)//2] if all_primary_days else 0,
                'range': [min(all_primary_days), max(all_primary_days)] if all_primary_days else [0, 0]
            },
            'secondary': {
                'count': len(all_secondary_days),
                'mean': sum(all_secondary_days) / len(all_secondary_days) if all_secondary_days else 0,
                'median': sorted(all_secondary_days)[len(all_secondary_days)//2] if all_secondary_days else 0,
                'range': [min(all_secondary_days), max(all_secondary_days)] if all_secondary_days else [0, 0]
            }
        }
        
        return {
            'trial_summaries': trial_summaries,
            'doc_insights': doc_insights,
            'timepoint_stats': timepoint_stats,
            'total_relevant_trials': len(trial_summaries)
        }
    
    async def predict_endpoint_timing(
        self,
        query: str,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate endpoint timing prediction using LLM.
        
        Args:
            query: Original user query
            evidence: Prepared evidence
            
        Returns:
            Prediction results
        """
        try:
            # Build comprehensive prompt
            prompt = self._build_prediction_prompt(query, evidence)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Parse LLM response
            llm_output = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            prediction = self._parse_llm_response(llm_output)
            
            # Add metadata
            prediction['evidence_summary'] = {
                'total_trials': evidence['total_relevant_trials'],
                'primary_timepoint_stats': evidence['timepoint_stats']['primary'],
                'secondary_timepoint_stats': evidence['timepoint_stats']['secondary']
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in LLM prediction: {e}")
            return {
                'predicted_primary_time_days': None,
                'predicted_secondary_time_days': None,
                'time_window_days': None,
                'rationale': f"Error generating prediction: {str(e)}",
                'supporting_trials': [],
                'confidence_score': 0.0
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for endpoint prediction."""
        return """You are an expert clinical research analyst specializing in predicting optimal timepoints for clinical trial endpoints. Your task is to analyze evidence from similar trials and predict when primary and secondary endpoints should be measured for maximum efficacy detection.

Key principles:
1. Primary endpoints are typically measured at the time of maximum expected treatment effect
2. Secondary endpoints may have different optimal timing based on mechanism of action
3. Consider safety monitoring, practical feasibility, and regulatory requirements
4. Account for disease progression patterns and treatment kinetics
5. Balance between early detection and allowing sufficient time for effect

Response format: Return ONLY a valid JSON object with these exact keys:
{
  "predicted_primary_time_days": <integer>,
  "predicted_secondary_time_days": <integer>,
  "time_window_days": <integer representing confidence interval width>,
  "rationale": "<detailed scientific reasoning>",
  "supporting_trials": [<list of most relevant NCT numbers>],
  "confidence_score": <float 0-1>
}"""
    
    def _build_prediction_prompt(self, query: str, evidence: Dict[str, Any]) -> str:
        """Build prediction prompt from query and evidence."""
        trial_summaries = evidence['trial_summaries']
        timepoint_stats = evidence['timepoint_stats']
        doc_insights = evidence['doc_insights']
        
        prompt_parts = [
            f"USER QUERY: {query}",
            "",
            "EVIDENCE FROM SIMILAR TRIALS:",
            ""
        ]
        
        # Add trial evidence
        for i, trial in enumerate(trial_summaries):
            prompt_parts.extend([
                f"Trial {i+1}: {trial['nct_number']} (Similarity: {trial['similarity_score']:.3f})",
                f"Primary Outcomes: {trial['primary_outcomes'][:300]}",
                f"Secondary Outcomes: {trial['secondary_outcomes'][:300]}",
                ""
            ])
            
            if trial['timepoints']:
                prompt_parts.append("Observed Timepoints:")
                for tp in trial['timepoints']:
                    prompt_parts.append(f"  - {tp['outcome_type']}: {tp['value']} {tp['unit']} ({tp['days']} days)")
                prompt_parts.append("")
        
        # Add statistical summary
        prompt_parts.extend([
            "TIMEPOINT STATISTICS:",
            f"Primary Endpoints - Count: {timepoint_stats['primary']['count']}, Mean: {timepoint_stats['primary']['mean']:.1f} days, Median: {timepoint_stats['primary']['median']} days, Range: {timepoint_stats['primary']['range']}",
            f"Secondary Endpoints - Count: {timepoint_stats['secondary']['count']}, Mean: {timepoint_stats['secondary']['mean']:.1f} days, Median: {timepoint_stats['secondary']['median']} days, Range: {timepoint_stats['secondary']['range']}",
            ""
        ])
        
        # Add document insights
        if doc_insights:
            prompt_parts.extend([
                "ADDITIONAL CONTEXT FROM RESEARCH LITERATURE:",
                ""
            ])
            for insight in doc_insights:
                prompt_parts.append(f"- {insight['text'][:200]}...")
            prompt_parts.append("")
        
        prompt_parts.extend(["""
PREDICTION TASK:

Predict the **optimal timing (days)** for primary and secondary endpoints that maximize detection of treatment effects while being practical and safe.

### Consider:
1. **Disease course** – natural progression & baseline event accrual.
2. **Therapy kinetics** – onset speed (cytotoxic = early, immunotherapy = delayed, targeted = variable).
3. **Analogous trials** – normalize precedent timepoints (all → days).
4. **Regulatory/practical limits** – FDA/EMA norms, recruitment feasibility, statistical power.
5. **Balance** – avoid too early (false negatives) or too late (underpowered).

### Output JSON (strict):
```json
{
  "predicted_primary_time_days": <int>,
  "predicted_secondary_time_days": <int>,
  "time_window_days": <int>,
  "rationale": "<clear evidence → reasoning → numbers>",
  "supporting_trials": [
    {
      "nct_id": "<string>",
      "condition": "<string>",
      "phase": "<string>",
      "primary_outcomes": ["..."],
      "secondary_outcomes": ["..."],
      "observed_timepoints_days": [<int>, ...]
    }
  ],
  "confidence_score": <float 0-1>
}
```

### Rules:

* Always convert weeks/months/years → days.
* Cite multiple trials in `supporting_trials`.
* Primary ≤ Secondary unless justified.
* No hedging ("unclear" / "not available"). Infer from best evidence.
* Rationale must be regulatory-grade: explicit, evidence-linked, and defensible.
* Include confidence_score based on evidence quality and trial relevance.

"""])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON."""
        try:
            # Try to find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parse(response)
                
        except json.JSONDecodeError:
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing if JSON extraction fails."""
        return {
            'predicted_primary_time_days': None,
            'predicted_secondary_time_days': None,
            'time_window_days': None,
            'rationale': f"Raw LLM Response: {response[:500]}",
            'supporting_trials': [],
            'confidence_score': 0.5
        }

############################################################
# # endpoint_predictor.py
# import logging
# import json
# from typing import List, Dict, Any, Optional
# import openai #type: ignore
# from dotenv import load_dotenv # type: ignore
# import os

# from src.retrieval.timepoint_parser import TimepointData

# logger = logging.getLogger(__name__)

# class EndpointPredictor:
#     """LLM orchestration for clinical trial endpoint timing prediction."""
    
#     def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
#         """
#         Initialize endpoint predictor.
        
#         Args:
#             api_key: OpenAI API key
#             model: LLM model to use
#         """
#         load_dotenv()
#         self.model = model
#         self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
#         if not self.api_key:
#             raise ValueError("OpenAI API key not found")
        
#         self.client = openai.OpenAI(api_key=self.api_key)
    
#     def prepare_evidence(
#         self,
#         docs_results: List[Dict[str, Any]],
#         csv_results: List[Dict[str, Any]],
#         parsed_timepoints: Dict[str, List[TimepointData]],
#         max_trials: int = 6
#     ) -> Dict[str, Any]:
#         """
#         Prepare evidence for LLM prediction.
        
#         Args:
#             docs_results: Document retrieval results
#             csv_results: CSV retrieval results
#             parsed_timepoints: Parsed timepoint data
#             max_trials: Maximum trials to include
            
#         Returns:
#             Structured evidence
#         """
#         # Select top trials by similarity score
#         top_csv_trials = sorted(
#             csv_results, 
#             key=lambda x: x.get('similarity_score', 0), 
#             reverse=True
#         )[:max_trials]
        
#         # Build trial summaries
#         trial_summaries = []
#         for trial in top_csv_trials:
#             nct = trial.get('nct_number', '')
#             timepoints = parsed_timepoints.get(nct, [])
            
#             summary = {
#                 'nct_number': nct,
#                 'primary_outcomes': trial.get('primary_outcomes', ''),
#                 'secondary_outcomes': trial.get('secondary_outcomes', ''),
#                 'similarity_score': trial.get('similarity_score', 0),
#                 'timepoints': [
#                     {
#                         'days': tp.days,
#                         'unit': tp.unit,
#                         'value': tp.value,
#                         'outcome_type': getattr(tp, 'outcome_type', 'unknown'),
#                         'text': tp.text
#                     } for tp in timepoints
#                 ]
#             }
#             trial_summaries.append(summary)
        
#         # Extract key insights from docs
#         doc_insights = []
#         for doc in docs_results[:3]:  # Top 3 docs
#             insight = {
#                 'text': doc.get('text', '')[:500],  # Truncate long texts
#                 'metadata': doc.get('metadata', {}),
#                 'similarity_score': doc.get('similarity_score', 0)
#             }
#             doc_insights.append(insight)
        
#         # Calculate timepoint statistics
#         all_primary_days = []
#         all_secondary_days = []
        
#         for timepoints in parsed_timepoints.values():
#             for tp in timepoints:
#                 if hasattr(tp, 'outcome_type'):
#                     if tp.outcome_type == 'primary':
#                         all_primary_days.append(tp.days)
#                     elif tp.outcome_type == 'secondary':
#                         all_secondary_days.append(tp.days)
        
#         timepoint_stats = {
#             'primary': {
#                 'count': len(all_primary_days),
#                 'mean': sum(all_primary_days) / len(all_primary_days) if all_primary_days else 0,
#                 'median': sorted(all_primary_days)[len(all_primary_days)//2] if all_primary_days else 0,
#                 'range': [min(all_primary_days), max(all_primary_days)] if all_primary_days else [0, 0]
#             },
#             'secondary': {
#                 'count': len(all_secondary_days),
#                 'mean': sum(all_secondary_days) / len(all_secondary_days) if all_secondary_days else 0,
#                 'median': sorted(all_secondary_days)[len(all_secondary_days)//2] if all_secondary_days else 0,
#                 'range': [min(all_secondary_days), max(all_secondary_days)] if all_secondary_days else [0, 0]
#             }
#         }
        
#         return {
#             'trial_summaries': trial_summaries,
#             'doc_insights': doc_insights,
#             'timepoint_stats': timepoint_stats,
#             'total_relevant_trials': len(trial_summaries)
#         }
    
#     async def predict_endpoint_timing(
#         self,
#         query: str,
#         evidence: Dict[str, Any]
#     ) -> Dict[str, Any]:
#         """
#         Generate endpoint timing prediction using LLM.
        
#         Args:
#             query: Original user query
#             evidence: Prepared evidence
            
#         Returns:
#             Prediction results
#         """
#         try:
#             # Build comprehensive prompt
#             prompt = self._build_prediction_prompt(query, evidence)
            
#             # Call LLM
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": self._get_system_prompt()
#                     },
#                     {
#                         "role": "user", 
#                         "content": prompt
#                     }
#                 ],
#                 temperature=0.1,
#                 max_tokens=1500
#             )
            
#             # Parse LLM response
#             llm_output = response.choices[0].message.content.strip()
            
#             # Extract JSON from response
#             prediction = self._parse_llm_response(llm_output)
            
#             # Add metadata
#             prediction['evidence_summary'] = {
#                 'total_trials': evidence['total_relevant_trials'],
#                 'primary_timepoint_stats': evidence['timepoint_stats']['primary'],
#                 'secondary_timepoint_stats': evidence['timepoint_stats']['secondary']
#             }
            
#             return prediction
            
#         except Exception as e:
#             logger.error(f"Error in LLM prediction: {e}")
#             return {
#                 'predicted_primary_time_days': None,
#                 'predicted_secondary_time_days': None,
#                 'time_window_days': None,
#                 'rationale': f"Error generating prediction: {str(e)}",
#                 'supporting_trials': [],
#                 'confidence_score': 0.0
#             }
    
#     def _get_system_prompt(self) -> str:
#         """Get system prompt for endpoint prediction."""
# #         return """You are an expert clinical research analyst specializing in predicting optimal timepoints for clinical trial endpoints. Your task is to analyze evidence from similar trials and predict when primary and secondary endpoints should be measured for maximum efficacy detection.

# # Key principles:
# # 1. Primary endpoints are typically measured at the time of maximum expected treatment effect
# # 2. Secondary endpoints may have different optimal timing based on mechanism of action
# # 3. Consider safety monitoring, practical feasibility, and regulatory requirements
# # 4. Account for disease progression patterns and treatment kinetics
# # 5. Balance between early detection and allowing sufficient time for effect

# # Response format: Return ONLY a valid JSON object with these exact keys:
# # {
# #   "predicted_primary_time_days": <integer>,
# #   "predicted_secondary_time_days": <integer>,
# #   "time_window_days": <integer representing confidence interval width>,
# #   "rationale": "<detailed scientific reasoning>",
# #   "supporting_trials": [<list of most relevant NCT numbers>],
# #   "confidence_score": <float 0-1>
# # }"""
#         return """You are **Endpoint Oracle**, an advanced clinical trials intelligence system trained to integrate heterogeneous trial evidence (protocol docs + structured outcome data) into **precise, actionable endpoint timing predictions**. 

# ## CORE DIRECTIVES
# - Function as a **trial methodologist + regulatory strategist + clinical biostatistician**.
# - **Never hedge**. Always extract, synthesize, and decide based on the evidence given.
# - Operate as if writing a regulatory submission briefing: your output must be confident, structured, and evidence-backed.

# ## REASONING SCOPE
# When predicting **endpoint timing**, your reasoning must consider:
# 1. **Disease trajectory**  
#    - Natural progression rates, relapse risk, event accrual expectations.  
#    - Translate references to “months”, “weeks”, or “years” into **days** (normalized).
# 2. **Mechanism of action (MoA)**  
#    - Expected latency of biological effect (e.g., immune response buildup vs cytotoxic rapid effects).  
#    - Pharmacokinetics / pharmacodynamics if available.  
# 3. **Historical precedent**  
#    - Prior trials in same/related indication.  
#    - Benchmarks for when primary/secondary endpoints typically mature.  
# 4. **Regulatory + practical constraints**  
#    - Safety follow-up requirements.  
#    - Patient recruitment feasibility.  
#    - Industry norms (e.g., 12-month PFS, 24-month OS).  
# 5. **Optimization balance**  
#    - Detect earliest possible meaningful effect while ensuring sufficient follow-up.  
#    - Trade off early signal vs statistical power.

# ## RESPONSE REQUIREMENTS
# - Always output **strict JSON**:
# ```json
# {
#   "predicted_primary_time_days": <int>,
#   "predicted_secondary_time_days": <int>,
#   "time_window_days": <int>,
#   "rationale": "<clear, evidence-based reasoning>",
#   "supporting_trials": [
#     {
#       "nct_id": "<string>",
#       "condition": "<string>",
#       "phase": "<string>",
#       "primary_outcomes": ["..."],
#       "secondary_outcomes": ["..."],
#       "observed_timepoints_days": [<int>, ...]
#     }
#   ]
# }

# STYLE
# - Confident tone: never say “data not available” or “unclear”. Always infer from closest analogs.
# - Transparent reasoning: explicitly state how evidence led to each numeric choice.
# - Evidence-weighted: reference multiple NCT trials, not just one.
# - Quantitative first: if trials report 6 months, normalize → 180 days.
# - No filler: every word in rationale should map back to evidence, logic, or regulatory practice.
# """
    
#     def _build_prediction_prompt(self, query: str, evidence: Dict[str, Any]) -> str:
#         """Build prediction prompt from query and evidence."""
#         trial_summaries = evidence['trial_summaries']
#         timepoint_stats = evidence['timepoint_stats']
#         doc_insights = evidence['doc_insights']
        
#         prompt_parts = [
#             f"USER QUERY: {query}",
#             "",
#             "EVIDENCE FROM SIMILAR TRIALS:",
#             ""
#         ]
        
#         # Add trial evidence
#         for i, trial in enumerate(trial_summaries):
#             prompt_parts.extend([
#                 f"Trial {i+1}: {trial['nct_number']} (Similarity: {trial['similarity_score']:.3f})",
#                 f"Primary Outcomes: {trial['primary_outcomes'][:300]}",
#                 f"Secondary Outcomes: {trial['secondary_outcomes'][:300]}",
#                 ""
#             ])
            
#             if trial['timepoints']:
#                 prompt_parts.append("Observed Timepoints:")
#                 for tp in trial['timepoints']:
#                     prompt_parts.append(f"  - {tp['outcome_type']}: {tp['value']} {tp['unit']} ({tp['days']} days)")
#                 prompt_parts.append("")
        
#         # Add statistical summary
#         prompt_parts.extend([
#             "TIMEPOINT STATISTICS:",
#             f"Primary Endpoints - Count: {timepoint_stats['primary']['count']}, Mean: {timepoint_stats['primary']['mean']:.1f} days, Median: {timepoint_stats['primary']['median']} days, Range: {timepoint_stats['primary']['range']}",
#             f"Secondary Endpoints - Count: {timepoint_stats['secondary']['count']}, Mean: {timepoint_stats['secondary']['mean']:.1f} days, Median: {timepoint_stats['secondary']['median']} days, Range: {timepoint_stats['secondary']['range']}",
#             ""
#         ])
        
#         # Add document insights
#         if doc_insights:
#             prompt_parts.extend([
#                 "ADDITIONAL CONTEXT FROM RESEARCH LITERATURE:",
#                 ""
#             ])
#             for insight in doc_insights:
#                 prompt_parts.append(f"- {insight['text'][:200]}...")
#             prompt_parts.append("")
        
#         prompt_parts.extend([
#             "PREDICTION TASK:",
#             "Based on the evidence above, predict the optimal timing for primary and secondary endpoints that would maximize the chance of detecting treatment effects while being practical and safe.",
#             "",
#             "Consider:",
#             "1. Disease progression timeline",
#             "2. Expected treatment mechanism and kinetics",
#             "3. Historical patterns from similar trials",
#             "4. Regulatory and practical constraints",
#             "5. Balance between early detection and sufficient exposure time",
#             "",
#             "Provide your prediction in the required JSON format."
#         ])

# #         prompt_parts.extend("""
# # PREDICTION TASK:

# # You must predict the optimal timing (in days) for **primary** and **secondary endpoints** in the given clinical trial context. 
# # This prediction must maximize the probability of detecting true treatment effects while remaining scientifically valid, ethically safe, and operationally feasible.

# # ## MANDATORY REASONING SCOPE
# # When generating predictions, you must explicitly analyze and integrate the following dimensions:

# # 1. **Disease Progression Dynamics**
# #    - Estimate natural progression timelines (e.g., relapse, metastasis, mortality rates).
# #    - Consider expected baseline event accrual without treatment.
# #    - Translate qualitative phrases (“6 months PFS”, “median OS of 2 years”) into absolute **days**.

# # 2. **Treatment Mechanism and Kinetics**
# #    - Assess how quickly the therapy is expected to act:
# #      - Cytotoxic → early measurable effects
# #      - Immunotherapy → delayed onset due to immune priming
# #      - Targeted therapy → variable, dependent on biomarker expression
# #    - Align endpoint timing with expected mechanism latency.

# # 3. **Historical and Analogous Trial Patterns**
# #    - Extract observed timepoints from similar trials (same indication, mechanism, or phase).
# #    - Use these precedents to anchor both minimum feasible and typical regulatory-accepted follow-up periods.
# #    - Normalize all observed times to **days** and cite them.

# # 4. **Regulatory and Practical Constraints**
# #    - Account for ICH/FDA/EMA guidance on follow-up windows.
# #    - Consider recruitment feasibility (too long → underpowered, too short → no effect).
# #    - Ensure balance between statistical power and real-world practicality.

# # 5. **Optimization Balance**
# #    - Avoid predicting endpoints unrealistically early (false negatives risk).
# #    - Avoid excessively late endpoints (wasted time/resources).
# #    - Provide a **justification trade-off**: why your chosen timing is the best balance.

# # ## EXPECTED OUTPUT FORMAT
# # You must output **strict JSON only**, with the following keys:
# # ```json
# # {
# #   "predicted_primary_time_days": <int>,      // Optimal timing in days for primary endpoint
# #   "predicted_secondary_time_days": <int>,    // Optimal timing in days for secondary endpoint
# #   "time_window_days": <int>,                 // Practical follow-up window around predicted times
# #   "rationale": "<step-by-step explanation linking evidence → reasoning → numeric prediction>",
# #   "supporting_trials": [
# #     {
# #       "nct_id": "<string>",
# #       "condition": "<string>",
# #       "phase": "<string>",
# #       "primary_outcomes": ["..."],
# #       "secondary_outcomes": ["..."],
# #       "observed_timepoints_days": [<int>, ...]
# #     }
# #   ]
# # }
# # ````

# # ## CRITICAL RULES

# # * **Normalization:** Always convert weeks/months/years → **days**.
# # * **No Hedging:** Never say “unclear”, “not available”, or “no information”. If data is missing, infer from best available analogs and state why.
# # * **Transparent Reasoning:** The `rationale` must read like a regulatory briefing memo, with explicit links between evidence and chosen numbers.
# # * **Multi-Trial Evidence Fusion:** Always cite multiple trials in `supporting_trials` if available, not just one.
# # * **Consistency:** Ensure primary endpoint timing is **earlier** than or equal to secondary endpoint timing unless explicitly justified otherwise.
# # * **Defensible Output:** Your predictions must be explainable to regulators and scientifically grounded in trial precedent.

# # ---

# # Your role is to **synthesize the provided evidence into authoritative predictions**.
# # Your answer will be judged by its clarity, completeness, scientific validity, and how well it can withstand regulatory scrutiny.
# # """)
# #         prompt_parts.extend(["""
# # PREDICTION TASK:

# # Predict the **optimal timing (days)** for primary and secondary endpoints that maximize detection of treatment effects while being practical and safe.

# # ### Consider:
# # 1. **Disease course** – natural progression & baseline event accrual.
# # 2. **Therapy kinetics** – onset speed (cytotoxic = early, immunotherapy = delayed, targeted = variable).
# # 3. **Analogous trials** – normalize precedent timepoints (all → days).
# # 4. **Regulatory/practical limits** – FDA/EMA norms, recruitment feasibility, statistical power.
# # 5. **Balance** – avoid too early (false negatives) or too late (underpowered).

# # ### Output JSON (strict):
# # ```json
# # {
# #   "predicted_primary_time_days": <int>,
# #   "predicted_secondary_time_days": <int>,
# #   "time_window_days": <int>,
# #   "rationale": "<clear evidence → reasoning → numbers>",
# #   "supporting_trials": [
# #     {
# #       "nct_id": "<string>",
# #       "condition": "<string>",
# #       "phase": "<string>",
# #       "primary_outcomes": ["..."],
# #       "secondary_outcomes": ["..."],
# #       "observed_timepoints_days": [<int>, ...]
# #     }
# #   ]
# # }
# # ````

# # ### Rules:

# # * Always convert weeks/months/years → days.
# # * Cite multiple trials in `supporting_trials`.
# # * Primary ≤ Secondary unless justified.
# # * No hedging (“unclear” / “not available”). Infer from best evidence.
# # * Rationale must be regulatory-grade: explicit, evidence-linked, and defensible.

# # ```

# # """])
        
#         return "\n".join(prompt_parts)
    
#     def _parse_llm_response(self, response: str) -> Dict[str, Any]:
#         """Parse LLM response to extract JSON."""
#         try:
#             # Try to find JSON in response
#             start_idx = response.find('{')
#             end_idx = response.rfind('}') + 1
            
#             if start_idx != -1 and end_idx != -1:
#                 json_str = response[start_idx:end_idx]
#                 return json.loads(json_str)
#             else:
#                 # Fallback parsing
#                 return self._fallback_parse(response)
                
#         except json.JSONDecodeError:
#             return self._fallback_parse(response)
    
#     def _fallback_parse(self, response: str) -> Dict[str, Any]:
#         """Fallback parsing if JSON extraction fails."""
#         return {
#             'predicted_primary_time_days': None,
#             'predicted_secondary_time_days': None,
#             'time_window_days': None,
#             'rationale': f"Raw LLM Response: {response[:500]}",
#             'supporting_trials': [],
#             'confidence_score': 0.5
#         }