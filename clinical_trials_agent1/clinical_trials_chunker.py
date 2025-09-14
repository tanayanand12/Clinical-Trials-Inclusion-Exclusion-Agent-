# clinical_trials_chunker.py
import logging
import re
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class ClinicalTrialsChunker:
    """
    Chunks clinical trial data into meaningful segments for vectorization.
    Handles different sections of clinical trial data with appropriate chunking strategies.
    """
    
    def __init__(self, max_chunk_size: int = 10000, overlap_size: int = 500):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap_size: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\/\%\<\>\=\+]', ' ', text)
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_study_sections(self, study_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract meaningful sections from a clinical trial study.
        
        Args:
            study_data: Clinical trial study data from API
            
        Returns:
            Dict mapping section names to content
        """
        sections = {}
        
        try:
            protocol_section = study_data.get('protocolSection', {})
            
            # Basic identification
            identification = protocol_section.get('identificationModule', {})
            sections['title'] = self.clean_text(identification.get('briefTitle', ''))
            sections['official_title'] = self.clean_text(identification.get('officialTitle', ''))
            sections['nct_id'] = identification.get('nctId', '')
            sections['acronym'] = identification.get('acronym', '')
            
            # Study description
            description = protocol_section.get('descriptionModule', {})
            sections['brief_summary'] = self.clean_text(description.get('briefSummary', ''))
            sections['detailed_description'] = self.clean_text(description.get('detailedDescription', ''))
            
            # Conditions and interventions
            conditions = protocol_section.get('conditionsModule', {})
            sections['conditions'] = ', '.join(conditions.get('conditions', []))
            
            arms_interventions = protocol_section.get('armsInterventionsModule', {})
            interventions = arms_interventions.get('interventions', [])
            intervention_texts = []
            for intervention in interventions:
                intervention_text = f"{intervention.get('type', '')} - {intervention.get('name', '')}: {intervention.get('description', '')}"
                intervention_texts.append(self.clean_text(intervention_text))
            sections['interventions'] = ' | '.join(intervention_texts)
            
            # Outcomes
            outcomes = protocol_section.get('outcomesModule', {})
            primary_outcomes = outcomes.get('primaryOutcomes', [])
            secondary_outcomes = outcomes.get('secondaryOutcomes', [])
            
            primary_text = []
            for outcome in primary_outcomes:
                outcome_text = f"Primary: {outcome.get('measure', '')} - {outcome.get('description', '')} (Timeframe: {outcome.get('timeFrame', '')})"
                primary_text.append(self.clean_text(outcome_text))
            
            secondary_text = []
            for outcome in secondary_outcomes:
                outcome_text = f"Secondary: {outcome.get('measure', '')} - {outcome.get('description', '')} (Timeframe: {outcome.get('timeFrame', '')})"
                secondary_text.append(self.clean_text(outcome_text))
            
            sections['primary_outcomes'] = ' | '.join(primary_text)
            sections['secondary_outcomes'] = ' | '.join(secondary_text)
            
            # Eligibility criteria
            eligibility = protocol_section.get('eligibilityModule', {})
            sections['eligibility_criteria'] = self.clean_text(eligibility.get('eligibilityCriteria', ''))
            sections['healthy_volunteers'] = str(eligibility.get('healthyVolunteers', False))
            sections['sex'] = eligibility.get('sex', '')
            sections['minimum_age'] = eligibility.get('minimumAge', '')
            sections['maximum_age'] = eligibility.get('maximumAge', '')
            
            # Study design
            design = protocol_section.get('designModule', {})
            sections['study_type'] = design.get('studyType', '')
            sections['phases'] = ', '.join(design.get('phases', []))
            design_info = design.get('designInfo', {})
            sections['allocation'] = design_info.get('allocation', '')
            sections['intervention_model'] = design_info.get('interventionModel', '')
            sections['primary_purpose'] = design_info.get('primaryPurpose', '')
            
            # Status information
            status = protocol_section.get('statusModule', {})
            sections['overall_status'] = status.get('overallStatus', '')
            sections['start_date'] = status.get('startDateStruct', {}).get('date', '')
            sections['completion_date'] = status.get('completionDateStruct', {}).get('date', '')
            
            # Location information
            contacts_locations = protocol_section.get('contactsLocationsModule', {})
            locations = contacts_locations.get('locations', [])
            location_texts = []
            for location in locations:
                location_text = f"{location.get('facility', '')} - {location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')} (Status: {location.get('status', '')})"
                location_texts.append(self.clean_text(location_text))
            sections['locations'] = ' | '.join(location_texts)
            
            # Sponsor information
            sponsor = protocol_section.get('sponsorCollaboratorsModule', {})
            lead_sponsor = sponsor.get('leadSponsor', {})
            sections['lead_sponsor'] = f"{lead_sponsor.get('name', '')} ({lead_sponsor.get('class', '')})"
            
            collaborators = sponsor.get('collaborators', [])
            collaborator_texts = [f"{collab.get('name', '')} ({collab.get('class', '')})" for collab in collaborators]
            sections['collaborators'] = ' | '.join(collaborator_texts)
            
        except Exception as e:
            logger.error(f"Error extracting study sections: {e}")
            
        return sections
    
    def create_semantic_chunks(self, sections: Dict[str, str], study_id: str) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from study sections.
        
        Args:
            sections: Dictionary of study sections
            study_id: Study identifier
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Create overview chunk
        overview_content = f"""
        Study: {sections.get('title', '')}
        NCT ID: {sections.get('nct_id', '')}
        Status: {sections.get('overall_status', '')}
        Study Type: {sections.get('study_type', '')}
        Phases: {sections.get('phases', '')}
        Conditions: {sections.get('conditions', '')}
        Brief Summary: {sections.get('brief_summary', '')}
        """.strip()
        
        chunks.append({
            'content': self.clean_text(overview_content),
            'chunk_type': 'overview',
            'study_id': study_id,
            'section': 'overview'
        })
        
        # Create intervention chunk
        if sections.get('interventions'):
            intervention_content = f"""
            Study: {sections.get('title', '')}
            Interventions: {sections.get('interventions', '')}
            Primary Purpose: {sections.get('primary_purpose', '')}
            Intervention Model: {sections.get('intervention_model', '')}
            Allocation: {sections.get('allocation', '')}
            """.strip()
            
            chunks.append({
                'content': self.clean_text(intervention_content),
                'chunk_type': 'intervention',
                'study_id': study_id,
                'section': 'interventions'
            })
        
        # Create outcomes chunk
        outcomes_content = f"""
        Study: {sections.get('title', '')}
        Primary Outcomes: {sections.get('primary_outcomes', '')}
        Secondary Outcomes: {sections.get('secondary_outcomes', '')}
        """.strip()
        
        if sections.get('primary_outcomes') or sections.get('secondary_outcomes'):
            chunks.append({
                'content': self.clean_text(outcomes_content),
                'chunk_type': 'outcomes',
                'study_id': study_id,
                'section': 'outcomes'
            })
        
        # Create eligibility chunk
        if sections.get('eligibility_criteria'):
            eligibility_content = f"""
            Study: {sections.get('title', '')}
            Eligibility Criteria: {sections.get('eligibility_criteria', '')}
            Sex: {sections.get('sex', '')}
            Age Range: {sections.get('minimum_age', '')} to {sections.get('maximum_age', '')}
            Healthy Volunteers: {sections.get('healthy_volunteers', '')}
            """.strip()
            
            # Split large eligibility criteria into smaller chunks
            if len(eligibility_content) > self.max_chunk_size:
                eligibility_chunks = self.split_large_text(eligibility_content, 'eligibility', study_id)
                chunks.extend(eligibility_chunks)
            else:
                chunks.append({
                    'content': self.clean_text(eligibility_content),
                    'chunk_type': 'eligibility',
                    'study_id': study_id,
                    'section': 'eligibility'
                })
        
        # Create detailed description chunk if available
        if sections.get('detailed_description'):
            detailed_content = f"""
            Study: {sections.get('title', '')}
            Detailed Description: {sections.get('detailed_description', '')}
            """.strip()
            
            if len(detailed_content) > self.max_chunk_size:
                detailed_chunks = self.split_large_text(detailed_content, 'detailed_description', study_id)
                chunks.extend(detailed_chunks)
            else:
                chunks.append({
                    'content': self.clean_text(detailed_content),
                    'chunk_type': 'detailed_description',
                    'study_id': study_id,
                    'section': 'detailed_description'
                })
        
        # Create location chunk
        if sections.get('locations'):
            location_content = f"""
            Study: {sections.get('title', '')}
            Study Locations: {sections.get('locations', '')}
            Lead Sponsor: {sections.get('lead_sponsor', '')}
            Collaborators: {sections.get('collaborators', '')}
            """.strip()
            
            chunks.append({
                'content': self.clean_text(location_content),
                'chunk_type': 'location',
                'study_id': study_id,
                'section': 'locations'
            })
        
        return chunks
    
    def split_large_text(self, text: str, chunk_type: str, study_id: str) -> List[Dict[str, Any]]:
        """
        Split large text into smaller chunks with overlap.
        
        Args:
            text: Text to split
            chunk_type: Type of chunk
            study_id: Study identifier
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'content': self.clean_text(current_chunk),
                    'chunk_type': f"{chunk_type}_{chunk_index}",
                    'study_id': study_id,
                    'section': chunk_type
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.overlap_size//5:] if len(words) > self.overlap_size//5 else words
                current_chunk = " ".join(overlap_words) + " " + sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'content': self.clean_text(current_chunk),
                'chunk_type': f"{chunk_type}_{chunk_index}",
                'study_id': study_id,
                'section': chunk_type
            })
        
        return chunks
    
    def chunk_clinical_trials_data(self, clinical_trials_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main method to chunk clinical trials data.
        
        Args:
            clinical_trials_data: Raw clinical trials data from API
            
        Returns:
            List of chunks ready for vectorization
        """
        all_chunks = []
        
        try:
            studies = clinical_trials_data.get('studies', [])
            
            for study in studies:
                study_id = study.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')
                
                # Extract sections from study
                sections = self.extract_study_sections(study)
                
                # Create semantic chunks
                study_chunks = self.create_semantic_chunks(sections, study_id)
                
                all_chunks.extend(study_chunks)
                
            logger.info(f"Created {len(all_chunks)} chunks from {len(studies)} studies")
            
        except Exception as e:
            logger.error(f"Error chunking clinical trials data: {e}")
            
        return all_chunks
    