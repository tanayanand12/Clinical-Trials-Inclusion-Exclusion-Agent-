# timepoint_parser.py
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TimepointData:
    """Data class for parsed timepoint information."""
    value: int
    unit: str
    days: int
    text: str
    confidence: float

class TimepointParser:
    """Parser for extracting timing information from clinical outcome measures."""
    
    def __init__(self):
        """Initialize timepoint parser with regex patterns."""
        # Time unit patterns with conversion to days
        self.time_patterns = {
            'days': {'pattern': r'(\d+)\s*days?', 'multiplier': 1},
            'weeks': {'pattern': r'(\d+)\s*(?:weeks?|wks?)', 'multiplier': 7},
            'months': {'pattern': r'(\d+)\s*(?:months?|mos?)', 'multiplier': 30},
            'years': {'pattern': r'(\d+)\s*(?:years?|yrs?)', 'multiplier': 365},
            'hours': {'pattern': r'(\d+)\s*(?:hours?|hrs?)', 'multiplier': 1/24},
            'minutes': {'pattern': r'(\d+)\s*(?:minutes?|mins?)', 'multiplier': 1/1440}
        }
        
        # Common clinical timepoint patterns
        self.clinical_patterns = [
            r'at\s+(\d+)\s*(days?|weeks?|months?|years?)',
            r'after\s+(\d+)\s*(days?|weeks?|months?|years?)',
            r'(\d+)\s*-\s*(day|week|month|year)',
            r'day\s+(\d+)',
            r'week\s+(\d+)',
            r'month\s+(\d+)',
            r'(\d+)\s*d(?:ay)?s?\b',
            r'(\d+)\s*w(?:eek)?s?\b',
            r'(\d+)\s*m(?:onth)?s?\b',
            r'baseline.*?(\d+)\s*(days?|weeks?|months?|years?)',
            r'follow[-\s]?up.*?(\d+)\s*(days?|weeks?|months?|years?)',
            r'post[-\s]?treatment.*?(\d+)\s*(days?|weeks?|months?|years?)',
            r'(\d+)\s*(?:days?|weeks?|months?|years?)\s*(?:post|after|follow)',
        ]
    
    def parse_timepoints(self, text: str) -> List[TimepointData]:
        """
        Parse timepoints from outcome measure text.
        
        Args:
            text: Outcome measure text
            
        Returns:
            List of parsed timepoints
        """
        if not text or text.strip() == '':
            return []
        
        timepoints = []
        text_lower = text.lower()
        
        # Try each time unit pattern
        for unit, config in self.time_patterns.items():
            pattern = config['pattern']
            multiplier = config['multiplier']
            
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    value = int(match.group(1))
                    days = int(value * multiplier)
                    
                    timepoint = TimepointData(
                        value=value,
                        unit=unit,
                        days=days,
                        text=match.group(0),
                        confidence=0.8
                    )
                    timepoints.append(timepoint)
                except (ValueError, IndexError):
                    continue
        
        # Try clinical patterns
        for pattern in self.clinical_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match.groups()) >= 2:
                        value = int(match.group(1))
                        unit = match.group(2).rstrip('s')  # Remove plural
                        
                        # Convert to days
                        multiplier = self._get_unit_multiplier(unit)
                        days = int(value * multiplier)
                        
                        timepoint = TimepointData(
                            value=value,
                            unit=unit,
                            days=days,
                            text=match.group(0),
                            confidence=0.9
                        )
                        timepoints.append(timepoint)
                except (ValueError, IndexError):
                    continue
        
        # Remove duplicates and sort
        unique_timepoints = self._deduplicate_timepoints(timepoints)
        return sorted(unique_timepoints, key=lambda x: x.days)
    
    def parse_multiple_outcomes(self, csv_results: List[Dict[str, Any]]) -> Dict[str, List[TimepointData]]:
        """
        Parse timepoints from multiple outcome measures (handles split documents).
        
        Args:
            csv_results: List of CSV results with outcome measures
            
        Returns:
            Dictionary mapping NCT numbers to timepoints
        """
        parsed_outcomes = {}
        
        for result in csv_results:
            nct_number = result.get('nct_number', '')
            if not nct_number:
                continue
            
            # Initialize if first time seeing this NCT
            if nct_number not in parsed_outcomes:
                parsed_outcomes[nct_number] = []
            
            # Parse based on outcome type (for split documents)
            outcome_type = result.get('outcome_type', 'unknown')
            outcome_content = result.get('outcome_content', '')
            
            if outcome_content and outcome_content != 'nan':
                timepoints = self.parse_timepoints(outcome_content)
                for tp in timepoints:
                    tp.outcome_type = outcome_type
                parsed_outcomes[nct_number].extend(timepoints)
        
        return parsed_outcomes
    
    def get_summary_timepoints(self, parsed_outcomes: Dict[str, List[TimepointData]]) -> Dict[str, Any]:
        """
        Get summary statistics of timepoints.
        
        Args:
            parsed_outcomes: Parsed timepoints by NCT
            
        Returns:
            Summary statistics
        """
        all_primary = []
        all_secondary = []
        all_other = []
        
        for timepoints in parsed_outcomes.values():
            for tp in timepoints:
                if hasattr(tp, 'outcome_type'):
                    if tp.outcome_type == 'primary':
                        all_primary.append(tp.days)
                    elif tp.outcome_type == 'secondary':
                        all_secondary.append(tp.days)
                    elif tp.outcome_type == 'other':
                        all_other.append(tp.days)
        
        summary = {
            'primary_timepoints': {
                'count': len(all_primary),
                'mean_days': sum(all_primary) / len(all_primary) if all_primary else 0,
                'median_days': sorted(all_primary)[len(all_primary)//2] if all_primary else 0,
                'min_days': min(all_primary) if all_primary else 0,
                'max_days': max(all_primary) if all_primary else 0
            },
            'secondary_timepoints': {
                'count': len(all_secondary),
                'mean_days': sum(all_secondary) / len(all_secondary) if all_secondary else 0,
                'median_days': sorted(all_secondary)[len(all_secondary)//2] if all_secondary else 0,
                'min_days': min(all_secondary) if all_secondary else 0,
                'max_days': max(all_secondary) if all_secondary else 0
            },
            'total_trials': len(parsed_outcomes)
        }
        
        return summary
    
    def _get_unit_multiplier(self, unit: str) -> float:
        """Get multiplier to convert unit to days."""
        unit_lower = unit.lower().rstrip('s')
        
        multipliers = {
            'day': 1,
            'week': 7,
            'month': 30,
            'year': 365,
            'hour': 1/24,
            'minute': 1/1440
        }
        
        return multipliers.get(unit_lower, 1)
    
    def _deduplicate_timepoints(self, timepoints: List[TimepointData]) -> List[TimepointData]:
        """Remove duplicate timepoints."""
        seen = set()
        unique_timepoints = []
        
        for tp in timepoints:
            key = (tp.days, tp.unit)
            if key not in seen:
                seen.add(key)
                unique_timepoints.append(tp)
        
        return unique_timepoints