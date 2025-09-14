from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AgentBase(ABC):
    @abstractmethod
    def get_summary(self) -> str:
        """Return agent's capabilities summary"""
        pass
    
    @abstractmethod
    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query and return response"""
        pass