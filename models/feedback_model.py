from typing import List, Dict, Any
from datetime import datetime


class FeedbackModel:
    def __init__(self, user_email: str, user_name: str, subject: str, message: str):
        self.user_email = user_email
        self.user_name = user_name
        self.subject = subject
        self.message = message
        self.created_at = datetime.now()
    
    def validate(self) -> List[str]:
        errors = []
        
        if not self.user_name or len(self.user_name.strip()) < 3:
            errors.append("Nome deve ter pelo menos 3 caracteres")
        
        if not self.user_email:
            errors.append("Email é obrigatório")
        elif not self._is_valid_email(self.user_email):
            errors.append("Email inválido")
        
        if not self.subject or len(self.subject.strip()) < 5:
            errors.append("Motivo do contato deve ter pelo menos 5 caracteres")
        
        if not self.message or len(self.message.strip()) < 10:
            errors.append("Mensagem deve ter pelo menos 10 caracteres")
        
        if self.message and len(self.message) > 1500:
            errors.append("Mensagem deve ter no máximo 1500 caracteres")
        
        return errors
    
    def is_valid(self) -> bool:
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'user_email': self.user_email,
            'user_name': self.user_name,
            'subject': self.subject,
            'message': self.message
        }
    
    def _is_valid_email(self, email: str) -> bool:
        if not email or len(email.strip()) < 5:
            return False
        
        email = email.strip()
        if "@" not in email or "." not in email:
            return False
        
        parts = email.split("@")
        if len(parts) != 2:
            return False
        
        local, domain = parts
        if not local or not domain or "." not in domain:
            return False
        
        return True
    
    def get_character_count(self) -> int:
        return len(self.message) if self.message else 0
    
    def __str__(self) -> str:
        return f"Feedback(from={self.user_email}, subject={self.subject[:30]}...)"