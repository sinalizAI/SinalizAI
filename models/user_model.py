from typing import Optional, Dict, Any
from datetime import datetime


class User:
    def __init__(self, email: str, display_name: str, local_id: str, 
                 id_token: Optional[str] = None, email_verified: bool = False):
        self.email = email
        self.display_name = display_name
        self.local_id = local_id
        self.id_token = id_token
        self.email_verified = email_verified
        self.created_at = datetime.now()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(
            email=data.get('email', ''),
            display_name=data.get('displayName', ''),
            local_id=data.get('localId', ''),
            id_token=data.get('idToken'),
            email_verified=data.get('emailVerified', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'email': self.email,
            'displayName': self.display_name,
            'localId': self.local_id,
            'idToken': self.id_token,
            'emailVerified': self.email_verified
        }
    
    def validate_email(self) -> bool:
        if not self.email:
            return False
        return '@' in self.email and '.' in self.email.split('@')[1]
    
    def validate_display_name(self) -> bool:
        return bool(self.display_name and len(self.display_name.strip()) >= 3)
    
    def is_authenticated(self) -> bool:
        return bool(self.id_token)
    
    def get_first_name(self) -> str:
        if not self.display_name:
            return ""
        return self.display_name.split()[0]
    
    def __str__(self) -> str:
        return f"User(email={self.email}, name={self.display_name})"
    
    def __repr__(self) -> str:
        return self.__str__()