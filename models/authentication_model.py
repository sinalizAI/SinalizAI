import re
from typing import List, Tuple


class AuthenticationModel:
    MIN_PASSWORD_LENGTH = 6
    MIN_NAME_LENGTH = 3
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        if not email:
            return False, "Email é obrigatório"
        
        email = email.strip()
        
        if len(email) < 5:
            return False, "Email muito curto"
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            return False, "Formato de email inválido"
        
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        if not password:
            return False, "Senha é obrigatória"
        
        if len(password) < AuthenticationModel.MIN_PASSWORD_LENGTH:
            return False, f"Senha deve ter pelo menos {AuthenticationModel.MIN_PASSWORD_LENGTH} caracteres"
        
        if not re.search(r'[A-Z]', password):
            return False, "Senha deve conter pelo menos uma letra maiúscula"
        
        if not re.search(r'[a-z]', password):
            return False, "Senha deve conter pelo menos uma letra minúscula"
        
        if not re.search(r'\d', password):
            return False, "Senha deve conter pelo menos um número"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Senha deve conter pelo menos um símbolo especial"
        
        return True, ""
    
    @staticmethod
    def validate_display_name(name: str) -> Tuple[bool, str]:
        if not name:
            return False, "Nome é obrigatório"
        
        name = name.strip()
        
        if len(name) < AuthenticationModel.MIN_NAME_LENGTH:
            return False, f"Nome deve ter pelo menos {AuthenticationModel.MIN_NAME_LENGTH} caracteres"
        
        if not re.match(r'^[a-zA-ZÀ-ÿ\s\'-]+$', name):
            return False, "Nome contém caracteres inválidos"
        
        return True, ""
    
    @staticmethod
    def validate_registration_data(email: str, password: str, confirm_password: str, name: str) -> List[str]:
        errors = []
        
        email_valid, email_error = AuthenticationModel.validate_email(email)
        if not email_valid:
            errors.append(email_error)
        
        password_valid, password_error = AuthenticationModel.validate_password(password)
        if not password_valid:
            errors.append(password_error)
        
        if password != confirm_password:
            errors.append("As senhas não coincidem")
        
        name_valid, name_error = AuthenticationModel.validate_display_name(name)
        if not name_valid:
            errors.append(name_error)
        
        return errors
    
    @staticmethod
    def validate_login_data(email: str, password: str) -> List[str]:
        errors = []
        
        if not email:
            errors.append("Email é obrigatório")
        
        if not password:
            errors.append("Senha é obrigatória")
        
        return errors