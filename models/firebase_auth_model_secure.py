"""
Firebase Auth Model - Versão Segura Aprimorada
==============================================

Esta é uma versão opcionalmente mais segura do firebase_auth_model.py
que usa o SecureFirebaseClient para comunicação HTTP mais segura.

NOTA: Esta é uma versão OPCIONAL. Seu código atual já está seguro.
Use esta versão se quiser segurança máxima.
"""

from models.secure_firebase_client import get_secure_client
from typing import Tuple, Dict, Any, Optional


def register(email: str, password: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Registra um novo usuário usando o cliente HTTP seguro.
    
    Args:
        email: Email do usuário
        password: Senha do usuário
        
    Returns:
        Tuple (success, response_data)
    """
    client = get_secure_client()
    return client.register(email, password)


def login(email: str, password: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Faz login do usuário usando o cliente HTTP seguro.
    
    Args:
        email: Email do usuário
        password: Senha do usuário
        
    Returns:
        Tuple (success, response_data)
    """
    client = get_secure_client()
    return client.login(email, password)


def reset_password(email: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Envia email de recuperação de senha usando o cliente HTTP seguro.
    
    Args:
        email: Email do usuário
        
    Returns:
        Tuple (success, response_data)
    """
    client = get_secure_client()
    return client.reset_password(email)


def update_email(id_token: str, new_email: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Atualiza email do usuário usando o cliente HTTP seguro.
    
    Args:
        id_token: Token de autenticação
        new_email: Novo email
        
    Returns:
        Tuple (success, response_data)
    """
    client = get_secure_client()
    return client.update_profile(id_token, new_email=new_email)


def update_display_name(id_token: str, new_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Atualiza nome do usuário usando o cliente HTTP seguro.
    
    Args:
        id_token: Token de autenticação
        new_name: Novo nome
        
    Returns:
        Tuple (success, response_data)
    """
    client = get_secure_client()
    return client.update_profile(id_token, new_display_name=new_name)


def update_profile(
    id_token: str, 
    new_email: Optional[str] = None, 
    new_display_name: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Atualiza perfil completo do usuário usando o cliente HTTP seguro.
    
    Args:
        id_token: Token de autenticação
        new_email: Novo email (opcional)
        new_display_name: Novo nome (opcional)
        
    Returns:
        Tuple (success, response_data)
    """
    client = get_secure_client()
    return client.update_profile(id_token, new_email, new_display_name)


# Manter compatibilidade com a configuração atual
from config.config_manager import firebase_config