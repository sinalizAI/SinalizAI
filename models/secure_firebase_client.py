"""
Cliente HTTP Seguro para Firebase - SinalizAI
==============================================

Este módulo implementa um cliente HTTP mais seguro para comunicação com Firebase,
incluindo validações de certificado, logging seguro e tratamento de erros robusto.
"""

import requests
import logging
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urlencode
import time
from config.config_manager import firebase_config


class SecureFirebaseClient:
    """
    Cliente HTTP seguro para comunicação com Firebase.
    
    Características de segurança:
    - Validação de certificados SSL/TLS
    - Logging seguro (não expõe API keys)
    - Rate limiting básico
    - Timeout configurável
    - Tratamento robusto de erros
    """
    
    def __init__(self):
        self.base_url = "https://identitytoolkit.googleapis.com/v1/accounts"
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Mínimo 100ms entre requisições
        
        # Configurar sessão com segurança aprimorada
        self.session.verify = True  # Sempre validar certificados
        self.session.timeout = 30   # Timeout de 30 segundos
        
        # Headers padrão de segurança
        self.session.headers.update({
            'User-Agent': 'SinalizAI/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Configurar logging seguro
        self._setup_secure_logging()
    
    def _setup_secure_logging(self):
        """Configura logging que não expõe informações sensíveis."""
        self.logger = logging.getLogger(__name__)
        
        # Se não há handlers, adiciona um básico
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _rate_limit(self):
        """Implementa rate limiting básico."""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        if time_diff < self.min_request_interval:
            sleep_time = self.min_request_interval - time_diff
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _log_request(self, method: str, endpoint: str, success: bool, status_code: int = None):
        """Log seguro de requisições (não expõe API keys)."""
        if success:
            self.logger.info(f"✅ {method} {endpoint} - Status: {status_code}")
        else:
            self.logger.warning(f"❌ {method} {endpoint} - Status: {status_code}")
    
    def _make_secure_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        timeout: int = 30
    ) -> Tuple[bool, Dict]:
        """
        Faz uma requisição HTTP segura.
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint da API (ex: 'signUp', 'signInWithPassword')
            data: Dados para enviar no body da requisição
            timeout: Timeout em segundos
        
        Returns:
            Tuple (success, response_data)
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            # Construir URL com API key
            url = f"{self.base_url}:{endpoint}"
            params = {"key": firebase_config['apiKey']}
            
            # Preparar dados
            json_data = data or {}
            
            # Fazer requisição
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                timeout=timeout,
                verify=True  # Sempre validar certificados
            )
            
            # Log da requisição (sem expor dados sensíveis)
            self._log_request(method, endpoint, response.status_code == 200, response.status_code)
            
            # Processar resposta
            if response.status_code == 200:
                return True, response.json()
            else:
                error_data = {}
                try:
                    error_data = response.json()
                except:
                    error_data = {
                        "error": {
                            "message": f"HTTP_{response.status_code}",
                            "code": response.status_code
                        }
                    }
                return False, error_data
                
        except requests.exceptions.SSLError as e:
            self.logger.error(f"❌ Erro SSL/TLS: {str(e)}")
            return False, {"error": {"message": "SSL_ERROR", "details": "Erro de certificado SSL"}}
            
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"❌ Erro de conexão: {str(e)}")
            return False, {"error": {"message": "CONNECTION_ERROR", "details": "Erro de conectividade"}}
            
        except requests.exceptions.Timeout as e:
            self.logger.error(f"❌ Timeout: {str(e)}")
            return False, {"error": {"message": "TIMEOUT_ERROR", "details": "Requisição expirou"}}
            
        except Exception as e:
            self.logger.error(f"❌ Erro inesperado: {str(e)}")
            return False, {"error": {"message": "UNKNOWN_ERROR", "details": str(e)}}
    
    def register(self, email: str, password: str) -> Tuple[bool, Dict]:
        """Registra novo usuário."""
        data = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        return self._make_secure_request("POST", "signUp", data)
    
    def login(self, email: str, password: str) -> Tuple[bool, Dict]:
        """Faz login de usuário."""
        data = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        return self._make_secure_request("POST", "signInWithPassword", data)
    
    def reset_password(self, email: str) -> Tuple[bool, Dict]:
        """Envia email de recuperação de senha."""
        data = {
            "requestType": "PASSWORD_RESET",
            "email": email
        }
        return self._make_secure_request("POST", "sendOobCode", data)
    
    def update_profile(
        self, 
        id_token: str, 
        new_email: Optional[str] = None,
        new_display_name: Optional[str] = None
    ) -> Tuple[bool, Dict]:
        """Atualiza perfil do usuário."""
        data = {
            "idToken": id_token,
            "returnSecureToken": True
        }
        
        if new_email:
            data["email"] = new_email
        if new_display_name:
            data["displayName"] = new_display_name
            
        return self._make_secure_request("POST", "update", data)


# Instância global do cliente seguro
_secure_client = None

def get_secure_client() -> SecureFirebaseClient:
    """Retorna uma instância singleton do cliente seguro."""
    global _secure_client
    if _secure_client is None:
        _secure_client = SecureFirebaseClient()
    return _secure_client