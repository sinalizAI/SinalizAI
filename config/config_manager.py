"""
Gerenciador de configuração seguro para o projeto SinalizAI.
Carrega as configurações a partir de variáveis de ambiente para manter
as credenciais sensíveis fora do código fonte.
"""

import os
from typing import Dict, Any


class ConfigError(Exception):
    """Exceção personalizada para erros de configuração."""
    pass


def load_env_file(file_path: str = '.env') -> None:
    """
    Carrega variáveis de ambiente a partir de um arquivo .env.
    
    Args:
        file_path: Caminho para o arquivo .env
    """
    if not os.path.exists(file_path):
        return
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove aspas se existirem
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    os.environ[key] = value
    except Exception as e:
        raise ConfigError(f"Erro ao carregar arquivo .env: {e}")


def get_firebase_config() -> Dict[str, Any]:
    """
    Obtém as configurações do Firebase a partir das variáveis de ambiente.
    
    Returns:
        Dict contendo as configurações do Firebase
        
    Raises:
        ConfigError: Se alguma variável obrigatória não for encontrada
    """
    # Primeiro tenta carregar o arquivo .env
    try:
        load_env_file()
    except ConfigError:
        # Se não conseguir carregar o .env, continua usando variáveis de ambiente do sistema
        pass
    
    required_vars = [
        'FIREBASE_API_KEY',
        'FIREBASE_AUTH_DOMAIN', 
        'FIREBASE_PROJECT_ID',
        'FIREBASE_STORAGE_BUCKET',
        'FIREBASE_MESSAGING_SENDER_ID',
        'FIREBASE_APP_ID'
    ]
    
    config = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Converte o nome da variável para o formato esperado pelo Firebase
            config_key = var.replace('FIREBASE_', '').lower()
            if config_key == 'messaging_sender_id':
                config_key = 'messagingSenderId'
            elif config_key == 'auth_domain':
                config_key = 'authDomain'
            elif config_key == 'project_id':
                config_key = 'projectId'
            elif config_key == 'storage_bucket':
                config_key = 'storageBucket'
            elif config_key == 'app_id':
                config_key = 'appId'
            elif config_key == 'api_key':
                config_key = 'apiKey'
                
            config[config_key] = value
    
    if missing_vars:
        raise ConfigError(
            f"Variáveis de ambiente obrigatórias não encontradas: {', '.join(missing_vars)}.\n"
            "Certifique-se de ter um arquivo .env com essas configurações ou "
            "defina as variáveis de ambiente do sistema."
        )
    
    return config


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Obtém um valor de configuração específico.
    
    Args:
        key: Chave da configuração
        default: Valor padrão caso a chave não seja encontrada
        
    Returns:
        Valor da configuração ou valor padrão
    """
    try:
        load_env_file()
    except ConfigError:
        pass
        
    return os.getenv(key, default)


# Valida as configurações ao importar o módulo
try:
    firebase_config = get_firebase_config()
except ConfigError as e:
    print(f"⚠️  AVISO: {e}")
    # Define configuração vazia para evitar erros de importação
    firebase_config = {
        'apiKey': '',
        'authDomain': '',
        'projectId': '',
        'storageBucket': '',
        'messagingSenderId': '',
        'appId': ''
    }