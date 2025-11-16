





import os
from typing import Dict, Any


class ConfigError(Exception):
    
    pass


def load_env_file(file_path: str = '.env') -> None:
    





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
                    

                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    os.environ[key] = value
    except Exception as e:
        raise ConfigError(f"Erro ao carregar arquivo .env: {e}")


def get_firebase_config() -> Dict[str, Any]:
    









    try:
        load_env_file()
    except ConfigError:

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
    









    try:
        load_env_file()
    except ConfigError:
        pass
        
    return os.getenv(key, default)



try:
    firebase_config = get_firebase_config()
except ConfigError as e:
    print(f"  AVISO: {e}")

    firebase_config = {
        'apiKey': '',
        'authDomain': '',
        'projectId': '',
        'storageBucket': '',
        'messagingSenderId': '',
        'appId': ''
    }