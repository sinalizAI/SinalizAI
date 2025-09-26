import requests
from config.config_manager import firebase_config

# URL base de autenticação Firebase
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts"

# Registrar novo usuário
def register(email, password):
    url = f"{FIREBASE_AUTH_URL}:signUp?key={firebase_config['apiKey']}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True, response.json()
    else:
        try:
            return False, response.json()
        except:
            return False, {"error": {"message": "UNKNOWN_ERROR"}}

# Fazer login
def login(email, password):
    url = f"{FIREBASE_AUTH_URL}:signInWithPassword?key={firebase_config['apiKey']}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True, response.json()
    else:
        try:
            return False, response.json()
        except:
            return False, {"error": {"message": "UNKNOWN_ERROR"}}

# Resetar senha
def reset_password(email):
    url = f"{FIREBASE_AUTH_URL}:sendOobCode?key={firebase_config['apiKey']}"
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": email
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            return True, response_data
        else:
            try:
                error_data = response.json()
                return False, error_data
            except:
                return False, {"error": {"message": "UNKNOWN_ERROR"}}
                
    except requests.exceptions.RequestException:
        return False, {"error": {"message": "NETWORK_ERROR"}}
    except Exception:
        return False, {"error": {"message": "UNKNOWN_ERROR"}}

# Alterar e-mail
def update_email(id_token, new_email):
    url = f"{FIREBASE_AUTH_URL}:update?key={firebase_config['apiKey']}"
    payload = {
        "idToken": id_token,
        "email": new_email,
        "returnSecureToken": True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return True, response.json()
    else:
        try:
            error_data = response.json()
            return False, error_data
        except:
            return False, {"error": {"message": "UNKNOWN_ERROR"}}

# Alterar nome (displayName)
def update_display_name(id_token, new_name):
    url = f"{FIREBASE_AUTH_URL}:update?key={firebase_config['apiKey']}"
    payload = {
        "idToken": id_token,
        "displayName": new_name,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True, response.json()
    else:
        try:
            return False, response.json()
        except:
            return False, {"error": {"message": "UNKNOWN_ERROR"}}
# Função combinada para atualizar perfil (nome e/ou email)
def update_profile(id_token, new_email=None, new_display_name=None):
    """Atualiza perfil do usuário - pode atualizar email e/ou nome em uma requisição"""
    url = f"{FIREBASE_AUTH_URL}:update?key={firebase_config['apiKey']}"
    payload = {
        "idToken": id_token,
        "returnSecureToken": True
    }
    
    # Adiciona campos que foram fornecidos
    if new_email:
        payload["email"] = new_email
    if new_display_name:
        payload["displayName"] = new_display_name
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return True, response.json()
    else:
        try:
            error_data = response.json()
            return False, error_data
        except:
            return False, {"error": {"message": "UNKNOWN_ERROR"}}
