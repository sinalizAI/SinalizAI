import requests

# Configurações do Firebase
firebase_config = {
    "apiKey": "***REMOVED***",
    "authDomain": "sinalizai.firebaseapp.com",
    "projectId": "sinalizai",
    "storageBucket": "sinalizai.firebasestorage.app",
    "messagingSenderId": "531928981509",
    "appId": "1:531928981509:web:91ee8b599aa29bbe25fc6c"
}

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
        print(f"Tentando enviar email de reset para: {email}")
        print(f"URL: {url}")
        
        response = requests.post(url, json=payload)
        
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Sucesso! Response: {response_data}")
            return True, response_data
        else:
            try:
                error_data = response.json()
                print(f"Erro retornado pelo Firebase: {error_data}")
                return False, error_data
            except:
                print(f"Erro ao decodificar resposta JSON. Status: {response.status_code}, Text: {response.text}")
                return False, {"error": {"message": "UNKNOWN_ERROR"}}
                
    except requests.exceptions.RequestException as e:
        print(f"Erro de rede na requisição: {e}")
        return False, {"error": {"message": "NETWORK_ERROR"}}
    except Exception as e:
        print(f"Erro inesperado na requisição de reset: {e}")
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
            return False, response.json()
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
