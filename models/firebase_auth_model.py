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
        return False, response.json()["error"]["message"]

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
        return False, response.json()["error"]["message"]

# Resetar senha
def reset_password(email):
    url = f"{FIREBASE_AUTH_URL}:sendOobCode?key={firebase_config['apiKey']}"
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": email
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True, "Email de recuperação enviado!"
    else:
        return False, response.json()["error"]["message"]

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
        return False, response.json()["error"]["message"]

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
        return False, response.json()["error"]["message"]
