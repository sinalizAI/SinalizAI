import os
import requests
from typing import Optional


DEFAULT_BASE = os.environ.get('FUNCTIONS_BASE_URL')
if not DEFAULT_BASE:

    if os.environ.get('FIRESTORE_EMULATOR_HOST') or os.environ.get('FIREBASE_EMULATOR_HUB'):
        DEFAULT_BASE = 'http://127.0.0.1:5001/sinalizai-tcc-2025/us-central1/api'
    else:

        try:
            from config.config_manager import firebase_config
            project_id = firebase_config.get('projectId') or firebase_config.get('project_id')
        except Exception:
            project_id = os.environ.get('FIREBASE_PROJECT_ID') or os.environ.get('GCLOUD_PROJECT')

        if project_id:
            DEFAULT_BASE = f'https://us-central1-{project_id}.cloudfunctions.net/api'
        else:

            DEFAULT_BASE = 'https://us-central1-sinalizai.cloudfunctions.net/api'


def _post(path: str, data: dict, token: Optional[str] = None):
    url = DEFAULT_BASE.rstrip('/') + '/' + path.lstrip('/')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    resp = requests.post(url, json=data, headers=headers, timeout=15)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {'text': resp.text}


def register(email: str, password: str, displayName: str):

    return _post('register', {'email': email, 'password': password, 'displayName': displayName})


def login(email: str, password: str):
    return _post('login', {'email': email, 'password': password})


def send_feedback(user_email: str, user_name: str, subject: str, message: str):
    return _post('sendFeedback', {'user_email': user_email, 'user_name': user_name, 'subject': subject, 'message': message})


def reset_password(email: str):
    return _post('resetPassword', {'email': email})


def generate_verification(email: str, uid: str):
    raise NotImplementedError('generate_verification removed: token flow disabled')


def verify_token(uid: str, token: str):
    raise NotImplementedError('verify_token removed: token flow disabled')


def update_profile(token: str, newEmail: str = None, newDisplayName: str = None):
    return _post('updateProfile', {'newEmail': newEmail, 'newDisplayName': newDisplayName}, token=token)


def change_password(token: str, newPassword: str):
    return _post('changePassword', {'newPassword': newPassword}, token=token)


def delete_account(token: str):
    return _post('deleteAccount', {}, token=token)
