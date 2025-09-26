import requests
from datetime import datetime
from config.config_manager import firebase_config

FIRESTORE_URL = f"https://firestore.googleapis.com/v1/projects/{firebase_config['projectId']}/databases/(default)/documents"

def save_legal_acceptance(user_id, id_token):
    url = f"{FIRESTORE_URL}/legal_acceptances?documentId={user_id}"
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "fields": {
            "accepted_terms_of_service": {"booleanValue": True},
            "accepted_privacy_policy": {"booleanValue": True},
            "accepted_at": {"timestampValue": datetime.utcnow().isoformat("T") + "Z"},
            "user_id": {"stringValue": user_id}
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return True, None
    else:
        try:
            error_message = response.json().get("error", {}).get("message", "Erro desconhecido")
            return False, error_message
        except:
            return False, "Erro desconhecido ao tentar salvar os dados."
