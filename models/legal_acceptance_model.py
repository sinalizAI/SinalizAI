import requests
from datetime import datetime
from config.config_manager import firebase_config

FIRESTORE_URL = f"https://firestore.googleapis.com/v1/projects/{firebase_config['projectId']}/databases/(default)/documents"

def save_legal_acceptance(user_id, id_token):
    if not user_id or not id_token:
        return False, 'user_id ou id_token ausente'

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

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        return False, f'Falha de conexão ao salvar aceite: {e}'

    if 200 <= response.status_code < 300:
        return True, None
    else:
        try:
            error_message = response.json().get("error", {}).get("message", "Erro desconhecido")
            return False, error_message
        except Exception:
            return False, "Erro desconhecido ao tentar salvar os dados."

        doc_url = f"{FIRESTORE_URL}/legal_acceptances/{user_id}?currentDocument.exists=true"
        try:
            patch_resp = requests.patch(doc_url, headers=headers, json=payload, timeout=10)
            if 200 <= patch_resp.status_code < 300:
                return True, None
            try:
                pbody = patch_resp.json()
                return False, f'Erro ao atualizar aceite: {pbody.get("error", {}).get("message") or str(pbody)}'
            except Exception:
                return False, f'Erro ao atualizar aceite: status {patch_resp.status_code}'
        except Exception as e:
            return False, f'Falha de conexão ao atualizar aceite: {e}'

    # Handle common auth errors explicitly for clarity in the app
    if response.status_code in (401, 403):
        return False, f'Autorização falhou ao gravar aceite (status {response.status_code}): verifique idToken/permssões.'

    return False, f'Erro ao salvar aceite: {error_message or response.text or "unknown"}'
