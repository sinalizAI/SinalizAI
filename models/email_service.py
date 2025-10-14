import os
import requests
import json

try:
    # prefer usar python-dotenv se estiver instalado
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


class EmailService:
    """Serviço para envio de emails via EmailJS

    Carrega configurações de `.env_emailJS` primeiro. Se não existir,
    tenta `.env`. As variáveis esperadas são:
      - EMAILJS_SERVICE_ID
      - EMAILJS_TEMPLATE_ID
      - EMAILJS_PUBLIC_KEY
      - FEEDBACK_EMAIL

    Se as variáveis não existirem, valores hardcoded antigos são usados como
    fallback para manter compatibilidade.
    """

    # Carregar arquivo .env específico para EmailJS, com fallback para .env
    ENV_EMAILJS = os.path.join(os.getcwd(), '.env_emailJS')
    ENV_DEFAULT = os.path.join(os.getcwd(), '.env')

    if load_dotenv:
        # tenta carregar .env_emailJS primeiro
        if os.path.exists(ENV_EMAILJS):
            load_dotenv(ENV_EMAILJS, override=False)
        elif os.path.exists(ENV_DEFAULT):
            load_dotenv(ENV_DEFAULT, override=False)
    else:
        # fallback simples: parsear linha a linha se existir
        def _load_simple(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln or ln.startswith('#') or '=' not in ln:
                            continue
                        k, v = ln.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('\"').strip('\'')
                        if k not in os.environ:
                            os.environ[k] = v
            except FileNotFoundError:
                pass

        if os.path.exists(ENV_EMAILJS):
            _load_simple(ENV_EMAILJS)
        elif os.path.exists(ENV_DEFAULT):
            _load_simple(ENV_DEFAULT)

    # Valores default (fallback) mantidos para compatibilidade
    _DEFAULT_SERVICE_ID = "service_4vex1zl"
    _DEFAULT_TEMPLATE_ID = "template_9lyg2p5"
    _DEFAULT_PUBLIC_KEY = "AETagjMuO4_iJD8dj"
    _DEFAULT_FEEDBACK_EMAIL = "***EMAIL_REMOVED***"

    EMAILJS_CONFIG = {
        "service_id": os.environ.get('EMAILJS_SERVICE_ID', _DEFAULT_SERVICE_ID),
        "template_id": os.environ.get('EMAILJS_TEMPLATE_ID', _DEFAULT_TEMPLATE_ID),
        "public_key": os.environ.get('EMAILJS_PUBLIC_KEY', _DEFAULT_PUBLIC_KEY)
    }

    FEEDBACK_EMAIL = os.environ.get('FEEDBACK_EMAIL', _DEFAULT_FEEDBACK_EMAIL)
    
    @staticmethod
    def send_feedback(user_email, user_name, subject, message):
        """Envia feedback via EmailJS"""
        
        url = "https://api.emailjs.com/api/v1.0/email/send"
        
        # Parâmetros que correspondem ao template
        template_params = {
            "from_name": user_name,                    # {{from_name}}
            "from_email": user_email,                  # {{from_email}}
            "to_email": EmailService.FEEDBACK_EMAIL,   # {{to_email}}
            "subject": subject,                        # {{subject}}
            "message": message,                        # {{message}}
            "reply_to": user_email                     # {{reply_to}}
        }
        
        payload = {
            "service_id": EmailService.EMAILJS_CONFIG["service_id"],
            "template_id": EmailService.EMAILJS_CONFIG["template_id"],
            "user_id": EmailService.EMAILJS_CONFIG["public_key"],
            "template_params": template_params
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": "https://www.emailjs.com",
            "Referer": "https://www.emailjs.com/",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Feedback enviado com sucesso!"
                }
            else:
                return {
                    "success": False,
                    "message": f"Erro ao enviar feedback: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Erro de conexão: {str(e)}"
            }
    
    @staticmethod
    def validate_feedback_data(user_email, user_name, subject, message):
        """Valida os dados do feedback"""
        errors = []
        
        if not user_name or len(user_name.strip()) < 3:
            errors.append("Nome deve ter pelo menos 3 caracteres")
            
        # Validação de email mais robusta
        if not user_email:
            errors.append("Email é obrigatório")
        elif len(user_email.strip()) == 0:
            errors.append("Email não pode estar vazio")
        elif "@" not in user_email or "." not in user_email:
            errors.append("Email inválido - deve conter @ e domínio")
        elif len(user_email.strip()) < 5:
            errors.append("Email muito curto")
            
        if not subject or len(subject.strip()) < 5:
            errors.append("Motivo do contato deve ter pelo menos 5 caracteres")
            
        if not message or len(message.strip()) < 10:
            errors.append("Mensagem deve ter pelo menos 10 caracteres")
            
        if message and len(message) > 1500:
            errors.append("Mensagem deve ter no máximo 1500 caracteres")
        
        return errors
