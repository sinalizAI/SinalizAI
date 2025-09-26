import requests
import json

class EmailService:
    """Serviço para envio de emails via EmailJS"""
    
    # Configurações do EmailJS
    EMAILJS_CONFIG = {
        "service_id": "service_4vex1zl",
        "template_id": "template_9lyg2p5",
        "public_key": "AETagjMuO4_iJD8dj"
    }
    
    # Email de destino para feedbacks
    FEEDBACK_EMAIL = "***EMAIL_REMOVED***"
    
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
