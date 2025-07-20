from helpers.base_screen import BaseScreen
from models.firebase_auth_model import reset_password
from helpers.message_helper import show_message

class ForgotScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""  # Para armazenar o email do usuário
    
    def on_enter(self):
        """Limpa o campo de email quando a tela é aberta"""
        self.ids.forgot_input.text = ""
    
    def send_reset_email(self):
        """Envia email de recuperação de senha"""
        email = self.ids.forgot_input.text.strip()
        
        # Validação básica
        if not email:
            show_message("Por favor, digite seu e-mail.")
            return
        
        # Validação do formato do email
        if not self.validate_email(email):
            show_message("Formato de e-mail inválido!")
            return
        
        # Chama a função do Firebase para resetar senha
        success, response = reset_password(email)
        
        if success:
            # Salva o email para usar na tela de confirmação
            self.user_email = email
            # Passa o email para a tela de confirmação
            try:
                confirmation_screen = self.manager.get_screen('reset_confirmation')
                confirmation_screen.user_email = email
            except Exception:
                pass
            
            show_message("E-mail de recuperação enviado!")
            self.go_to_reset_confirmation()
        else:
            # Trata os erros do Firebase
            error_message = self.get_friendly_error(response)
            
            # Mensagens específicas para reset de senha
            error_code = response.get("error", {}).get("message", "")
            
            if "EMAIL_NOT_FOUND" in error_code:
                show_message("Este e-mail não está cadastrado ou foi digitado errado. Por favor, corrija e tente novamente.")
            elif "INVALID_EMAIL" in error_code:
                show_message("Formato de e-mail inválido!")
            elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_code:
                show_message("Muitas tentativas. Tente novamente mais tarde.")
            else:
                show_message(error_message)
