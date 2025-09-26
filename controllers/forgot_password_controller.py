from utils.base_screen import BaseScreen
from models.firebase_auth_model import reset_password
from utils.message_helper import show_message

class ForgotScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""  # Para armazenar o email do usuário
        self.came_from_edit = False  # Para saber se veio da tela de edição
        self.original_edit_previous = "profile"  # Para saber onde a edição deve voltar
    
    def on_enter(self):
        """Limpa o campo de email quando a tela é aberta e verifica de onde veio"""
        self.ids.forgot_input.text = ""
        
        # Verifica se veio da tela de edição de perfil
        if hasattr(self.manager, 'previous_screen') and self.manager.previous_screen == "edit":
            self.came_from_edit = True
        else:
            self.came_from_edit = False
    
    def go_to_back(self):
        """Método personalizado para voltar considerando de onde veio"""
        if self.came_from_edit:
            # Se veio da edição, restaura a navegação original da edição
            self.manager.edit_original_previous = self.original_edit_previous
            self.manager.current = "edit"
        else:
            # Se não veio da edição, usa o método padrão
            super().go_to_back()
    
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
                # Informa à tela de confirmação de onde viemos
                confirmation_screen.came_from_edit = self.came_from_edit
                confirmation_screen.original_edit_previous = self.original_edit_previous
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
