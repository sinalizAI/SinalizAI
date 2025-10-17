from utils.base_screen import BaseScreen
from services import backend_client
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
        
        # Chama a função do backend (functions) para resetar senha
        status, response = backend_client.reset_password(email)
        if status == 200 and isinstance(response, dict) and response.get('success'):
            
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
            # Tratamento de erro: tenta extrair mensagem
            error_message = self.get_friendly_error(response)
            error_code = None
            if isinstance(response, dict):
                error_code = response.get('message') or (response.get('error') and response.get('error').get('message'))

            if error_code and "EMAIL_NOT_FOUND" in error_code:
                show_message("Este e-mail não está cadastrado ou foi digitado errado. Por favor, corrija e tente novamente.")
            elif error_code and "INVALID_EMAIL" in error_code:
                show_message("Formato de e-mail inválido!")
            elif error_code and "TOO_MANY_ATTEMPTS_TRY_LATER" in error_code:
                show_message("Muitas tentativas. Tente novamente mais tarde.")
            else:
                show_message(error_message)
