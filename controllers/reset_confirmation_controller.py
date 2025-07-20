from helpers.base_screen import BaseScreen
from helpers.message_helper import show_message

class ConfirmationScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""  # Email do usuário que solicitou o reset
    
    def on_enter(self):
        """Atualiza o texto com o email do usuário quando a tela é aberta"""
        if self.user_email:
            # Atualiza o texto do label com o email do usuário
            email_label = self.ids.get('email_label')
            if email_label:
                email_label.text = f"Enviamos uma mensagem para\n'{self.user_email}'."
    
    def close_screen(self):
        """Volta para a tela de login"""
        self.go_to_back()
