from controllers.base_screen import BaseScreen
from controllers.message_helper import show_message

class ConfirmationScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""  # Email do usuário que solicitou o reset
    
    def on_enter(self):
        """Atualiza o texto com o email do usuário quando a tela é aberta"""
        print(f"[DEBUG] Tela de confirmação aberta com email: {self.user_email}")
        if self.user_email:
            # Atualiza o texto do label com o email do usuário
            email_label = self.ids.get('email_label')
            if email_label:
                email_label.text = f"Enviamos uma mensagem para\n'{self.user_email}'."
                print(f"[DEBUG] Texto do label atualizado para: {email_label.text}")
            else:
                print("[ERROR] Label email_label não encontrado!")
    
    def close_screen(self):
        """Volta para a tela de login"""
        self.go_to_back()
