from utils.base_screen import BaseScreen
from utils.message_helper import show_message

class ConfirmationScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""  # Email do usuário que solicitou o reset
        self.came_from_edit = False  # Para saber se veio da tela de edição
        self.original_edit_previous = "profile"  # Para saber onde a edição deve voltar
    
    def on_enter(self):
        """Atualiza o texto com o email do usuário quando a tela é aberta"""
        if self.user_email:
            # Atualiza o texto do label com o email do usuário
            email_label = self.ids.get('email_label')
            if email_label:
                email_label.text = f"Enviamos uma mensagem para\n'{self.user_email}'."
    
    def close_screen(self):
        """Volta para a tela apropriada baseado em onde veio"""
        if self.came_from_edit:
            # Se veio da edição de perfil, volta para lá e restaura a navegação correta
            self.manager.edit_original_previous = self.original_edit_previous  # Salva para a edição restaurar
            self.manager.current = "edit"
        else:
            # Se veio do login, volta para a tela de login
            self.go_to_back()
