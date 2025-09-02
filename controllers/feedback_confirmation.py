from helpers.base_screen import BaseScreen
from helpers.message_helper import show_message

class FeedbackConfirmationScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""  # Email do usuário que enviou o feedback
        self.user_name = ""   # Nome do usuário que enviou o feedback
        self.feedback_subject = ""  # Assunto do feedback enviado
    
    def on_enter(self):
        """Atualiza as informações quando a tela é aberta"""
        # Atualiza informações se disponíveis
        if self.user_name and self.feedback_subject:
            message_label = self.ids.get('feedback_message_label')
            if message_label:
                message_label.text = f"Obrigado, {self.user_name}!\nSeu feedback sobre '{self.feedback_subject}'\nfoi enviado com sucesso!"
    
    def close_screen(self):
        """Volta para a tela de perfil quando fechar"""
        self.go_to_profile()
    
    def go_to_profile(self):
        """Vai para a tela de perfil"""
        self.manager.transition.direction = 'right'
        self.manager.current = 'profile'