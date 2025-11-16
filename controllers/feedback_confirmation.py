from utils.base_screen import BaseScreen
from utils.message_helper import show_message

class FeedbackConfirmationScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""
        self.user_name = ""
        self.feedback_subject = ""
    
    def on_enter(self):
        

        if self.user_name and self.feedback_subject:
            message_label = self.ids.get('feedback_message_label')
            if message_label:
                message_label.text = f"Obrigado, {self.user_name}!\nSeu feedback sobre '{self.feedback_subject}'\nfoi enviado com sucesso!"
    
    def close_screen(self):
        
        self.go_to_profile()
    
    def go_to_profile(self):
        
        self.manager.transition.direction = 'right'
        self.manager.current = 'profile'