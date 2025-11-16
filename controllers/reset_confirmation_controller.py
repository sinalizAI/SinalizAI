from utils.base_screen import BaseScreen
from utils.message_helper import show_message

class ConfirmationScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""
        self.came_from_edit = False
        self.original_edit_previous = "profile"
    
    def on_enter(self):
        
        if self.user_email:

            email_label = self.ids.get('email_label')
            if email_label:
                email_label.text = f"Enviamos uma mensagem para\n'{self.user_email}'."
    
    def close_screen(self):
        
        if self.came_from_edit:

            self.manager.edit_original_previous = self.original_edit_previous
            self.manager.current = "edit"
        else:

            self.go_to_back()
