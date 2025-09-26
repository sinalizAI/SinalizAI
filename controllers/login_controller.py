import re
from utils.base_screen import BaseScreen
from models import firebase_auth_model
from utils.message_helper import show_message
from kivy.clock import Clock

class LoginScreen(BaseScreen):
    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email)

    def do_login(self, email, password):
        if not email or not password:
            show_message("Email e senha não podem estar vazios!")
            return

        if not self.validate_email(email):
            show_message("Email inválido!")
            return

        Clock.schedule_once(lambda dt: self._perform_login(email, password), 0.1)

    def _perform_login(self, email, password):
        success, response = firebase_auth_model.login(email, password)

        if success:
            self.manager.user_data = {
                "email": response["email"],
                "idToken": response["idToken"],
                "displayName": response.get("displayName", "")
            }
            show_message("Login realizado com sucesso!")
            self.go_to_home()
        else:
            error_message = self.get_friendly_error(response)
            show_message(error_message)
