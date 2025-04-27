import re
from controllers.base_screen import BaseScreen
from models import firebase_auth_model
from kivymd.toast import toast
from kivy.clock import Clock

class LoginScreen(BaseScreen):
    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email)

    def do_login(self, email, password):
        if not email or not password:
            toast("Email e senha não podem estar vazios!")
            return

        if not self.validate_email(email):
            toast("Email inválido!")
            return

        # 🔥 Esconde o botão Entrar e mostra o Spinner
        # Faz o login depois de 0.1s para dar tempo do botão sumir
        Clock.schedule_once(lambda dt: self._perform_login(email, password), 0.1)

    def _perform_login(self, email, password):
        success, response = firebase_auth_model.login(email, password)

        if success:
            self.manager.user_data = {
                "email": response["email"],
                "idToken": response["idToken"],
                "displayName": response.get("displayName", "")
            }
            toast("Login realizado com sucesso!")
            self.go_to_home()
        else:
            error_message = self.get_friendly_error(response)
            toast(error_message)


    def get_friendly_error(self, response):
        if not isinstance(response, dict):
            return "Erro desconhecido. Tente novamente."

        error_code = response.get("error", {}).get("message", "")

        friendly_errors = {
            "INVALID_EMAIL": "Email inválido!",
            "INVALID_PASSWORD": "Senha incorreta!",
            "EMAIL_NOT_FOUND": "Usuário não encontrado!",
            "USER_DISABLED": "Conta desativada!",
            "MISSING_PASSWORD": "Senha não informada!",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Muitas tentativas, tente mais tarde.",
            "UNKNOWN_ERROR": "Erro desconhecido. Tente novamente."
        }

        return friendly_errors.get(error_code, "Erro no login. Verifique seus dados.")