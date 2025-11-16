from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import SlideTransition
from kivymd.toast import toast
import re

class BaseScreen(MDScreen):
    
    def go_to_welcome(self):
        self.manager.previous_screen = self.manager.current
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "welcome"
    
    def go_to_home(self):
        self.manager.previous_screen = self.manager.current 
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "home"

    def go_to_profile(self):
        self.manager.previous_screen = self.manager.current  
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "profile"

    def go_to_faq(self):
        self.manager.previous_screen = self.manager.current
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "faq"

    def go_to_feedback(self):
        self.manager.previous_screen = self.manager.current  
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "feedback"

    def go_to_policy(self):
        self.manager.previous_screen = self.manager.current  
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "policy"

    def go_to_terms(self):
        self.manager.previous_screen = self.manager.current  
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "terms"
    
    def go_to_about(self):
        self.manager.previous_screen = self.manager.current
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "aboutus"

    def go_to_edit(self):
        self.manager.previous_screen = self.manager.current
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "edit"
    
    def go_to_fg_passwd(self):
        self.manager.previous_screen = self.manager.current
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "fg_passwd"       
    
    def go_to_reset_confirmation(self):
        self.manager.previous_screen = self.manager.current
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "reset_confirmation"
    

    def go_to_back(self):
        if hasattr(self.manager, 'previous_screen') and self.manager.previous_screen:
            self.manager.transition = SlideTransition(direction='right', duration=0.0)
            self.manager.current = self.manager.previous_screen
        else:

            self.manager.transition = SlideTransition(direction='right', duration=0.0)
            self.manager.current = "profile"
    

    def validate_email(self, email):

        return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

    def validate_password(self, password):
        return bool(re.match(
            r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$',
            password
        ))

    def show_error(self, message):

        from kivymd.toast import toast
        toast(message)

    def get_friendly_error(self, response):

        try:
            error_code = ''
            if response is None:
                error_code = 'UNKNOWN_ERROR'
            elif isinstance(response, str):
                error_code = response
            elif isinstance(response, dict):

                if 'error' in response and isinstance(response['error'], dict) and 'message' in response['error']:
                    error_code = response['error']['message']

                elif 'message' in response:
                    error_code = response['message']

                elif 'data' in response and isinstance(response['data'], dict) and 'error' in response['data'] and isinstance(response['data']['error'], dict) and 'message' in response['data']['error']:
                    error_code = response['data']['error']['message']
                else:

                    error_code = str(response)
            else:
                error_code = str(response)
        except Exception:
            return "Erro desconhecido. Tente novamente."

        friendly_errors = {

            "EMAIL_EXISTS": "Este email já está cadastrado!",
            "OPERATION_NOT_ALLOWED": "Cadastro de email/senha não está habilitado!",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Muitas tentativas. Tente novamente mais tarde.",
            "INVALID_EMAIL": "Formato de email inválido!",
            "WEAK_PASSWORD": "Senha fraca! Use pelo menos 6 caracteres com letra maiúscula, minúscula, número e símbolo.",


            "EMAIL_NOT_FOUND": "Email não encontrado. Verifique ou cadastre-se.",
            "INVALID_PASSWORD": "Senha incorreta. Tente novamente.",
            "USER_DISABLED": "Conta desativada. Contate o suporte.",
            "INVALID_LOGIN_CREDENTIAL": "Email ou senha inválidos.",
            "INVALID_LOGIN_CREDENTIALS": "Email ou senha inválidos.",


            "INVALID_ID_TOKEN": "Sessão inválida ou expirada. Faça login novamente.",
            "TOKEN_EXPIRED": "Sessão expirada. Faça login novamente.",
            "USER_NOT_FOUND": "Usuário não encontrado. A conta pode ter sido removida.",
            "CREDENTIAL_TOO_OLD_LOGIN_AGAIN": "Sessão antiga. Faça login novamente.",


            "MISSING_PASSWORD": "Senha obrigatória!",
            "MISSING_EMAIL": "Email obrigatório!",
            

            "NETWORK_ERROR": "Erro de conexão. Verifique sua internet e tente novamente.",
            "UNKNOWN_ERROR": "Erro desconhecido. Tente novamente.",
        }


        if isinstance(error_code, str):
            for key, val in friendly_errors.items():
                if key in error_code:
                    return val
        return friendly_errors.get(error_code, f"Erro no cadastro ou login ({error_code}). Verifique os dados.")
