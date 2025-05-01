from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import SlideTransition
from kivymd.toast import toast
import re

class BaseScreen(MDScreen):
    
    def go_to_welcome(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
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
        self.manager.previous_screen = self.manager.current  #
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
    
    # Função para voltar à tela anterior
    def go_to_back(self):
        if hasattr(self.manager, 'previous_screen') and self.manager.previous_screen:
            self.manager.transition = SlideTransition(direction='right', duration=0.0)
            self.manager.current = self.manager.previous_screen
    
    #Função que cuida das validações
    def validate_email(self, email):
        # Valida o formato do email
        return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

    def validate_password(self, password):
        return bool(re.match(
            r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$',
            password
        ))

    def show_error(self, message):
        # Exibe mensagem de erro no formato de toast
        from kivymd.toast import toast
        toast(message)
    # Função que cuida dos erros do Firebase
    def get_friendly_error(self, response):
        try:
            error_code = response.get("error", {}).get("message", "")
        except AttributeError:
            return "Erro desconhecido. Tente novamente."

        friendly_errors = {
            # Registro
            "EMAIL_EXISTS": "Este email já está cadastrado!",
            "OPERATION_NOT_ALLOWED": "Cadastro de email/senha não está habilitado!",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Muitas tentativas. Tente novamente mais tarde.",
            "INVALID_EMAIL": "Formato de email inválido!",
            "WEAK_PASSWORD": "Senha fraca! Use pelo menos 6 caracteres com letra maiúscula, minúscula, número e símbolo.",

            # Login
            "EMAIL_NOT_FOUND": "Email não encontrado. Verifique ou cadastre-se.",
            "INVALID_PASSWORD": "Senha incorreta. Tente novamente.",
            "USER_DISABLED": "Conta desativada. Contate o suporte.",
            "INVALID_LOGIN_CREDENTIAL": "Email ou senha inválidos.",
            "INVALID_LOGIN_CREDENTIALS": "Email ou senha inválidos.",

            # Tokens e sessão
            "INVALID_ID_TOKEN": "Sessão inválida ou expirada. Faça login novamente.",
            "TOKEN_EXPIRED": "Sessão expirada. Faça login novamente.",
            "USER_NOT_FOUND": "Usuário não encontrado. A conta pode ter sido removida.",
            "CREDENTIAL_TOO_OLD_LOGIN_AGAIN": "Sessão antiga. Faça login novamente.",

            # Campos ausentes
            "MISSING_PASSWORD": "Senha obrigatória!",
            "MISSING_EMAIL": "Email obrigatório!",
        }

        return friendly_errors.get(error_code, f"Erro no cadastro ou login ({error_code}). Verifique os dados.")   
