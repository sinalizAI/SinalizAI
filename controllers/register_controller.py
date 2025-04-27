import re
from controllers.base_screen import BaseScreen
from models import firebase_auth_model
from kivymd.toast import toast

class RegisterScreen(BaseScreen):
    def do_register(self, email, password, confirm_password, name):
        create_button = self.ids.create_button
        create_button.disabled = True  # Impede múltiplos envios

        # Verificação de campos obrigatórios
        if not email or not password or not confirm_password or not name:
            self.show_error("Todos os campos devem ser preenchidos!")
            return

        # Verificação de senhas iguais
        if password != confirm_password:
            self.show_error("As senhas não coincidem!")
            return

        # Verificação do formato do email
        if not self.validate_email(email):
            self.show_error("Email inválido!")
            return

        # Verificação do comprimento do nome
        if len(name) < 3:
            self.show_error("Nome deve ter pelo menos 3 caracteres!")
            return

        # Verificação do formato da senha
        if not self.validate_password(password):
            self.show_error("Senha fraca! Deve ter 8 caracteres, incluindo letra maiúscula, minúscula, número e símbolo especial.")
            return

        # Tenta registrar o usuário
        success, response = firebase_auth_model.register(email, password)
        if success:
            id_token = response["idToken"]
            name_success, name_response = firebase_auth_model.update_display_name(id_token, name)
            if name_success:
                toast("Cadastro realizado com sucesso!")
                self.go_to_home()
            else:
                self.show_error(f"Erro ao atualizar nome: {name_response}")
        else:
            error_message = self.get_friendly_error(response)
            self.show_error(error_message)

    def show_error(self, message):
        toast(message)
        create_button = self.ids.create_button
        create_button.disabled = False  # Reativa o botão se erro

    def get_friendly_error(self, response):
        try:
            error_code = response.get("error", {}).get("message", "")
        except AttributeError:
            return "Erro desconhecido. Tente novamente."

        friendly_errors = {
            "EMAIL_EXISTS": "Este email já está cadastrado!",
            "OPERATION_NOT_ALLOWED": "Cadastro de email/senha não está habilitado!",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Muitas tentativas. Tente novamente mais tarde.",
            "INVALID_EMAIL": "Formato de email inválido!",
            "WEAK_PASSWORD": "Senha muito fraca!",
            "EMAIL_NOT_FOUND": "Email não encontrado!",
            "INVALID_PASSWORD": "Senha incorreta!",
            "USER_DISABLED": "Conta desativada. Contate o suporte.",
            "MISSING_PASSWORD": "Senha obrigatória!",
            "MISSING_EMAIL": "Email obrigatório!",
        }

        return friendly_errors.get(error_code, "Erro no cadastro. Verifique os dados.")

    def validate_email(self, email):
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        return re.match(pattern, email) is not None

    def validate_password(self, password):
        pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$"
        return re.match(pattern, password) is not None
