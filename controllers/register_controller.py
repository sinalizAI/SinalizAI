from utils.base_screen import BaseScreen
from models import firebase_auth_model
from models.legal_acceptance_model import save_legal_acceptance  # novo import
from utils.message_helper import show_message

class RegisterScreen(BaseScreen):

    def on_pre_enter(self):
        # Garante que o botão esteja ativado ao entrar na tela
        self.ids.create_button.disabled = False

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
            user_id = response["localId"]

            # Salva o nome do usuário
            name_success, name_response = firebase_auth_model.update_display_name(id_token, name)
            if not name_success:
                self.show_error(f"Erro ao atualizar nome: {name_response}")
                return

            # Salva o aceite dos termos e política
            acceptance_success, acceptance_error = save_legal_acceptance(user_id, id_token)
            if not acceptance_success:
                self.show_error(f"Erro ao registrar aceite dos termos legais: {acceptance_error}")
                return

            # Salva os dados do usuário na sessão
            self.manager.user_data = {
                "email": email,
                "idToken": id_token,
                "displayName": name,
                "localId": user_id
            }

            show_message("Cadastro realizado com sucesso!")
            self.go_to_home()
        else:
            error_message = self.get_friendly_error(response)
            self.show_error(error_message)

    def show_error(self, message):
        show_message(message)
        self.ids.create_button.disabled = False
