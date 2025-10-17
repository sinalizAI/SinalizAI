from utils.base_screen import BaseScreen
from services import backend_client
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

        # Tenta registrar o usuário via backend functions
        status, response = backend_client.register(email, password, name)
        # Se registro OK, façamos login automático usando email+password
        if status == 200 and isinstance(response, dict) and response.get('success'):
            # tentar login após registro para obter idToken (o backend fornece customToken em alguns casos)
            ls, lr = backend_client.login(email, password)
            if ls == 200 and isinstance(lr, dict) and lr.get('success', True) is not False:
                data = lr.get('data') if 'data' in lr else lr
                id_token = data.get('idToken')
                user_id = data.get('localId') or data.get('userId') or response.get('data', {}).get('uid') or response.get('uid')

                # Salva o aceite dos termos (server já tentou salvar, mas reafirmar caso necessário)
                try:
                    acceptance_success, acceptance_error = save_legal_acceptance(user_id, id_token)
                    if not acceptance_success:
                        self.show_error(f"Erro ao registrar aceite dos termos: {acceptance_error}")
                        return
                except Exception:
                    # não bloquear o fluxo por problemas secundários
                    pass

                # Salva os dados do usuário na sessão
                self.manager.user_data = {
                    "email": email,
                    "idToken": id_token,
                    "displayName": name,
                    "localId": user_id
                }
                show_message("Cadastro realizado com sucesso!")
                self.go_to_home()
                return
            else:
                # Não conseguiu efetuar login automático: informe o usuário de forma amigável
                err_msg = self.get_friendly_error(lr) if isinstance(lr, dict) else str(lr)
                self.show_error(f"Cadastro realizado, mas falha ao autenticar automaticamente: {err_msg}")
                return

        # Em caso de erro, exibir mensagem amigável
        error_message = self.get_friendly_error(response)
        self.show_error(error_message)

    def show_error(self, message):
        show_message(message)
        self.ids.create_button.disabled = False
