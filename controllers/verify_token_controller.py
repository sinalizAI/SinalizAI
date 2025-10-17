from utils.base_screen import BaseScreen
from services import backend_client
from utils.message_helper import show_message


class VerifyTokenScreen(BaseScreen):

    def on_pre_enter(self):
        # Limpa o campo
        if 'verify_input' in self.ids:
            self.ids.verify_input.text = ''

    def submit_token(self, token_text):
        if not token_text or not token_text.strip():
            show_message('Informe o token recebido por email')
            return

        if not hasattr(self.manager, 'user_data') or not self.manager.user_data:
            show_message('Sessão inválida. Faça login novamente.')
            self.go_to_welcome()
            return

        uid = self.manager.user_data.get('localId') or self.manager.user_data.get('userId')
        if not uid:
            show_message('UID do usuário não encontrado. Faça login novamente.')
            self.go_to_welcome()
            return

        status, resp = backend_client.verify_token(uid, token_text.strip())
        if status == 200 and isinstance(resp, dict) and resp.get('success'):
            show_message('Email verificado com sucesso!')
            # marcar localmente
            self.manager.user_data['emailVerified'] = True
            self.go_to_home()
            return

        # erro
        msg = None
        if isinstance(resp, dict):
            msg = resp.get('message')
        if not msg:
            msg = 'Token inválido ou expirado.'
        show_message(f'Erro: {msg}')
