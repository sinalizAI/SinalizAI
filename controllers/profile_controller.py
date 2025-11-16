from utils.base_screen import BaseScreen
from utils.message_helper import show_exit_dialog

class ProfileScreen(BaseScreen):
    def confirmar_exclusao_dados(self):
        from utils.message_helper import show_delete_data_dialog
        show_delete_data_dialog(self)

    def excluir_dados_usuario(self):

        from services import backend_client
        from utils.message_helper import show_message
        user_data = getattr(self.manager, 'user_data', None)
        id_token = user_data.get('idToken') if user_data else None
        if id_token:
            status, resp = backend_client.delete_account(id_token)
            if status == 200 and isinstance(resp, dict) and resp.get('success'):
                self.manager.user_data = None
                show_message("Dados excluídos com sucesso.")
                self.go_to_welcome()
            else:

                msg = None
                if isinstance(resp, dict):
                    msg = resp.get('message') or (resp.get('error') and resp.get('error').get('message'))
                if not msg:
                    msg = 'Erro ao excluir dados.'
                show_message(f"Erro: {msg}")
        else:
            show_message("Usuário não autenticado.")
    
    def on_enter(self):
        
        self.load_user_profile()
    
    def load_user_profile(self):
        
        if hasattr(self.manager, 'user_data') and self.manager.user_data:
            user_data = self.manager.user_data
            

            name = user_data.get('displayName', '')
            

            if not name or name.strip() == '':
                email = user_data.get('email', '')
                if email:

                    name = email.split('@')[0].capitalize()
                else:
                    name = 'Usuário'
            

            if hasattr(self, 'user_name'):
                self.user_name = name
            

            email = user_data.get('email', '')
            if hasattr(self, 'user_email'):
                self.user_email = email
                

            try:
                if 'profile_username_label' in self.ids:
                    self.ids.profile_username_label.text = f"Olá, {name}!"
                if 'profile_email_label' in self.ids:
                    self.ids.profile_email_label.text = email
            except Exception:
                pass
        else:

            from utils.message_helper import show_message
            show_message("Sessão expirada. Faça login novamente.")
            self.go_to_welcome()
    
    def confirmar_saida(self):
        show_exit_dialog(self)
