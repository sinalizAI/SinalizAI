from utils.base_screen import BaseScreen
from utils.message_helper import show_exit_dialog

class ProfileScreen(BaseScreen):
    def confirmar_exclusao_dados(self):
        from utils.message_helper import show_delete_data_dialog
        show_delete_data_dialog(self)

    def excluir_dados_usuario(self):
        # Remove conta do Firebase Authenticator, mantém aceite legal
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
                # tentar extrair mensagem
                msg = None
                if isinstance(resp, dict):
                    msg = resp.get('message') or (resp.get('error') and resp.get('error').get('message'))
                if not msg:
                    msg = 'Erro ao excluir dados.'
                show_message(f"Erro: {msg}")
        else:
            show_message("Usuário não autenticado.")
    
    def on_enter(self):
        """Carrega os dados do usuário quando a tela é aberta"""
        self.load_user_profile()
    
    def load_user_profile(self):
        """Carrega e exibe os dados do usuário logado"""
        if hasattr(self.manager, 'user_data') and self.manager.user_data:
            user_data = self.manager.user_data
            
            # Atualiza nome do usuário com fallbacks
            name = user_data.get('displayName', '')
            
            # Se o displayName estiver vazio, tenta usar o email como base
            if not name or name.strip() == '':
                email = user_data.get('email', '')
                if email:
                    # Usa a parte antes do @ como nome
                    name = email.split('@')[0].capitalize()
                else:
                    name = 'Usuário'
            
            # Atualiza nome do usuário
            if hasattr(self, 'user_name'):
                self.user_name = name
            
            # Atualiza email do usuário  
            email = user_data.get('email', '')
            if hasattr(self, 'user_email'):
                self.user_email = email
                
            # Atualiza labels na interface se existirem
            try:
                if 'profile_username_label' in self.ids:
                    self.ids.profile_username_label.text = f"Olá, {name}!"
                if 'profile_email_label' in self.ids:
                    self.ids.profile_email_label.text = email
            except Exception:
                pass
        else:
            # Se não há dados do usuário, redireciona para login
            from utils.message_helper import show_message
            show_message("Sessão expirada. Faça login novamente.")
            self.go_to_welcome()
    
    def confirmar_saida(self):
        show_exit_dialog(self)
