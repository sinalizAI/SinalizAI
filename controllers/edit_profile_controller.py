from utils.base_screen import BaseScreen

class EditScreen(BaseScreen):
    
    def on_enter(self):
        """Carrega os dados atuais do usuário quando a tela é aberta"""
        # Restaura a navegação original se existir
        if hasattr(self.manager, 'edit_original_previous'):
            self.manager.previous_screen = self.manager.edit_original_previous
        elif not hasattr(self.manager, 'previous_screen') or not self.manager.previous_screen:
            # Garante que há uma tela anterior definida (fallback para perfil)
            self.manager.previous_screen = "profile"
        
        self.load_user_data()
    
    def load_user_data(self):
        """Carrega e preenche os campos com os dados atuais do usuário"""
        if hasattr(self.manager, 'user_data') and self.manager.user_data:
            user_data = self.manager.user_data
            
            # Preenche o campo nome
            name_field = self.ids.get('edit_input_name')
            if name_field:
                name_field.text = user_data.get('displayName', '')
            
            # Preenche o campo email
            email_field = self.ids.get('edit_input_email')
            if email_field:
                email_field.text = user_data.get('email', '')
        else:
            # Se não há dados do usuário, redireciona para login
            from utils.message_helper import show_message
            show_message("Sessão expirada. Faça login novamente.")
            self.go_to_welcome()
    
    def go_to_change_password(self):
        """Redireciona para a tela de esqueceu a senha para alterar a senha"""
        # Preserva a cadeia de navegação original - salva de onde a edição veio
        original_previous = getattr(self.manager, 'previous_screen', 'profile')
        
        # Salva no manager para uso posterior
        self.manager.edit_original_previous = original_previous
        
        # Informa de onde está vindo para a tela de esqueceu a senha
        try:
            forgot_screen = self.manager.get_screen('forgot')
            forgot_screen.came_from_edit = True
            forgot_screen.original_edit_previous = original_previous  # Salva a origem da edição
            
            # Passa o email atual do usuário se disponível
            if hasattr(self.manager, 'user_data') and self.manager.user_data:
                current_email = self.manager.user_data.get('email', '')
                if current_email:
                    # Preenche o campo de email na tela de esqueceu senha
                    def fill_email(dt):
                        if hasattr(forgot_screen, 'ids') and 'forgot_input' in forgot_screen.ids:
                            forgot_screen.ids.forgot_input.text = current_email
                    
                    from kivy.clock import Clock
                    Clock.schedule_once(fill_email, 0.1)
        except Exception:
            pass
        
        # Vai para a tela de esqueceu a senha
        self.go_to_fg_passwd()
    
    def save_changes(self):
        """Salva as alterações do perfil"""
        from utils.message_helper import show_message

        # Verifica se o usuário está logado
        if not hasattr(self.manager, 'user_data') or not self.manager.user_data.get('idToken'):
            show_message("Erro: Usuário não autenticado. Faça login novamente.")
            self.go_to_welcome()
            return

        # Pega os valores dos campos
        name_field = self.ids.get('edit_input_name')
        email_field = self.ids.get('edit_input_email')

        if not name_field or not email_field:
            show_message("Erro ao acessar os campos do formulário")
            return

        new_name = name_field.text.strip()
        new_email = email_field.text.strip()

        # Validações básicas
        if not new_name:
            show_message("O nome não pode estar vazio")
            return

        if not new_email:
            show_message("O e-mail não pode estar vazio")
            return

        if not self.validate_email(new_email):
            show_message("Formato de e-mail inválido")
            return

        # Obtém dados atuais do usuário
        current_name = self.manager.user_data.get('displayName', '')
        current_email = self.manager.user_data.get('email', '')
        id_token = self.manager.user_data['idToken']

        # Verifica se há mudanças
        name_changed = new_name != current_name
        email_changed = new_email != current_email

        if not name_changed and not email_changed:
            show_message("Nenhuma alteração foi feita")
            return

        # Tratamento especial para alteração de email
        if email_changed:
            show_message("Alteração de email não é permitida pelo Firebase. Para alterar o email, entre em contato com o suporte ou use a função 'Esqueceu a senha' para redefinir sua conta.")
            return

        # Atualiza apenas o nome se mudou
        if name_changed:
            show_message("Atualizando nome...")

            # Usa o backend functions para atualizar o perfil (nome)
            from services import backend_client
            status, response = backend_client.update_profile(id_token, newDisplayName=new_name)

            if status != 200 or not isinstance(response, dict) or not response.get('success'):
                # tenta extrair mensagem de erro
                msg = None
                if isinstance(response, dict):
                    msg = response.get('message') or (response.get('error') and response.get('error').get('message'))
                if not msg:
                    msg = self.get_friendly_error(response)
                show_message(f"Erro ao atualizar nome: {msg}")
                return

            # Atualiza dados locais
            self.manager.user_data['displayName'] = new_name
            # se backend retornou novo idToken, atualiza
            if isinstance(response.get('data'), dict) and response['data'].get('idToken'):
                self.manager.user_data['idToken'] = response['data'].get('idToken')

        # Sucesso
        show_message("Perfil atualizado com sucesso!")

        # Atualiza a tela de perfil se existir
        try:
            profile_screen = self.manager.get_screen('profile')
            # Força a atualização dos dados na tela de perfil
            profile_screen.load_user_profile()
        except Exception:
            pass

        # Volta para a tela de perfil
        self.go_to_back()
