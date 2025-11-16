from utils.base_screen import BaseScreen
from services import backend_client
from utils.message_helper import show_message

class ForgotScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_email = ""
        self.came_from_edit = False
        self.original_edit_previous = "profile"
    
    def on_enter(self):
        
        self.ids.forgot_input.text = ""
        

        if hasattr(self.manager, 'previous_screen') and self.manager.previous_screen == "edit":
            self.came_from_edit = True
        else:
            self.came_from_edit = False
    
    def go_to_back(self):
        
        if self.came_from_edit:

            self.manager.edit_original_previous = self.original_edit_previous
            self.manager.current = "edit"
        else:

            super().go_to_back()
    
    def send_reset_email(self):
        
        import time
        start = time.time()
        email = self.ids.forgot_input.text.strip()


        if not email:
            show_message("Por favor, digite seu e-mail.")
            return


        if not self.validate_email(email):
            show_message("Formato de e-mail inválido!")
            return


        status, response = backend_client.reset_password(email)
        end = time.time()
        tempo = end - start
        if status == 200 and isinstance(response, dict) and response.get('success'):

            self.user_email = email

            try:
                confirmation_screen = self.manager.get_screen('reset_confirmation')
                confirmation_screen.user_email = email

                confirmation_screen.came_from_edit = self.came_from_edit
                confirmation_screen.original_edit_previous = self.original_edit_previous
            except Exception:
                pass

            show_message("E-mail de recuperação enviado!")
            print(f" Tempo de resposta (reset senha): {tempo:.3f}s")
            from utils.benchmark_logger import log_benchmark
            log_benchmark('reset_senha', tempo, {'email': email})
            self.go_to_reset_confirmation()
        else:

            error_message = self.get_friendly_error(response)
            error_code = None
            if isinstance(response, dict):
                error_code = response.get('message') or (response.get('error') and response.get('error').get('message'))

            if error_code and "EMAIL_NOT_FOUND" in error_code:
                show_message("Este e-mail não está cadastrado ou foi digitado errado. Por favor, corrija e tente novamente.")
            elif error_code and "INVALID_EMAIL" in error_code:
                show_message("Formato de e-mail inválido!")
            elif error_code and "TOO_MANY_ATTEMPTS_TRY_LATER" in error_code:
                show_message("Muitas tentativas. Tente novamente mais tarde.")
            else:
                show_message(error_message)
            print(f" Tempo de resposta (reset senha): {tempo:.3f}s")
            from utils.benchmark_logger import log_benchmark
            log_benchmark('reset_senha', tempo, {'email': email})
