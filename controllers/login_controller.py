import re
from utils.base_screen import BaseScreen
from services import backend_client
from utils.message_helper import show_message
from kivy.clock import Clock

class LoginScreen(BaseScreen):
    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email)

    def do_login(self, email, password):
        import time
        start = time.time()
        if not email or not password:
            show_message("Email e senha não podem estar vazios!")
            return

        if not self.validate_email(email):
            show_message("Email inválido!")
            return

        def login_callback(dt):
            self._perform_login(email, password)
            end = time.time()
            tempo = end - start
            print(f" Tempo de resposta (login): {tempo:.3f}s")
            from utils.benchmark_logger import log_benchmark
            log_benchmark('login', tempo, {'email': email})

        Clock.schedule_once(login_callback, 0.1)

    def _perform_login(self, email, password):
        status, response = backend_client.login(email, password)

        if status == 200 and isinstance(response, dict) and response.get('success', True) is not False:

            data = response.get('data') if 'data' in response else response
            self.manager.user_data = {
                "email": data.get("email"),
                "idToken": data.get("idToken"),
                "displayName": data.get("displayName", ""),
                "localId": data.get("localId") or data.get("userId")
            }
            show_message("Login realizado com sucesso!")
            self.go_to_home()
        else:

            try:
                friendly = self.get_friendly_error(response if isinstance(response, dict) else { 'error': { 'message': str(response) } })
            except Exception:
                friendly = 'Erro no login. Verifique seu email e senha.'
            show_message(friendly)
