from utils.base_screen import BaseScreen
from services import backend_client
from utils.message_helper import show_message
from models import AuthenticationModel

class RegisterScreen(BaseScreen):

    def on_pre_enter(self):

        self.ids.create_button.disabled = False

    def do_register(self, email, password, confirm_password, name):
        import time
        start = time.time()
        create_button = self.ids.create_button
        create_button.disabled = True


        validation_errors = AuthenticationModel.validate_registration_data(email, password, confirm_password, name)
        if validation_errors:
            self.show_error(validation_errors[0]) 
            return


        status, response = backend_client.register(email, password, name)

        if status == 200 and isinstance(response, dict) and response.get('success'):

            ls, lr = backend_client.login(email, password)
            if ls == 200 and isinstance(lr, dict) and lr.get('success', True) is not False:
                data = lr.get('data') if 'data' in lr else lr
                id_token = data.get('idToken')
                user_id = data.get('localId') or data.get('userId') or response.get('data', {}).get('uid') or response.get('uid')



                self.manager.user_data = {
                    "email": email,
                    "idToken": id_token,
                    "displayName": name,
                    "localId": user_id
                }
                show_message("Cadastro realizado com sucesso!")
                end = time.time()
                tempo = end - start
                print(f" Tempo de resposta (registro): {tempo:.3f}s")
                from utils.benchmark_logger import log_benchmark
                log_benchmark('registro', tempo, {'email': email, 'name': name})
                self.go_to_home()
                return
            else:

                err_msg = self.get_friendly_error(lr) if isinstance(lr, dict) else str(lr)
                self.show_error(f"Cadastro realizado, mas falha ao autenticar automaticamente: {err_msg}")
                end = time.time()
                tempo = end - start
                print(f" Tempo de resposta (registro): {tempo:.3f}s")
                from utils.benchmark_logger import log_benchmark
                log_benchmark('registro', tempo, {'email': email, 'name': name})
                return


        error_message = self.get_friendly_error(response)
        self.show_error(error_message)
        end = time.time()
        tempo = end - start
        print(f" Tempo de resposta (registro): {tempo:.3f}s")
        from utils.benchmark_logger import log_benchmark
        log_benchmark('registro', tempo, {'email': email, 'name': name})

    def show_error(self, message):
        show_message(message)
        self.ids.create_button.disabled = False
