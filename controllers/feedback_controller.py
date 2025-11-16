from utils.base_screen import BaseScreen
from utils.message_helper import show_message
from services import backend_client
def validate_feedback_data(user_email, user_name, subject, message):
    errors = []
    if not user_name or len(user_name.strip()) < 3:
        errors.append("Nome deve ter pelo menos 3 caracteres")
    if not user_email:
        errors.append("Email é obrigatório")
    elif len(user_email.strip()) == 0:
        errors.append("Email não pode estar vazio")
    elif "@" not in user_email or "." not in user_email:
        errors.append("Email inválido - deve conter @ e domínio")
    elif len(user_email.strip()) < 5:
        errors.append("Email muito curto")
    if not subject or len(subject.strip()) < 5:
        errors.append("Motivo do contato deve ter pelo menos 5 caracteres")
    if not message or len(message.strip()) < 10:
        errors.append("Mensagem deve ter pelo menos 10 caracteres")
    if message and len(message) > 1500:
        errors.append("Mensagem deve ter no máximo 1500 caracteres")
    return errors
from services import backend_client

class FeedbackScreen(BaseScreen):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def on_enter(self):
        
        self.load_user_data()
    
    def load_user_data(self):
        
        
        if hasattr(self.manager, 'user_data') and self.manager.user_data:
            user_data = self.manager.user_data
            

            name_field = self.ids.get('feedback_screen_name_input')
            if name_field:
                display_name = user_data.get('displayName', '')
                if display_name:
                    name_field.text = display_name
                else:

                    email = user_data.get('email', '')
                    if email and '@' in email:
                        fallback_name = email.split('@')[0].capitalize()
                        name_field.text = fallback_name
            

            email_field = self.ids.get('feedback_screen_email_input')
            if email_field:
                user_email = user_data.get('email', '')
                email_field.text = user_email
                email_field.hint_text = "Email do usuário logado"
        else:

            show_message("Você precisa estar logado para enviar feedback")
            self.go_to_welcome()
    
    def send_feedback(self):
        
        

        if not hasattr(self.manager, 'user_data') or not self.manager.user_data:
            show_message("Erro: Usuário não autenticado")
            self.go_to_welcome()
            return
        

        name_field = self.ids.get('feedback_screen_name_input')
        email_field = self.ids.get('feedback_screen_email_input')
        subject_field = self.ids.get('feedback_screen_subject_input')
        message_field = self.ids.get('feedback_screen_input_message')
        
        if not all([name_field, email_field, subject_field, message_field]):
            show_message("Erro ao acessar os campos do formulário")
            return
        

        user_name = name_field.text.strip()
        user_email = email_field.text.strip()
        subject = subject_field.text.strip()
        message = message_field.text.strip()
        

        if not user_email and hasattr(self.manager, 'user_data') and self.manager.user_data:
            user_email = self.manager.user_data.get('email', '')


        errors = validate_feedback_data(user_email, user_name, subject, message)

        if errors:
            show_message("\n".join(errors))
            return
        

        show_message("Enviando feedback...")
        

        send_button = self.ids.get('feedback_screen_send_button')
        if send_button:
            send_button.disabled = True
        

        status, result = backend_client.send_feedback(user_email, user_name, subject, message)

        if isinstance(result, dict) and result.get('success') is True:
            result = {"success": True, "message": "Feedback enviado com sucesso!"}
        elif status != 200:

            msg = None
            if isinstance(result, dict):
                msg = result.get('message') or result.get('error') or str(result)
            else:
                msg = str(result)
            result = {"success": False, "message": f"Erro ao enviar feedback (status {status}): {msg}"}
        

        if send_button:
            send_button.disabled = False
        
        if result["success"]:
            show_message(result["message"])
            

            confirmation_screen = self.manager.get_screen('feedback_confirmation')
            confirmation_screen.user_name = user_name
            confirmation_screen.user_email = user_email
            confirmation_screen.feedback_subject = subject
            

            self.go_to_feedback_confirmation()
        else:
            show_message(result["message"])
    
    def update_character_count(self, message_field):
        
        counter_label = self.ids.get('feedback_screen_counter_label')
        if counter_label and message_field:
            current_length = len(message_field.text)
            max_length = 1500
            counter_label.text = f"{current_length}/{max_length}"
            

            if current_length > max_length * 0.9:
                counter_label.text_color = [1, 0, 0, 1]
            elif current_length > max_length * 0.7:
                counter_label.text_color = [1, 0.5, 0, 1]
            else:
                counter_label.text_color = [0.5, 0.5, 0.5, 1]
    
    def clear_form(self):
        
        subject_field = self.ids.get('feedback_screen_subject_input')
        message_field = self.ids.get('feedback_screen_input_message')
        
        if subject_field:
            subject_field.text = ""
        if message_field:
            message_field.text = ""
            

        self.update_character_count(message_field)
    
    def go_to_feedback_confirmation(self):
        
        self.manager.transition.direction = 'left'
        self.manager.current = 'feedback_confirmation'
