# Firebase Functions (SendGrid)

Este diretório contém a função HTTP `api` utilizada como backend para operações de autenticação, perfil e envio de e-mails via SendGrid.

Como usar:

1. Instale dependências:

   npm install

2. Faça deploy (ou emulação local com Firebase CLI):

   firebase deploy --only functions

3. Armazene a chave do SendGrid com Secret Manager e mapeie para a função:

   gcloud secrets create sendgrid-api-key --data-file=PATH_TO_KEY
   gcloud functions deploy api --set-secrets="SENDGRID_API_KEY=sendgrid-api-key:latest" --update-env-vars "SENDGRID_FROM=your@verified-sender.com"

4. Se você usar endpoints que chamam a REST API de autenticação (Identity Toolkit), configure também a chave da API do Firebase:

   firebase functions:config:set firebase.api_key="YOUR_FIREBASE_API_KEY"

5. Endpoints disponíveis (POST):
   - /register { email, password, displayName } -> cria usuário e retorna idToken/refreshToken
   - /login { email, password } -> retorna idToken/refreshToken via REST
   - /resetPassword { email } -> envia email de reset
   - /updateProfile (Authorization: Bearer <idToken>) { newEmail?, newDisplayName? }
   - /changePassword (Authorization: Bearer <idToken>) { newPassword }
   - /deleteAccount (Authorization: Bearer <idToken>) -> deleta usuario e doc do Firestore
   - /sendFeedback { user_email, user_name, subject, message } -> envia feedback para caixa de entrada configurada
   - /generateVerification { email, uid } -> gera token de verificação e tenta enviar e-mail
   - /verifyToken { uid, token } -> valida e marca e-mail como verificado

6. Autenticação do cliente: após login via /login obtenha idToken e envie no header `Authorization: Bearer <idToken>` para endpoints protegidos.
