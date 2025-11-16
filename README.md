# SinalizAI - Documentação Técnica Completa

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Estrutura de Diretórios](#estrutura-de-diretórios)
4. [Dependências e Tecnologias](#dependências-e-tecnologias)
5. [Configuração e Instalação](#configuração-e-instalação)
6. [Componentes Principais](#componentes-principais)
7. [Funcionalidades Detalhadas](#funcionalidades-detalhadas)
8. [API e Backend](#api-e-backend)
9. [Machine Learning](#machine-learning)
10. [Segurança](#segurança)
11. [Testes](#testes)
12. [Deploy e Produção](#deploy-e-produção)

## Visão Geral

SinalizAI é uma aplicação móvel desktop desenvolvida em Python utilizando o framework Kivy/KivyMD para tradução automática de sinais da Língua Brasileira de Sinais (LIBRAS) em tempo real. O sistema utiliza técnicas de Machine Learning para reconhecimento visual de gestos através de câmera, oferecendo uma interface intuitiva para comunicação inclusiva.

### Características Principais

- **Detecção em Tempo Real**: Reconhecimento de sinais LIBRAS através de câmera
- **Interface Responsiva**: Design moderno com Material Design
- **Autenticação Segura**: Sistema completo de login/registro com Firebase
- **Feedback de Usuário**: Sistema de envio de feedback via SendGrid
- **Multiplataforma**: Compatível com Windows, Linux e macOS
- **Arquitetura MVC**: Separação clara entre Model, View e Controller

## Arquitetura do Sistema

### Padrão Arquitetural: MVC (Model-View-Controller)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     MODELS      │    │   CONTROLLERS   │    │     VIEWS       │
│                 │    │                 │    │                 │
│ - User          │◄───┤ - Login         │◄───┤ - login.kv      │
│ - Authentication│    │ - Register      │    │ - register.kv   │
│ - Feedback      │    │ - Detection     │    │ - home.kv       │
│ - Benchmark     │    │ - Camera        │    │ - camera.kv     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │    SERVICES     │
                    │                 │
                    │ - backend_client│
                    │ - ML Models     │
                    │ - Firebase      │
                    └─────────────────┘
```

### Fluxo de Dados

1. **View**: Interface do usuário (arquivos .kv)
2. **Controller**: Lógica de negócio e manipulação de eventos
3. **Model**: Estruturas de dados e validações
4. **Services**: Comunicação externa (API, ML, Firebase)

## Estrutura de Diretórios

```
SinalizAI/
├── app.py                          # Ponto de entrada alternativo
├── main.py                         # Aplicação principal Kivy/KivyMD
├── benchmarks.json                 # Dados de performance
├── .gitignore                      # Arquivos ignorados pelo Git
│
├── assets/                         # Recursos visuais
│   ├── fonts/                      # Fontes personalizadas
│   │   ├── Athiti/
│   │   ├── Black_Han_Sans/
│   │   ├── PT_Serif/
│   │   └── palanquin/
│   └── images/                     # Imagens e ícones
│
├── config/                         # Configurações do sistema
│   ├── config_manager.py           # Gerenciador de configurações
│   └── kivy_config.py              # Configurações do Kivy
│
├── controllers/                    # Controladores MVC
│   ├── login_controller.py
│   ├── register_controller.py
│   ├── home_controller.py
│   ├── camera_controller.py
│   ├── detection_controller_camera4kivy.py
│   ├── feedback_controller.py
│   ├── profile_controller.py
│   └── [outros controladores...]
│
├── models/                         # Modelos de dados MVC
│   ├── __init__.py
│   ├── user_model.py               # Modelo de usuário
│   ├── authentication_model.py     # Validações de autenticação
│   ├── feedback_model.py           # Modelo de feedback
│   └── benchmark_model.py          # Modelo de performance
│
├── views/                          # Interfaces visuais (Kivy)
│   ├── screen_manager.py           # Gerenciador de telas
│   ├── login/
│   ├── register_page/
│   ├── home_page/
│   ├── camera_page/
│   ├── detection_page/
│   └── [outras páginas...]
│
├── services/                       # Serviços externos
│   ├── backend_client.py           # Cliente da API Firebase
│   └── ml/                         # Machine Learning
│       ├── alfabeto.pt
│       ├── isolated_prediction.py
│       ├── movinet_libras_final_base.keras
│       └── teste_tensorflow/
│
├── utils/                          # Utilitários
│   ├── base_screen.py              # Classe base para telas
│   ├── message_helper.py           # Sistema de mensagens
│   └── benchmark_logger.py         # Logger de performance
│
├── functions/                      # Firebase Functions (Backend)
│   ├── index.js                    # API principal
│   ├── package.json                # Dependências Node.js
│   └── README.md
│
├── tests/                          # Testes automatizados
│   ├── test_app.py
│   └── __init__.py
│
└── scripts/                        # Scripts utilitários
    ├── clean_code.py
    └── remove_emojis.py
```

## Dependências e Tecnologias

### Frontend (Python/Kivy)

```python
# Principais dependências Python
kivy>=2.1.0              # Framework de interface
kivymd>=1.1.1            # Material Design para Kivy
camera4kivy>=0.2.0       # Integração de câmera
opencv-python>=4.8.0     # Processamento de imagem
numpy>=1.24.0            # Computação numérica
tensorflow>=2.13.0       # Machine Learning
pillow>=10.0.0           # Manipulação de imagens
requests>=2.31.0         # Requisições HTTP
python-dotenv>=1.0.0     # Gerenciamento de variáveis ambiente
```

### Backend (Node.js/Firebase)

```json
{
  "dependencies": {
    "firebase-admin": "^11.10.0",
    "firebase-functions": "^4.4.0",
    "@sendgrid/mail": "^7.7.0",
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "axios": "^1.4.0"
  }
}
```

### Machine Learning

```python
# Dependências específicas de ML
tensorflow>=2.13.0       # Framework principal
opencv-python>=4.8.0     # Processamento de vídeo
numpy>=1.24.0            # Arrays multidimensionais
scipy>=1.10.0            # Computação científica
```

## Configuração e Instalação

### Pré-requisitos

- Python 3.8+
- Node.js 18+
- Firebase CLI
- Câmera integrada ou externa

### Instalação Python

```bash
# 1. Clone o repositório
git clone https://github.com/sinalizAI/SinalizAI.git
cd SinalizAI

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale dependências
pip install kivy[base] kivymd camera4kivy opencv-python
pip install tensorflow numpy pillow requests python-dotenv
```

### Configuração Firebase

```bash
# 1. Entre no diretório functions
cd functions

# 2. Instale dependências Node.js
npm install

# 3. Configure Firebase CLI
firebase login
firebase init

# 4. Deploy das funções
firebase deploy --only functions
```

### Configuração de Ambiente

Crie arquivo `.env` na raiz:

```env
# Firebase Configuration
FIREBASE_API_KEY=sua_api_key_aqui
FIREBASE_AUTH_DOMAIN=projeto.firebaseapp.com
FIREBASE_PROJECT_ID=seu_projeto_id
FIREBASE_STORAGE_BUCKET=projeto.appspot.com
FIREBASE_MESSAGING_SENDER_ID=123456789
FIREBASE_APP_ID=1:123:web:abc123

# Backend URL
FUNCTIONS_BASE_URL=https://us-central1-projeto.cloudfunctions.net/api
```

### Execução

```bash
# Execução principal
python main.py

# Execução alternativa
python app.py
```

## Componentes Principais

### 1. Aplicação Principal (main.py)

```python
class SinalizAIApp(MDApp):
    def build(self):
        Window.size = (900, 700)
        return ScreenManagement()
```

**Responsabilidades:**
- Configuração inicial da aplicação
- Definição do tamanho da janela
- Carregamento do gerenciador de telas

### 2. Gerenciador de Telas (screen_manager.py)

**Funcionalidades:**
- Navegação entre telas
- Gerenciamento de estado da aplicação
- Controle de fluxo de usuário

### 3. Classe Base (base_screen.py)

```python
class BaseScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Funcionalidades comuns a todas as telas
```

**Recursos Compartilhados:**
- Validações comuns (email, senha)
- Navegação padrão
- Tratamento de erros
- Interface com backend

## Funcionalidades Detalhadas

### 1. Sistema de Autenticação

#### Login (login_controller.py)

```python
class LoginScreen(BaseScreen):
    def do_login(self, email, password):
        # Validação de entrada
        if not email or not password:
            self.show_error("Campos obrigatórios")
            return
        
        # Chamada para API
        status, response = backend_client.login(email, password)
        
        # Processamento da resposta
        if status == 200 and response.get('success'):
            self.manager.user_data = response.get('data')
            self.go_to_home()
        else:
            self.show_error(self.get_friendly_error(response))
```

**Características:**
- Validação de entrada em tempo real
- Integração com Firebase Auth via backend
- Feedback visual de erros
- Armazenamento seguro de sessão

#### Registro (register_controller.py)

```python
class RegisterScreen(BaseScreen):
    def do_register(self, email, password, confirm_password, name):
        # Validação usando AuthenticationModel
        validation_errors = AuthenticationModel.validate_registration_data(
            email, password, confirm_password, name
        )
        if validation_errors:
            self.show_error(validation_errors[0])
            return
        
        # Registro via backend
        status, response = backend_client.register(email, password, name)
```

**Validações Implementadas:**
- Formato de email válido
- Força da senha (maiúscula, minúscula, número, símbolo)
- Confirmação de senha
- Nome mínimo de 3 caracteres
- Prevenção de duplicatas

### 2. Detecção de Sinais LIBRAS

#### Controller Principal (detection_controller_camera4kivy.py)

```python
class DetectionScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preview = None
        self.interpreter = None
    
    def _load_model(self):
        model_path = current_dir / "services" / "ml" / "teste_tensorflow" / "best_float16.tflite"
        self.interpreter = load_tflite_model(model_path)
    
    def analyze_pixels_callback(self, pixels_data):
        # Processamento em tempo real dos frames da câmera
        frame = self._pixels_to_frame(pixels_data)
        detections = self._detect_signs(frame)
        self._display_results(detections)
```

**Processo de Detecção:**

1. **Captura de Vídeo**: Camera4Kivy captura frames em tempo real
2. **Pré-processamento**: Redimensionamento e normalização
3. **Inferência**: Modelo TensorFlow Lite processa o frame
4. **Pós-processamento**: Filtragem por confiança e NMS
5. **Visualização**: Desenho de bounding boxes e labels

#### Classes de Sinais Suportadas

```python
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W'
]
```

**Observação**: Letras H, J, K, X, Y, Z requerem movimento e não são detectadas por este modelo estático.

### 3. Sistema de Feedback

#### Controller (feedback_controller.py)

```python
class FeedbackScreen(BaseScreen):
    def send_feedback(self, user_email, user_name, subject, message):
        # Validação usando FeedbackModel
        feedback = FeedbackModel(user_email, user_name, subject, message)
        validation_errors = feedback.validate()
        
        if validation_errors:
            self.show_error(validation_errors[0])
            return
        
        # Envio via backend
        status, result = backend_client.send_feedback(
            user_email, user_name, subject, message
        )
```

**Validações:**
- Nome: mínimo 3 caracteres
- Email: formato válido
- Assunto: mínimo 5 caracteres
- Mensagem: 10-1500 caracteres

### 4. Gerenciamento de Perfil

#### Funcionalidades Disponíveis

- **Visualização de dados**: Nome, email, data de criação
- **Edição de perfil**: Alteração de nome de exibição
- **Exclusão de conta**: Remoção completa com confirmação
- **Navegação**: Acesso a configurações e ajuda

## API e Backend

### Arquitetura do Backend

O backend é implementado usando Firebase Functions com Express.js, fornecendo uma API REST segura.

#### Endpoints Principais

```javascript
// Autenticação
POST /api/register
POST /api/login
POST /api/resetPassword

// Perfil
POST /api/updateProfile      // Requer autenticação
POST /api/changePassword    // Requer autenticação
POST /api/deleteAccount     // Requer autenticação

// Comunicação
POST /api/sendFeedback
```

#### Middleware de Autenticação

```javascript
const requireAuth = async (req, res, next) => {
    try {
        const authHeader = req.headers.authorization || '';
        if (!authHeader.startsWith('Bearer ')) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }
        const idToken = authHeader.split(' ')[1];
        const decoded = await admin.auth().verifyIdToken(idToken);
        req.user = decoded;
        return next();
    } catch (err) {
        return res.status(401).json({ success: false, message: 'Invalid token' });
    }
};
```

#### Cliente Backend (backend_client.py)

```python
def _post(path: str, data: dict, token: Optional[str] = None):
    url = DEFAULT_BASE.rstrip('/') + '/' + path.lstrip('/')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    resp = requests.post(url, json=data, headers=headers, timeout=15)
    return resp.status_code, resp.json()

def register(email: str, password: str, displayName: str):
    return _post('register', {
        'email': email, 
        'password': password, 
        'displayName': displayName
    })
```

### Integração SendGrid

O sistema utiliza SendGrid para envio de emails de feedback:

```javascript
const msg = {
    to: feedback_email,
    from: sgFrom,
    templateId: TEMPLATE_FEEDBACK,
    dynamic_template_data: {
        from_name: user_name,
        from_email: user_email,
        subject,
        message
    },
    replyTo: user_email
};

await sgMail.send(msg);
```

## Machine Learning

### Modelo de Detecção

**Arquitetura**: YOLOv5 + TensorFlow Lite
**Input**: Frame de vídeo 640x640 RGB
**Output**: Bounding boxes, scores, classes

#### Pipeline de Processamento

1. **Captura**: Camera4Kivy fornece pixels em formato RGBA
2. **Conversão**: RGBA → RGB usando NumPy
3. **Redimensionamento**: 640x640 com manutenção de aspect ratio
4. **Normalização**: Valores [0-255] → [0-1]
5. **Inferência**: TensorFlow Lite interpreta o modelo
6. **Pós-processamento**: NMS (Non-Maximum Suppression)

#### Implementação da Detecção

```python
def detect_frame(interpreter, frame, conf_threshold=0.5, iou_threshold=0.45, max_det=300):
    # Pré-processamento
    input_tensor = preprocess_frame(frame)
    
    # Inferência
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # Obter resultados
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    
    # Filtrar por confiança
    valid_detections = scores > conf_threshold
    
    # Aplicar NMS
    selected_indices = tf.image.non_max_suppression(
        boxes[valid_detections],
        scores[valid_detections],
        max_det,
        iou_threshold
    )
    
    return final_boxes, final_scores, final_classes
```

### Performance e Otimização

- **Modelo**: TensorFlow Lite para inferência rápida
- **Resolução**: 640x640 para balancear precisão/velocidade
- **FPS**: ~15-30 FPS dependendo do hardware
- **Latência**: ~50-100ms por frame

## Segurança

### Autenticação e Autorização

1. **Firebase Authentication**: Gerenciamento seguro de usuários
2. **JWT Tokens**: Tokens de acesso com expiração
3. **HTTPS**: Todas as comunicações criptografadas
4. **Validação Server-side**: Dupla validação (cliente + servidor)

### Proteção de Dados

```python
# Exemplo de validação segura
class AuthenticationModel:
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        if len(password) < 6:
            return False, "Senha deve ter pelo menos 6 caracteres"
        
        if not re.search(r'[A-Z]', password):
            return False, "Senha deve conter pelo menos uma letra maiúscula"
        
        if not re.search(r'[a-z]', password):
            return False, "Senha deve conter pelo menos uma letra minúscula"
        
        if not re.search(r'\d', password):
            return False, "Senha deve conter pelo menos um número"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Senha deve conter pelo menos um símbolo especial"
        
        return True, ""
```

### Configuração Segura

- **Variáveis de Ambiente**: Credenciais nunca hardcoded
- **Gitignore**: Arquivos sensíveis protegidos
- **Sanitização**: Validação de entrada em todos os endpoints

## Testes

### Estrutura de Testes

```python
# test_app.py
def test_translate_alphabet_benchmark(benchmark):
    from controllers.home_controller import HomeScreen
    screen = HomeScreen()
    screen.manager = MagicMock()
    
    def call_translate_alphabet():
        return screen.translate_alphabet()
    
    result = benchmark(call_translate_alphabet)
    assert result is None

def test_signs_detection_benchmark(benchmark):
    # Teste de performance da detecção de sinais
    # Simula frames de entrada e mede tempo de processamento
```

### Benchmarking

O sistema inclui um sistema de benchmark para monitorar performance:

```python
class BenchmarkModel:
    def __init__(self, operation: str, duration: float, metadata: dict = None):
        self.operation = operation
        self.duration = duration
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def is_slow(self, threshold_seconds: float = 2.0) -> bool:
        return self.duration > threshold_seconds
    
    def get_performance_level(self) -> str:
        if self.duration < 0.5:
            return "Excelente"
        elif self.duration < 1.0:
            return "Bom"
        elif self.duration < 2.0:
            return "Aceitável"
        else:
            return "Lento"
```

## Deploy e Produção

### Compilação para Executável

```bash
# Usando PyInstaller
pip install pyinstaller
pyinstaller --onefile --windowed --add-data "assets;assets" --add-data "views;views" main.py
```

### Deploy Firebase Functions

```bash
# Configurar projeto
firebase use --add your-project-id

# Deploy das funções
firebase deploy --only functions

# Monitoramento
firebase functions:log
```

### Variáveis de Ambiente de Produção

```bash
# Firebase Functions
firebase functions:config:set sendgrid.api_key="SG.your_key"
firebase functions:config:set sendgrid.from="your@verified-domain.com"
firebase functions:config:set firebase.api_key="your_firebase_web_api_key"
```

### Monitoramento e Logs

1. **Firebase Console**: Monitoramento de funções e autenticação
2. **SendGrid Dashboard**: Métricas de email
3. **Benchmark Logs**: Performance da aplicação local
4. **Error Handling**: Sistema robusto de tratamento de erros

---

## Conclusão

SinalizAI representa uma solução completa e robusta para tradução de LIBRAS, implementando as melhores práticas de desenvolvimento de software, incluindo:

- **Arquitetura Limpa**: Separação clara de responsabilidades
- **Segurança**: Autenticação robusta e proteção de dados
- **Performance**: Otimização para detecção em tempo real
- **Usabilidade**: Interface intuitiva e responsiva
- **Manutenibilidade**: Código bem estruturado e documentado
- **Escalabilidade**: Arquitetura preparada para crescimento

A aplicação demonstra a integração eficaz de tecnologias modernas para resolver um problema social importante, proporcionando uma ferramenta valiosa para a comunicação inclusiva.

**Versão**: 1.0.0  
**Última Atualização**: Novembro 2025  
**Licença**: Projeto Acadêmico TCC
