# TCC - SinalizAI

Sistema de reconhecimento de letras em Libras usando Machine Learning

## 📁 Estrutura do Projeto (MVC)

```
TCC/
├── app.py                      # 🚀 Ponto de entrada principal
├── app/                        # 📱 Aplicação principal
│   ├── main.py                 # Core da aplicação
│   ├── controllers/            # 🎮 Controladores (lógica de controle)
│   ├── models/                 # 🗃️ Modelos de dados
│   ├── views/                  # 🖼️ Interface de usuário (UI)
│   ├── helpers/                # 🛠️ Utilitários e helpers
│   └── services/               # ⚙️ Camada de serviços
│       └── ml/                 # 🤖 Serviços de Machine Learning
│           ├── alfabeto.pt     # Modelo treinado principal
│           ├── alfabeto.torchscript.zip
│           ├── modelo_extraido/
│           ├── treinamento_colab.ipynb
│           └── yolov5/         # Framework YOLOv5
├── config/                     # ⚙️ Configurações
│   └── kivy_config.py
├── static/                     # 🎨 Arquivos estáticos
│   ├── css/
│   ├── js/
│   └── images/
├── public/                     # 🌐 Arquivos públicos
├── logs/                       # 📝 Logs da aplicação
└── tests/                      # 🧪 Testes
```

## 🏗️ Arquitetura MVC

### **Model (Modelo)**
- `app/models/` - Modelos de dados e estruturas
- `app/services/ml/` - Lógica de Machine Learning

### **View (Visão)**  
- `app/views/` - Interface de usuário
- `static/` - Assets visuais (CSS, JS, imagens)

### **Controller (Controlador)**
- `app/controllers/` - Lógica de controle e coordenação
- `app/helpers/` - Funções auxiliares

## 🚀 Como Executar

```bash
# Ativar ambiente conda
conda activate kivymd_app

# Executar aplicação principal
python app.py

# Executar detecção de letras (ML)
cd app/services/ml/yolov5
python detect.py --source 0
```

## 🤖 Machine Learning

O sistema utiliza YOLOv5 para detecção e reconhecimento de letras em Libras:

- **Modelo**: `app/services/ml/alfabeto.pt`
- **Framework**: YOLOv5
- **Treinamento**: Notebook Colab disponível
- **Deployment**: TorchScript para produção

## 📋 Benefícios da Nova Estrutura

✅ **Separação clara** de responsabilidades (MVC)  
✅ **Escalabilidade** - fácil adicionar novos módulos  
✅ **Manutenibilidade** - código organizado e limpo  
✅ **Padrões profissionais** de desenvolvimento  
✅ **Testabilidade** - estrutura propícia para testes  
✅ **Deploy amigável** - organização para produção  

---

🔬 **Projeto de TCC** - Sistema de reconhecimento de Libras com IA