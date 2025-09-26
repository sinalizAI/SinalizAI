# TCC - SinalizAI 📱

**Aplicativo Mobile KivyMD** para reconhecimento de letras em Libras usando Machine Learning

## � Estrutura Mobile MVC (KivyMD)

```
TCC/
├── app.py                      # 🚀 Ponto de entrada principal
├── main.py                     # 📱 Aplicação KivyMD principal
├── models/                     # 🗃️ MODELS - Modelos de dados
│   ├── firebase_auth_model.py
│   ├── email_service.py
│   └── legal_acceptance_model.py
├── views/                      # 📱 VIEWS - Telas e interfaces KivyMD
│   ├── screen_manager.py       # Gerenciador de telas
│   ├── login/                  # Telas de autenticação
│   ├── home_page/             # Tela principal
│   ├── profile_page/          # Perfil do usuário
│   ├── welcome_page/          # Tela de boas-vindas
│   └── [outras_telas]/        # Demais telas do app
├── controllers/                # 🎮 CONTROLLERS - Lógica de controle
│   └── [controladores]/       # Lógica de negócio das telas
├── services/                   # ⚙️ SERVICES - Serviços externos
│   └── ml/                     # 🤖 Machine Learning
│       ├── alfabeto.pt         # Modelo treinado YOLOv5
│       ├── modelo_extraido/    # Modelo serializado
│       ├── treinamento_colab.ipynb
│       └── yolov5/             # Framework YOLOv5
├── utils/                      # 🛠️ UTILS - Utilitários e helpers
├── assets/                     # 🎨 ASSETS - Recursos visuais
│   ├── fonts/                  # Fontes personalizadas
│   │   ├── Athiti/
│   │   ├── PT_Serif/
│   │   └── palanquin/
│   └── images/                 # Imagens e ícones
│       ├── SinalizAI.png
│       ├── welcome_image.jpg
│       └── perfil_semfoto.png
└── config/                     # ⚙️ Configurações
    └── kivy_config.py          # Configurações do Kivy
```

## 🏗️ Arquitetura MVC Mobile (KivyMD)

### **Model (Modelo)**
- `models/` - Modelos de dados, entidades e lógica de negócio
- `services/` - Serviços de ML, autenticação e APIs

### **View (Visão)**  
- `views/` - Telas e componentes de interface KivyMD
- `assets/` - Recursos visuais (fontes, imagens, ícones)

### **Controller (Controlador)**
- `controllers/` - Controladores de tela e lógica de coordenação
- `utils/` - Utilitários e funções auxiliares

## 🚀 Como Executar

```bash
# Ativar ambiente conda
conda activate kivymd_app

# Executar aplicação principal
python app.py

# Executar detecção de letras (ML)
cd services/ml/yolov5
python detect.py --source 0
```

## 🤖 Machine Learning

O sistema utiliza YOLOv5 para detecção e reconhecimento de letras em Libras:

- **Modelo**: `services/ml/alfabeto.pt`
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

🔬 **Projeto de TCC** - Sistema de reconhecimento de Libras com IA'