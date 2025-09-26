# Guia de Segurança - SinalizAI

## 🔒 Configuração Segura das Credenciais

### Problema Anterior
Anteriormente, as chaves da API do Firebase estavam expostas diretamente no código fonte, o que representa um risco de segurança significativo.

### Solução Implementada
Agora usamos variáveis de ambiente para armazenar credenciais sensíveis:

1. **Arquivo .env**: Contém as credenciais reais (NÃO deve ser versionado)
2. **Arquivo .env.example**: Template das variáveis necessárias (pode ser versionado)
3. **config_manager.py**: Gerencia o carregamento seguro das configurações

### Como Configurar

#### 1. Copiar o template
```bash
cp .env.example .env
```

#### 2. Editar o arquivo .env com suas credenciais
```bash
nano .env
```

#### 3. Preencher as variáveis:
```
FIREBASE_API_KEY=sua_api_key_aqui
FIREBASE_AUTH_DOMAIN=seu_projeto.firebaseapp.com
FIREBASE_PROJECT_ID=seu_projeto_id
FIREBASE_STORAGE_BUCKET=seu_projeto.firebasestorage.app
FIREBASE_MESSAGING_SENDER_ID=123456789
FIREBASE_APP_ID=1:123456789:web:abcdef123456
```

### Benefícios da Nova Abordagem

✅ **Segurança**: Credenciais não ficam expostas no código fonte
✅ **Flexibilidade**: Fácil configuração para diferentes ambientes (dev, prod)
✅ **Versionamento**: .env é ignorado pelo Git automaticamente
✅ **Colaboração**: Outros desenvolvedores podem usar .env.example como base

### Estrutura dos Arquivos

```
config/
├── config_manager.py     # Gerenciador de configuração
├── kivy_config.py       # Configurações do Kivy
.env                     # Suas credenciais (NÃO versionar)
.env.example             # Template (pode versionar)
.gitignore               # Ignora .env e outros arquivos sensíveis
```

### Para Desenvolvedores

Se você é um novo desenvolvedor no projeto:

1. Clone o repositório
2. Copie `.env.example` para `.env`
3. Preencha suas próprias credenciais no `.env`
4. Execute o projeto normalmente

### Importante ⚠️

- **NUNCA** commite o arquivo `.env`
- **SEMPRE** use `.env.example` como template
- **SEMPRE** adicione novos campos em ambos os arquivos
- **REVOGUE** credenciais expostas acidentalmente

### Validação

O sistema valida automaticamente se todas as variáveis necessárias estão configuradas. Se alguma estiver faltando, você verá um erro claro indicando qual variável está ausente.

### Rotação de Credenciais

Para trocar credenciais:

1. Gere novas credenciais no Firebase Console
2. Atualize o arquivo `.env`  
3. Revogue as credenciais antigas no Firebase Console