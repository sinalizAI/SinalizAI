#!/usr/bin/env python3
"""
Teste da função reset do app SinalizAI
"""

# Importa as funções do seu app
from models.firebase_auth_model import reset_password, register
from controllers.base_screen import BaseScreen

def test_app_reset():
    """Testa a função de reset do app"""
    
    print("📱 === TESTE DA FUNÇÃO DO APP ===\n")
    
    # Criar instância do BaseScreen para testar as funções
    base_screen = BaseScreen()
    
    # E-mail de teste
    test_email = "teste.app@example.com"
    
    print(f"📧 E-mail de teste: {test_email}")
    
    # Passo 1: Criar usuário se não existir
    print(f"\n🔄 PASSO 1: Verificando/criando usuário...")
    success, response = register(test_email, "TesteSenha123!")
    
    if success:
        print(f"✅ Usuário criado com sucesso!")
    elif "EMAIL_EXISTS" in str(response):
        print(f"✅ Usuário já existe (perfeito!)")
    else:
        print(f"❌ Erro ao criar usuário: {response}")
        return False
    
    # Passo 2: Testar validação de e-mail
    print(f"\n🔄 PASSO 2: Testando validação de e-mail...")
    if base_screen.validate_email(test_email):
        print(f"✅ Validação de e-mail: OK")
    else:
        print(f"❌ Validação de e-mail: FALHOU")
        return False
    
    # Passo 3: Testar reset de senha
    print(f"\n🔄 PASSO 3: Testando reset de senha...")
    success, response = reset_password(test_email)
    
    print(f"📊 Resultado: Sucesso = {success}")
    print(f"📄 Response: {response}")
    
    if success:
        print(f"\n🎉 === TESTE DO APP COMPLETO! ===")
        print(f"✅ Validação: OK")
        print(f"✅ Usuário: OK")
        print(f"✅ Reset: OK")
        print(f"\n📱 O app está pronto para usar!")
        print(f"📧 Verifique o e-mail: {test_email}")
        
        # Testar tratamento de erro amigável
        print(f"\n🔄 PASSO 4: Testando tratamento de erro...")
        error_message = base_screen.get_friendly_error(response)
        print(f"💬 Mensagem amigável: {error_message}")
        
        return True
    else:
        print(f"\n❌ Falha no reset de senha")
        error_message = base_screen.get_friendly_error(response)
        print(f"💬 Mensagem de erro: {error_message}")
        return False

def test_invalid_cases():
    """Testa casos inválidos"""
    
    print(f"\n🧪 === TESTE DE CASOS INVÁLIDOS ===")
    
    base_screen = BaseScreen()
    
    # Teste 1: E-mail inválido
    invalid_email = "email_invalido"
    print(f"\n📧 Testando e-mail inválido: {invalid_email}")
    
    if not base_screen.validate_email(invalid_email):
        print(f"✅ Validação rejeitou e-mail inválido: OK")
    else:
        print(f"❌ Validação aceitou e-mail inválido: ERRO")
    
    # Teste 2: E-mail não existente
    fake_email = "naoexiste12345@fake.com"
    print(f"\n📧 Testando e-mail inexistente: {fake_email}")
    
    success, response = reset_password(fake_email)
    
    if not success:
        error_message = base_screen.get_friendly_error(response)
        print(f"✅ Reset rejeitado para e-mail inexistente: {error_message}")
    else:
        print(f"❓ Reset aceito para e-mail inexistente (Firebase pode permitir isso)")
    
    print(f"\n🎯 Testes de casos inválidos concluídos!")

if __name__ == "__main__":
    # Teste principal
    if test_app_reset():
        # Se o teste principal passou, testa casos inválidos
        test_invalid_cases()
    else:
        print(f"\n❌ Teste principal falhou!")
    
    print(f"\n📱 Para testar no app real:")
    print(f"   python3 main.py")
    print(f"   • Vá para 'Esqueci a senha'")
    print(f"   • Digite: teste.app@example.com")
    print(f"   • Verifique se vai para tela de confirmação")
