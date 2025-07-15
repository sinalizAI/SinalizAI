#!/usr/bin/env python3
"""
Teste simples das funções Firebase sem dependências do Kivy
"""

import re
from models.firebase_auth_model import reset_password, register

def validate_email_simple(email):
    """Validação simples de e-mail"""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def test_firebase_functions():
    """Testa as funções do Firebase"""
    
    print("🔬 === TESTE DAS FUNÇÕES FIREBASE ===\n")
    
    # E-mail de teste
    test_email = "usuario.teste@gmail.com"
    test_password = "MinhaSenh@123"
    
    print(f"📧 E-mail de teste: {test_email}")
    print(f"🔑 Senha de teste: {test_password}")
    
    # Passo 1: Validar e-mail
    print(f"\n🔄 PASSO 1: Validando formato do e-mail...")
    if validate_email_simple(test_email):
        print(f"✅ E-mail válido!")
    else:
        print(f"❌ E-mail inválido!")
        return False
    
    # Passo 2: Registrar usuário (se não existir)
    print(f"\n🔄 PASSO 2: Registrando usuário...")
    success, response = register(test_email, test_password)
    
    if success:
        print(f"✅ Usuário registrado com sucesso!")
        print(f"🆔 ID: {response.get('localId', 'N/A')}")
    elif "EMAIL_EXISTS" in str(response):
        print(f"✅ Usuário já existe (perfeito para o teste!)")
    else:
        print(f"❌ Erro ao registrar: {response}")
        print(f"ℹ️  Vamos tentar o reset mesmo assim...")
    
    # Passo 3: Testar reset de senha
    print(f"\n🔄 PASSO 3: Enviando e-mail de reset...")
    success, response = reset_password(test_email)
    
    print(f"📊 Resultado: {success}")
    print(f"📄 Response: {response}")
    
    if success:
        print(f"\n🎉 === TESTE COMPLETO COM SUCESSO! ===")
        print(f"✅ Validação: OK")
        print(f"✅ Usuário: OK") 
        print(f"✅ Reset de senha: OK")
        print(f"\n📧 E-MAIL ENVIADO PARA: {test_email}")
        print(f"📬 Verifique:")
        print(f"   • Caixa de entrada")
        print(f"   • Pasta de spam/lixo eletrônico")
        print(f"   • Remetente: noreply@sinalizai.firebaseapp.com")
        print(f"   • Assunto: Redefinir sua senha - SinalizAI")
        
        return True
    else:
        print(f"\n❌ Falha no envio do e-mail de reset")
        error_code = response.get("error", {}).get("message", "")
        
        if "EMAIL_NOT_FOUND" in error_code:
            print(f"💡 E-mail não encontrado no Firebase")
        elif "INVALID_EMAIL" in error_code:
            print(f"💡 Formato de e-mail inválido")
        else:
            print(f"💡 Erro: {error_code}")
        
        return False

def test_error_cases():
    """Testa casos de erro"""
    
    print(f"\n🧪 === TESTE DE CASOS DE ERRO ===")
    
    # Teste 1: E-mail inválido
    print(f"\n📧 Teste 1: E-mail com formato inválido")
    invalid_email = "email.invalido.sem.arroba"
    
    if not validate_email_simple(invalid_email):
        print(f"✅ Validação rejeitou '{invalid_email}': OK")
    else:
        print(f"❌ Validação aceitou e-mail inválido!")
    
    # Teste 2: E-mail inexistente
    print(f"\n📧 Teste 2: E-mail inexistente no Firebase")
    fake_email = "emailquenaoexiste999888@dominiofalso.net"
    
    success, response = reset_password(fake_email)
    
    if not success:
        error_code = response.get("error", {}).get("message", "")
        print(f"✅ Reset rejeitado para e-mail inexistente: {error_code}")
    else:
        print(f"⚠️  Firebase aceitou e-mail inexistente (pode ser comportamento normal)")
    
    print(f"\n🎯 Testes de erro concluídos!")

if __name__ == "__main__":
    print("🚀 Iniciando testes das funções Firebase...")
    
    # Teste principal
    if test_firebase_functions():
        # Se passou, testa casos de erro
        test_error_cases()
        
        print(f"\n📱 === PRÓXIMOS PASSOS ===")
        print(f"1. ✅ Firebase configurado e funcionando")
        print(f"2. ✅ E-mails sendo enviados")
        print(f"3. 📱 Teste no app real:")
        print(f"   • Execute: python3 main.py")
        print(f"   • Vá para 'Esqueci a senha'")
        print(f"   • Digite um e-mail existente")
        print(f"   • Verifique se vai para tela de confirmação")
        print(f"4. 📧 Verifique sua caixa de e-mail!")
        
    else:
        print(f"\n❌ Teste principal falhou. Verifique:")
        print(f"   • Conectividade com internet")
        print(f"   • Configurações do Firebase")
        print(f"   • Se o usuário existe no Firebase Authentication")
