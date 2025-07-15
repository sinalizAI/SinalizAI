#!/usr/bin/env python3
"""
Script de teste para a função de reset de senha do Firebase
"""

from models.firebase_auth_model import reset_password, register

def test_with_existing_user():
    print("=== Teste com usuário existente ===")
    
    # Primeiro, vamos tentar registrar um usuário de teste
    test_email = "teste.reset@gmail.com"  # Use um email real para teste
    test_password = "TesteSenha123!"
    
    print(f"1. Tentando registrar usuário: {test_email}")
    success, response = register(test_email, test_password)
    
    if success:
        print("✅ Usuário registrado com sucesso!")
    else:
        print("ℹ️  Usuário pode já existir:", response.get('error', {}).get('message', ''))
    
    # Agora testa o reset
    print(f"\n2. Testando reset para email: {test_email}")
    success, response = reset_password(test_email)
    
    print(f"Sucesso: {success}")
    print(f"Resposta: {response}")
    
    if success:
        print("✅ Email de reset enviado com sucesso!")
        print("📧 Verifique a caixa de entrada do email (e spam/lixo eletrônico)")
    else:
        print("❌ Falha ao enviar email de reset")
        error_msg = response.get('error', {}).get('message', 'Erro desconhecido')
        print("Motivo:", error_msg)

def test_with_nonexistent_user():
    print("\n=== Teste com usuário inexistente ===")
    
    fake_email = "emailquenaoexiste123456@fakeemail.com"
    
    print(f"Testando reset para email inexistente: {fake_email}")
    success, response = reset_password(fake_email)
    
    print(f"Sucesso: {success}")
    print(f"Resposta: {response}")
    
    if not success:
        error_msg = response.get('error', {}).get('message', 'Erro desconhecido')
        print(f"❌ Como esperado, falhou: {error_msg}")

if __name__ == "__main__":
    test_with_existing_user()
    test_with_nonexistent_user()
