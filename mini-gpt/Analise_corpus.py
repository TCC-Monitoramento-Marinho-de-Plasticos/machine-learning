"""
Script para carregar e analisar o meu_corpus.txt
"""

import os
import sys
from pathlib import Path
from collections import Counter

def encontrar_arquivo():
    """Encontra o arquivo meu_corpus.txt"""
    
    # Caminhos possíveis
    caminhos = [
        "/home/marise/Downloads/MiniGPTv01/meu_corpus.txt",
        "./meu_corpus.txt",
        "../meu_corpus.txt",
        "meu_corpus.txt"
    ]
    
    for caminho in caminhos:
        if os.path.exists(caminho):
            return caminho
    
    # Se não encontrou, procurar recursivamente
    for root, dirs, files in os.walk("/home/marise/Downloads/"):
        for file in files:
            if file == "meu_corpus.txt":
                return os.path.join(root, file)
    
    return None

def analisar_corpus(caminho_arquivo):
    """Analisa o corpus completo"""
    
    print(f"📁 Carregando arquivo: {caminho_arquivo}")
    print("=" * 60)
    
    try:
        # Ler arquivo
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            texto = f.read()
        
        # Análises básicas
        total_chars = len(texto)
        chars_unicos = len(set(texto))
        linhas = texto.split('\n')
        paragrafos = [p for p in texto.split('\n\n') if p.strip()]
        
        # Análise de palavras
        import re
        palavras = re.findall(r'\b\w+\b', texto.lower())
        palavras_unicas = len(set(palavras))
        
        # Caracteres mais comuns
        char_freq = Counter(texto)
        chars_comuns = char_freq.most_common(10)
        
        # Relatório
        print(f"📊 ESTATÍSTICAS GERAIS:")
        print(f"   📝 Total de caracteres: {total_chars:,}")
        print(f"   🔤 Caracteres únicos: {chars_unicos}")
        print(f"   📚 Total de palavras: {len(palavras):,}")
        print(f"   🎯 Palavras únicas: {palavras_unicas:,}")
        print(f"   📄 Linhas: {len(linhas):,}")
        print(f"   📋 Parágrafos: {len(paragrafos):,}")
        
        print(f"\n🔤 CARACTERES MAIS FREQUENTES:")
        for char, freq in chars_comuns:
            if char == ' ':
                print(f"   [ESPAÇO]: {freq:,} ({freq/total_chars*100:.1f}%)")
            elif char == '\n':
                print(f"   [QUEBRA]: {freq:,} ({freq/total_chars*100:.1f}%)")
            else:
                print(f"   '{char}': {freq:,} ({freq/total_chars*100:.1f}%)")
        
        print(f"\n📖 AMOSTRA DO TEXTO:")
        print("-" * 60)
        print(texto[:300])
        print("-" * 60)
        
        # Vocabulário para MiniGPT
        chars_ordenados = sorted(list(set(texto)))
        print(f"\n🧠 VOCABULÁRIO PARA MINIGPT:")
        print(f"   Tamanho do vocabulário: {len(chars_ordenados)}")
        print(f"   Primeiros 30 chars: {''.join(chars_ordenados[:30])}")
        
        return texto
        
    except Exception as e:
        print(f"❌ Erro ao processar arquivo: {e}")
        return None

def main():
    """Função principal"""
    print("🚀 ANALISADOR DE CORPUS - MiniGPT")
    print("=" * 60)
    
    # Encontrar arquivo
    caminho = encontrar_arquivo()
    
    if caminho:
        print(f"✅ Arquivo encontrado: {caminho}")
        texto = analisar_corpus(caminho)
        
        if texto:
            print(f"\n✅ Análise concluída com sucesso!")
            print(f"📁 Arquivo pronto para uso no MiniGPT")
        
    else:
        print("❌ Arquivo meu_corpus.txt não encontrado!")
        print("\n🔍 Verifique se o arquivo existe em:")
        print("   - /home/marise/Downloads/MiniGPTv01/meu_corpus.txt")
        print("   - No diretório atual")
        
        # Listar arquivos .txt disponíveis
        print(f"\n📁 Arquivos .txt encontrados em Downloads:")
        downloads = "/home/marise/Downloads/"
        if os.path.exists(downloads):
            for root, dirs, files in os.walk(downloads):
                for file in files:
                    if file.endswith('.txt'):
                        print(f"   📝 {os.path.join(root, file)}")

if __name__ == "__main__":
    main()