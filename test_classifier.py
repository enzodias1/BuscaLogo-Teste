import sys
import os
import base64
import json

current_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(current_dir, 'src', 'template')
sys.path.insert(1, template_dir)

print(f"Adicionando ao path: {template_dir}")

try:
    from classificador_logo import ClassificadorLogo
except ImportError as e:
    print(e)
    sys.exit(1)

try:
    classifier = ClassificadorLogo()
except Exception as e:
    sys.exit(1)

image_file = 'logo_nike.png' 
try:
    print(f"Abrindo a imagem de teste: {image_file}...")
    with open(image_file, "rb") as f:
        image_bytes = f.read()

    print("Enviando logo para classificação no GPT-4")
    results = classifier.run(input_logos=[image_bytes])

    print("\nCLASSIFICAÇÃO DA IA")
    print(json.dumps(results[0].model_dump(), indent=2))

except FileNotFoundError:
    print(f"Erro: Arquivo '{image_file}' não encontrado.")
    print("Por favor, baixe um logo e salve-o como 'logo_teste.png' na pasta principal.")
except Exception as e:
    print(f"Ocorreu um erro durante a classificação: {e}")