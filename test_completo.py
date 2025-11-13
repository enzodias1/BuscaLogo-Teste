import sys
import os
import json
import requests
import base64
from PIL import Image
from io import BytesIO
current_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(current_dir, 'src', 'template')
sys.path.insert(1, template_dir)

print(f"Adicionando ao path: {template_dir}\n")

try:
    # Sistema 1 (Caçador)
    from main import BuscaLogo
    # Sistema 2 (Classificador)
    from classificador_logo import LogoClassifier 
    # Sistema 3 (Comparador/Deduplicador)
    from posprocessamento import region_posprocessing 
except ImportError as e:
    print(f"ERRO DE IMPORTAÇÃO: {e}")
    print("Certifique-se que todos os ficheiros .py estão em src/template/")
    sys.exit(1)

print("A carregar Sistema 1 (BuscaLogo - Caçador)")
caçador = BuscaLogo()

print("A carregar Sistema 2 (LogoClassifier - Classificador)")
classificador = LogoClassifier()

print("A carregar Sistema 3 (posprocessamento - Comparador)")
comparador = region_posprocessing()
print("Todos os sistemas carregados\n")


TEST_URL = "https://www.google.com"
TEST_SCREENSHOT = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
TEST_COMPANY = "Google"


print(f"TESTE COMPLETO: {TEST_COMPANY}")


print("\n[PASSO 1: CAÇADOR (BuscaLogo)]")
print(f"A caçar logos em {TEST_URL}...")
try:
    input_data = {
        'company_name': TEST_COMPANY,
        'screenshot_url': TEST_SCREENSHOT,
        'html_url': TEST_URL
    }
    resultado_caçador = caçador.invoke_pipeline(input_data)

    if not resultado_caçador:
        print("Caçador não encontrou logos.")
        sys.exit(0)

    print(f"Caçador encontrou {len(resultado_caçador)} logos (potenciais)")

    imagens_encontradas = []
    for logo_info in resultado_caçador:
        img_bytes = base64.b64decode(logo_info['image'])
        img_pil = Image.open(BytesIO(img_bytes))
        imagens_encontradas.append(img_pil)

except Exception as e:
    print(f"ERRO no Caçador (Sistema 1): {e}")
    sys.exit(1)

print("\n[PASSO 2: COMPARADOR]")
print(f"A remover duplicatas dos {len(imagens_encontradas)} logos encontrados")

MEU_THRESHOLD = 0.90 

indices_unicos = comparador.run_pos_processing_pipeline(imagens_encontradas, threshold=MEU_THRESHOLD)
logos_unicos = [imagens_encontradas[i] for i in indices_unicos]

print(f"Comparador filtrou de {len(imagens_encontradas)} para {len(logos_unicos)} logos únicos.")

if not logos_unicos:
    print("Nenhum logo único encontrado. Fim do teste.")
    sys.exit(0)

logo_ancora = logos_unicos[0]


print("\n[PASSO 3: CLASSIFICADOR (LogoClassifier)]")
print("A classificar a genericidade do logo âncora...")

buffered = BytesIO()
logo_ancora.convert("RGB").save(buffered, format="PNG")
logo_bytes = buffered.getvalue()

resultado_classificador = classificador.run(input_logos=[logo_bytes])
classificacao = resultado_classificador[0].classification
justificativa = resultado_classificador[0].justification

print(f"Classificação da IA: '{classificacao}'")
print(f"Justificativa: {justificativa}")


print("\n[PASSO 4: LÓGICA DO THRESHOLD DINÂMICO]")

if classificacao == "less_generic" or classificacao == "textual":
    print("Resultado: O logo é ÚNICO ('less_generic' ou 'textual').")
    print("AÇÃO: Usar um THRESHOLD BAIXO (ex: 0.80) para ser mais permissivo.")
else: # 'medium_generic' or 'more_generic'
    print("Resultado: O logo é COMUM ('medium_generic' ou 'more_generic').")
    print("AÇÃO: Usar um THRESHOLD ALTO (ex: 0.95) para ser mais rigoroso.")
