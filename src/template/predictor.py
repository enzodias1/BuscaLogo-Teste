from __future__ import print_function
import sys
import os
import json
import ast
import flask
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, current_dir)
print(f"Adicionado ao path: {current_dir}")
from main import BuscaLogo
from utils import download_models_from_s3

bucket = 'buscalogo-artifacts' 
models = ['models/best.pt'] 
destination_path = os.path.join(os.getcwd(), 'models')
download_models_from_s3(models, bucket, destination_path)

print("Iniciando BuscaLogo")
try:
	buscalogo = BuscaLogo()
except Exception as e:
	print(f"Erro: {e}")
	BuscaLogo = None 

class ScoringService(object):
	@classmethod
	def predict(cls, input):
		if buscalogo is None:
			print("Erro: A predição foi chamada, mas o BuscaLogo não foi inicializado.")
			
		try:
			print(f"Recebida invocação: {input}")
			input_data = ast.literal_eval(input)
			
			result = buscalogo.invoke_pipeline(input_data)
			
			input_data['result'] = result
			return json.dumps(input_data)
		except Exception as e:
			return json.dumps({"error": str(e)}, indent=4, ensure_ascii=False)

app = flask.Flask(__name__)
app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False

@app.route('/ping', methods=['GET'])
def ping():
	"""Verifica se o contêiner está saudável E se os modelos carregaram."""

	if buscalogo is not None:
		status = 200
		response_text = "pong (modelos de IA carregados com sucesso)"
	else:
		status = 500 
		response_text = "ERRO (Falha ao carregar modelos de IA. Verifique o log.)"
		
	return flask.Response(response=response_text, status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
	input = flask.request.data.decode('utf-8')
	predictions = ScoringService.predict(input)
	return flask.Response(response=str(predictions), status=200)

if __name__ == '__main__':

	print("Iniciando servidor Flask de desenvolvimento no http://127.0.0.1:5000")
	app.run(host='0.0.0.0', port=5000, debug=True)