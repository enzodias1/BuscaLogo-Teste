import os
import re
from io import BytesIO
import cv2
import requests 
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

def parse_s3_url(url):
	return None, None, None

def get_file_from_S3(bucket, key, s3):
	raise NotImplementedError

def get_cv2_image(image_url):
	try:
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
		}
		response = requests.get(image_url, headers=headers, timeout=10)
		response.raise_for_status() 
		
		img_str = np.frombuffer(response.content, np.uint8)
		image = cv2.imdecode(img_str, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError("Não foi possível decodificar a imagem.")
		return image
	except Exception as e:
		print(f"Erro ao baixar/processar imagem de {image_url}: {e}")
		return None 

def get_pil_image(image_url):
	try:
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
		}
		response = requests.get(image_url, headers=headers, timeout=10)
		response.raise_for_status()
		return Image.open(BytesIO(response.content)).convert('RGB')
	except Exception as e:
		print(f"Erro ao baixar/processar imagem PIL de {image_url}: {e}")
		return None

def download_models_from_s3(models, bucket, destination_path):
	if not os.path.exists(os.path.join(os.getcwd(), "models")):
		try:
			os.mkdir(os.path.join(os.getcwd(), "models"))
		except OSError as e:
			print(f"Erro ao criar diretório models: {e}")

def get_pil_image_RGBA(image_url):
	try:
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
		}
		response = requests.get(image_url, headers=headers, timeout=10)
		response.raise_for_status()
		return Image.open(BytesIO(response.content)).convert('RGBA')
	except Exception as e:
		print(f"Erro ao baixar/processar imagem RGBA de {image_url}: {e}")
		return None

def numpy_to_pil(array):
	return Image.fromarray(array.astype(np.uint8))

def Image_open(image_path):
	return Image.open(image_path)

def get_html_text_from_url(s3_path): #
	html_url = s3_path 
	try:
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
		}
		response = requests.get(html_url, headers=headers, timeout=10)
		response.raise_for_status()
		html_str = response.content.decode("utf-8") 
		return html_str
	except Exception as e:
		print(f"Erro ao baixar HTML de {html_url}: {e}")
		return ""

def get_image_urls(html_content):
	"""
	Extracts all image URLs from an HTML string.
 
	Args:
		html_content: The HTML content as a string.
  
	Returns:
		A list of image URLs.
	"""
	soup = BeautifulSoup(html_content, 'html.parser')
	img_tags = soup.find_all('img')
	img_urls = [img['src'] for img in img_tags if 'src' in img.attrs]
	return img_urls

def get_images_from_html(html_path):
	html_str = get_html_text_from_url(html_path)
	if not html_str:
		return [] 
		
	image_urls = get_image_urls(html_str)
	image_urls = [image_url for image_url in image_urls if 'logo' in image_url]
	images = []
	for image_url in image_urls:
		try:
			if image_url.startswith('//'):
				image_url = 'https:' + image_url
			elif image_url.startswith('/'):
				from urllib.parse import urlparse
				base_url = "{0.scheme}://{0.netloc}".format(urlparse(html_path))
				image_url = base_url + image_url

			response = requests.get(image_url, stream=True, timeout=10)
			response.raise_for_status()
			image = Image.open(BytesIO(response.content))
			width, height = image.size
			if max(width, height) < 9000:
				images.append(image)
		except:
			pass
	return images