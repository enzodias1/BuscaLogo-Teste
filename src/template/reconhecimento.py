import base64
import json
from io import BytesIO
import os 
from pathlib import Path  
import cv2
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from utils import parse_s3_url, get_cv2_image, get_html_text_from_url


class region_recognition_pipeline:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        self.s3_client = None  
        self.headers = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {self.OPENAI_API_KEY}"}

        try:
            base_dir = Path(__file__).resolve().parent
            prompt_dir = base_dir / "prompts"

            self.initial_prompt = (
                prompt_dir / "initial_prompt.txt").read_text()
            self.main_prompt = (prompt_dir / "main_prompt.txt").read_text()

            print(
                "Prompts 'initial_prompt.txt' e 'main_prompt.txt' carregados com sucesso.")
        except FileNotFoundError as e:
            print(
                "Certifique-se que os arquivos .txt estão na pasta src/template/prompts/")
            self.initial_prompt = "ERRO: PROMPT NÃO ENCONTRADO"
            self.main_prompt = "ERRO: PROMPT NÃO ENCONTRADO"


    def get_html_text(self, html_str):
        soup = BeautifulSoup(html_str, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    def get_html_parsed(self, s3_path):
        html_str = get_html_text_from_url(s3_path)
        return self.get_html_text(html_str)

    def convert_pil_to_base64(self, pil_image):
        im_file = BytesIO()
        try:
            pil_image.save(im_file, format="png")
        except:
            pil_image = pil_image.convert("RGB")
            pil_image.save(im_file, format="png")

        im_bytes = im_file.getvalue()
        return base64.b64encode(im_bytes).decode('utf-8')

    def encode_image(self, image_path):
        screenshot = get_cv2_image(image_path)
        if screenshot is None:
            print("Falha ao baixar screenshot para encode_image")
            return ""  
        screenshot = screenshot.copy()[0:int(
            0.3 * screenshot.shape[1]), 0:screenshot.shape[1]]
        return base64.b64encode(cv2.imencode('.jpg', screenshot)[1]).decode()

    def run_region_recognition_pipeline(self, html_path, image_path, cropped_images, model="o3"):
        html_text = self.get_html_parsed(html_path)
        prompt_template = [{"type": "text", "text": self.initial_prompt},
                           {"type": "text",
                               "text": "below is the HTML text extracted from website: "},
                           {'type': 'text', 'text': str("```")}, {
            "type": "text", "text": str(html_text)[0:5000]},
            {'type': 'text', 'text': str("```")},
            {"type": "text", "text": "below is the screenshot extracted from website: "},
            {"type": "image_url",
             "image_url": {
                 "url": f"data:image/jpeg;base64,{self.encode_image(image_path)}",
                 "detail": "high"}},
            {'type': 'text', 'text': "below are the regions cropped from the screenshot"},
            {'type': 'text', 'text': "Total regions to analyze: {}".format(len(cropped_images))}]

        for i, cropped_image in enumerate(cropped_images):
            if cropped_image is None:
                continue  
            prompt_template.extend([{'type': 'text', 'text': str("```")}, {'type': 'text', 'text': "region_{}".format(i)},
                                    {"type": "image_url",
                                     "image_url": {
                                         "url": f"data:image/jpeg;base64,{self.convert_pil_to_base64(cropped_image)}",
                                         "detail": "high"}}, {'type': 'text', 'text': str("```")}])
        prompt_template.extend(
            [{'type': 'text', 'text': str(self.main_prompt)}])

        payload = {"model": model, "messages": [
            {"role": "user", "content": prompt_template}], "temperature": 1}
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=self.headers, json=payload, timeout=300)

        return response.json()['choices'][0]['message']['content']
