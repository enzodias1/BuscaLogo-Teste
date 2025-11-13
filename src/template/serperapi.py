import http.client
import json
from io import BytesIO
from dotenv import load_dotenv
import os, base64
import requests
from PIL import Image


class get_images_from_google:
    def __init__(self):
        load_dotenv()
        self.SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
        
        self.headers = {
            'X-API-KEY': self.SERPER_API_KEY,  
            'Content-Type': 'application/json'
        }
    
    def get_images(self, query):
        self.conn = http.client.HTTPSConnection("google.serper.dev", timeout=10)
        payload = json.dumps({
            "q": query,
            "gl": "br"
        })
        try:
            self.conn.request("POST", "/images", payload, self.headers)
            res = self.conn.getresponse()
            data = res.read()
            r = json.loads(data.decode("utf-8"))
            images_url = [image_url['imageUrl'] for image_url in r['images'] if '.png' in image_url['imageUrl'] or '.jpg' in image_url['imageUrl']]
            images = []
            images_url = list(set(images_url))
            images_url = [image_url for image_url in images_url if 'transparent' not in image_url]

            headers = {"authority": "www.google.com",
                     "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                    "accept-language": "en-US,en;q=0.9",
                    "cache-control": "max-age=0",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}

            for image_url in images_url[0:50]:
                try:
                    response = requests.get(str(image_url), stream=True, headers=headers, timeout=20)
                    response.raise_for_status()

                    image = Image.open(BytesIO(response.content))
                    width, height = image.size
                    if max(width, height) < 9000:
                        images.append(image)
                except Exception as e:
                    print("serperAPI timeout: ", e)
            print(f" --- INSIDE SERPERAPI --- {len(images)} images found")
            return images
        except Exception as e:
            print("error", e)
            return []