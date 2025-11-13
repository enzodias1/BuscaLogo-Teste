import base64
from typing import *
import requests
from pathlib import Path
import ast
from dotenv import load_dotenv
import json
import os
from pydantic import BaseModel
from openai import OpenAI

class Classification(BaseModel):
    id: int
    justification: str
    classification: str
    logo: str = None

class ClassificationOutput(BaseModel):
    classifications: List[Classification]

class LogoClassifier:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        
    def run(self, input_logos: List[str] | List[bytes], return_logos: bool = False) -> List[Classification]:
        base_dir = Path(__file__).resolve().parent
        template_path = base_dir / "prompts" / "logo_classifier_prompt.txt"
        
        try:
            template = template_path.read_text()
        except FileNotFoundError:
            print(f"Não foi possível encontrar o prompt em {template_path}")
            return [] 

        template = template.format(
            textual="",
            less="",
            medium="",
            more=""
        )
        
        messages = ast.literal_eval(template)
        user_content = messages[1]['content']
        filtered_content = []
        for item in user_content:
            if item.get("type") == "image_url":
                if item["image_url"]["url"] == "data:image/jpeg;base64,":
                    continue 
            filtered_content.append(item)
        messages[1]['content'] = filtered_content

        for i, logo in enumerate(input_logos):
            if isinstance(logo, bytes):
                logo = base64.b64encode(logo).decode('utf-8')
                
            messages[1]['content'].append(
                {"type": "text", "text": f"Logo id {i}:"}
            )
            messages[1]['content'].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{logo}"}}
            )

        
        response = self.client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            temperature=1,
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "classification_output",
                    "schema": ClassificationOutput.model_json_schema()
                }
            }
        )
        
        parsed = ClassificationOutput.model_validate_json(response.choices[0].message.content)
        
        if return_logos:
            for c in parsed.classifications:
                i = int(c.id)
                c.logo = input_logos[i]
                
        return parsed.classifications