import cv2
import numpy as np
import openvino as ov
import os 
from PIL import Image 

class region_posprocessing:
	def __init__(self):
		core = ov.Core()
		model_name = "h-generic-reid-0001"
		model_path_xml = f"{model_name}.xml"

		from sentence_transformers import SentenceTransformer
		# 'clip-ViT-B-32' é o modelo padrão-ouro para embeddings de imagem.
		self.recognition_model = SentenceTransformer('clip-ViT-B-32')

	def extract_embeddings(self, image):
		return self.recognition_model.encode(image)
		
	def run_pos_processing_pipeline(self, pil_images, threshold=0.95):
		pil_images = [image for image in pil_images if isinstance(image, Image.Image) and (min(image.size) > 0) and ('L' not in image.getbands())]
		
		if not pil_images:
			return [] 
			
		print(f"A criar 'impressões digitais' para {len(pil_images)} imagens...")
		embeddings = [self.extract_embeddings(pil_image) for pil_image in pil_images]
	
		from sklearn.metrics.pairwise import cosine_similarity
		
		distance_matrix = cosine_similarity(embeddings)
		indices = np.where(distance_matrix >= threshold)
		same_image_indices = []
		for i in range(len(indices[0])):
			if indices[0][i] < indices[1][i]:
				same_image_indices.append((indices[0][i], indices[1][i]))
		
		duplicate_indices = set()
		for i, j in same_image_indices:
			duplicate_indices.add(j)
			
		unique_indices = [i for i, img in enumerate(pil_images) if i not in duplicate_indices]
		return unique_indices