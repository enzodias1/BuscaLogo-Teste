from ultralytics import YOLO
import os
from PIL import Image 

class region_enhancement:
	"returns a list of cropped pil images"
	def __init__(self):
		model_path = os.path.join(os.getcwd(), "models/best.pt")
		self.region_enhancement_model = YOLO(model_path)
	
	def run_region_enhancement_pipeline(self, cropped_regions):
		parsed_results = []
		for cropped_region in cropped_regions:
			if not isinstance(cropped_region, Image.Image) or min(cropped_region.size) <= 30:
				continue 
				
			try:
				results = self.region_enhancement_model.predict(cropped_region, save=False, iou=0.5, conf=0.3, verbose=False)
				
				if results and results[0].boxes:
					for result in results[0].boxes:
						score = result.conf.cpu().item()
						if score >= 0.80:
							rectanglelabels = results[0].names[int(result.cls.cpu().item())]
							x1, y1, x2, y2 = result.xyxy.cpu().numpy()[0]
							x1, y1, x2, y2 = max(0, x1), max(0, y1), min(cropped_region.width, x2), min(cropped_region.height, y2)
							
							if x2 > x1 and y2 > y1: 
								new_image = cropped_region.crop((x1, y1, x2, y2))
								if min(new_image.size) >= 20:
									parsed_results.append(
										{"score": score, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": rectanglelabels,
										 "image": new_image})
			except Exception as e:
				print(f"Erro no region_enhancement: {e}")
				pass 
		
		return parsed_results