from paddleocr import PaddleOCR
import os
from PIL import Image

# Initialize OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Folder containing cropped images from YOLOv5
crops_dir = 'runs/detect/exp5/crops/text_region'

# Output file to save results
output_file = 'ocr_results.txt'
with open(output_file, 'w') as out:
    for img_name in os.listdir(crops_dir):
        img_path = os.path.join(crops_dir, img_name)
        result = ocr.ocr(img_path, cls=True)

        out.write(f"\n==== {img_name} ====\n")
        for line in result:
            for box in line:
                text = box[1][0]
                confidence = box[1][1]
                out.write(f"{text} (conf: {confidence:.2f})\n")

print(f"\n✅ OCR complete! Results saved to {output_file}")

