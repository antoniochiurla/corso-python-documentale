import cv2
import numpy as np

class ImageEnhancer:
    @staticmethod
    def binarize_for_ocr(image_path):
        """Pipeline completa per preparare un'immagine all'OCR."""
        # 1. Carica in scala di grigi
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # 2. Rimuove rumore iniziale
        img = cv2.medianBlur(img, 3)
        
        # 3. Deskewing (Semplificato)
        angle = ImageEnhancer._get_skew_angle(img)
        if abs(angle) > 0.5:
            img = ImageEnhancer._rotate_image(img, angle)
            
        # 4. Adaptive Thresholding per gestire ombre e carta vecchia
        # block_size=11 e C=2 sono parametri standard da regolare
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary

    @staticmethod
    def _get_skew_angle(img):
        # Inversione per trovare i contorni del testo
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        return -(90 + angle) if angle < -45 else -angle

    @staticmethod
    def _rotate_image(img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Esempio di utilizzo
if __name__ == "__main__":
    enhancer = ImageEnhancer()
    processed = enhancer.binarize_for_ocr("input_sporco.jpg")
    cv2.imwrite("output_pulito.png", processed)
