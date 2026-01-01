import easyocr

import logging

import numpy as np



# Set up logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



def initialize_reader():

    """

    Loads the EasyOCR model into memory.

    Using 'en' (English) as standard for game UI.

    """

    logger.info("Loading EasyOCR model into memory...")

    try:

        # gpu=True is critical for performance on your AWS instance

        reader = easyocr.Reader(['en'], gpu=True)

        logger.info("EasyOCR model loaded successfully.")

        return reader

    except Exception as e:

        logger.error(f"Failed to initialize EasyOCR: {e}")

        raise e



def extract_text(image_crop, reader):

    """

    Extracts text from a cropped image using the pre-initialized EasyOCR reader.

    

    Args:

        image_crop (numpy.ndarray): The cropped frame (ROI) containing the text.

        reader (easyocr.Reader): The loaded EasyOCR model.

    

    Returns:

        str: The detected text, lowercased and stripped of extra whitespace.

    """

    try:

        # easyocr expects numpy array or file path

        if image_crop is None or image_crop.size == 0:

            return ""

            

        # detail=0 returns just the list of text strings

        # detail=1 (default) returns coords, text, and confidence

        results = reader.readtext(image_crop)

        

        if not results:

            return ""

            

        # Combine all detected text blocks into one string

        full_text = " ".join([result[1] for result in results])

        

        return full_text.strip().lower()

        

    except Exception as e:

        logger.error(f"EasyOCR extraction failed: {e}")

        return ""