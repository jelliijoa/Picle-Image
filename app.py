from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

@app.route('/verify-image-similarity', methods=['POST'])
def verify_image_similarity():
    try:
        img = request.get_json()
        image_url1 = img['image_url1']
        image_url2 = img['image_url2']

        response1 = requests.get(image_url1)
        response2 = requests.get(image_url2)

        image1 = np.asarray(bytearray(response1.content), dtype="uint8")
        image2 = np.asarray(bytearray(response2.content), dtype="uint8")

        image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)

        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        resized_image1 = cv2.resize(gray_image1, (256, 256))
        resized_image2 = cv2.resize(gray_image2, (256, 256))

        similarity_score = cv2.compareHist(cv2.calcHist([resized_image1], [0], None, [256], [0, 256]),
                                           cv2.calcHist([resized_image2], [0], None, [256], [0, 256]),
                                           cv2.HISTCMP_CORREL)

        return jsonify({'similarity_score': similarity_score})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)