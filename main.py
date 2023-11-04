from flask import Flask, request, jsonify
import easyocr
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']

        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if image_file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file format'}), 400

        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        reader = easyocr.Reader(['ar'], gpu=True)
        result = reader.readtext(gray_image, paragraph=True)

        paragraph_total_confidence = 0.0

        full_text = ''
        for res in result:
            text = res[1]
            confidence = 0.0

            if len(res) > 2:
                try:
                    confidence = float(res[2])
                except ValueError:
                    pass

            full_text += text
            paragraph_total_confidence += confidence

        result = reader.readtext(gray_image, paragraph=False)

        recognized_text = []
        individual_total_confidence = 0.0

        for res in result:
            text = res[1]
            confidence = 0.0

            if len(res) > 2:
                try:
                    confidence = float(res[2])
                except ValueError:
                    pass
            recognized_text.append(text)
            individual_total_confidence += confidence

        individual_average_confidence = individual_total_confidence / len(result)

        return jsonify({
            'sentence': full_text,
            'average_confidence': individual_average_confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port="8080")

