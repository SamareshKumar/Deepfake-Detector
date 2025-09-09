from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from enhanced_model import enhanced_model
from flask import jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction_result = enhanced_model.predict(filepath)
        if prediction_result is None:
            return jsonify({'error': 'Model not loaded'}), 500
        return jsonify({
            'filename': filename,
            'label': prediction_result['label'],
            'confidence': prediction_result['confidence']
        })
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Use enhanced model for prediction
            prediction_result = enhanced_model.predict(filepath)
            if prediction_result is None:
                return render_template('index.html', error="Model not loaded")

            return render_template('index.html', filename=filename, label=prediction_result['label'], confidence=prediction_result['confidence'])
        else:
            return render_template('index.html', error="Allowed file types are png, jpg, jpeg")
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)