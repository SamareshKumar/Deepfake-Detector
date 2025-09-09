from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from enhanced_model import enhanced_model # Import the instantiated model
import json # For pretty printing JSON in debug

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['XAI_FOLDER'] = 'static/xai_explanations' # New folder for XAI images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload and XAI folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['XAI_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/predict_json', methods=['POST'])
def predict_json():
    """API endpoint for prediction (without XAI details)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction_result, error = enhanced_model.predict(filepath)
        if error:
            return jsonify({'error': error}), 500
        return jsonify({
            'filename': filename,
            'label': prediction_result['label'],
            'confidence': prediction_result['confidence'],
            'raw_prediction': prediction_result['raw_prediction']
        })
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_info = None
    xai_results = None
    error_message = None
    original_image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = "No file part"
        file = request.files['file']
        if file.filename == '':
            error_message = "No selected file"
        
        if not error_message and file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            original_image_url = url_for('uploaded_file', filename=filename)

            # Use enhanced model for prediction and XAI
            xai_results = enhanced_model.analyze_image_with_xai(filepath)
            
            if "error" in xai_results:
                error_message = xai_results['error']
            else:
                prediction_info = xai_results['prediction']
                # XAI image base64 strings will be passed directly
                # You might want to remove the temporary file after processing if storage is a concern
                # os.remove(filepath) 
        else:
            if not error_message: # If no specific error was set yet
                error_message = "Allowed file types are png, jpg, jpeg"
                
    return render_template('index.html', 
                           prediction_info=prediction_info,
                           xai_results=xai_results,
                           error=error_message,
                           original_image_url=original_image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/xai_explanations/<filename>')
def xai_explanation_file(filename):
    return redirect(url_for('static', filename='xai_explanations/' + filename), code=301)

if __name__ == '__main__':
    # Make sure to have a 'model' directory in the root and place your .h5 model there
    # Example: 'model/deepfake_detector_model.h5'
    app.run(debug=True)