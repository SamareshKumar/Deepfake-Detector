import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shap
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries
import cv2
import os

class EnhancedModel:
    def __init__(self, model_path='model/deepfake_detector_model.h5'):
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = (img_array / 127.5) - 1  # Rescale to [-1, 1]
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img_path):
        if self.model is None:
            return None, "Model not loaded"
        img = self.preprocess_image(img_path)
        prediction = self.model.predict(img)[0][0]
        label = "Fake" if prediction > 0.5 else "Real"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        return {
            'label': label,
            'confidence': round(confidence * 100, 2),
            'raw_prediction': prediction
        }

    def explain_with_shap(self, img_path):
        if self.model is None:
            return "Model not loaded"
        img = self.preprocess_image(img_path)
        # SHAP explanation (simplified)
        explainer = shap.Explainer(self.model)
        shap_values = explainer(img)
        return shap_values

    def explain_with_lime(self, img_path):
        if self.model is None:
            return "Model not loaded"
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = img.astype('double') / 255.0

        def predict_fn(images):
            processed = []
            for im in images:
                im = np.expand_dims(im, axis=0)
                processed.append(im)
            processed = np.array(processed)
            return self.model.predict(processed)

        explainer = lime.lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        img_boundry = mark_boundaries(temp / 255.0, mask)
        return img_boundry

    def textual_explanation(self, prediction_result):
        if prediction_result['label'] == 'Fake':
            return f"The model predicts this image is {prediction_result['label']} with {prediction_result['confidence']}% confidence. This suggests potential manipulation in the image."
        else:
            return f"The model predicts this image is {prediction_result['label']} with {prediction_result['confidence']}% confidence. This indicates the image appears authentic."

    def uncertainty_quantification(self, img_path, num_samples=10):
        if self.model is None:
            return "Model not loaded"
        img = self.preprocess_image(img_path)
        predictions = []
        for _ in range(num_samples):
            pred = self.model.predict(img)[0][0]
            predictions.append(pred)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        return {
            'mean_prediction': mean_pred,
            'std_deviation': std_pred,
            'uncertainty': std_pred / mean_pred if mean_pred != 0 else 0
        }

    def analyze_image_with_xai(self, img_path):
        prediction = self.predict(img_path)
        if prediction is None:
            return {"error": "Model not loaded"}
        explanation_shap = self.explain_with_shap(img_path)
        explanation_lime = self.explain_with_lime(img_path)
        textual_exp = self.textual_explanation(prediction)
        uncertainty = self.uncertainty_quantification(img_path)
        return {
            'prediction': prediction,
            'shap_explanation': str(explanation_shap),  # Simplified
            'lime_explanation': 'LIME explanation generated',  # Placeholder
            'textual_explanation': textual_exp,
            'uncertainty': uncertainty
        }

# Instantiate the enhanced model
enhanced_model = EnhancedModel()
