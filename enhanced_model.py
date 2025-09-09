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
import matplotlib.pyplot as plt
import io
import base64

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EnhancedModel:
    def __init__(self, model_path='model/deepfake_detector_model.h5'):
        self.model_path = model_path
        try:
            # Custom objects might be needed if custom layers or loss functions are used
            self.model = load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.model = None

    def preprocess_image(self, img_path, target_size=(128, 128)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = (img_array / 127.5) - 1  # Rescale to [-1, 1] as per original
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img # Return original img for LIME/SHAP display

    def predict(self, img_path):
        if self.model is None:
            return None, "Model not loaded"
        img_array, _ = self.preprocess_image(img_path)
        prediction = self.model.predict(img_array)[0][0]
        label = "Fake" if prediction > 0.5 else "Real"
        # Confidence should reflect how far it is from 0.5
        confidence = abs(prediction - 0.5) * 2
        return {
            'label': label,
            'confidence': round(confidence * 100, 2),
            'raw_prediction': prediction
        }

    # Helper function to convert matplotlib plot to base64 image
    def fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # Close the figure to free memory
        return img_base64

    def explain_with_shap(self, img_path):
        if self.model is None:
            return "Model not loaded", None

        # For DeepExplainer, we need a background dataset.
        # Since we don't have training data, we'll create a simple dummy background.
        # In a real scenario, you'd use a small representative sample of your training data.
        dummy_background = np.zeros((1, 128, 128, 3)) # Single black image as background
        # You might also consider using K-means on a subset of training data for a better background

        try:
            # DeepExplainer is suitable for TensorFlow/Keras models
            explainer = shap.DeepExplainer(self.model, dummy_background)
            img_array, original_img = self.preprocess_image(img_path)
            shap_values = explainer.shap_values(img_array)

            # Assuming binary classification, shap_values will be a list of two arrays.
            # We are interested in the 'fake' class (index 0 or 1 depending on model output)
            # Let's assume the model outputs probability for 'fake' being the second class (index 1 for shap values)
            # Or if it's a single output node, shap_values[0] directly.
            # We'll visualize the impact on the positive class prediction.
            
            # The original model output is a single value, so shap_values will be a list of one array
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[0][0] # Take the first (and only) explanation array for the image
            else:
                shap_values_to_plot = shap_values[0] # If shap_values is already an array, take the first image's values

            # Original image needs to be un-normalized for plotting
            original_img_np = image.img_to_array(original_img) / 255.0

            # Plotting with SHAP's image plot function
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            shap.image_plot([shap_values_to_plot], -img_array[0], show=False) # Negative for positive contributions
            ax.set_title("SHAP Explanation")
            plt.tight_layout()
            
            img_base64 = self.fig_to_base64(fig)
            return "SHAP explanation generated", img_base64

        except Exception as e:
            print(f"Error during SHAP explanation: {e}")
            return f"Error during SHAP explanation: {e}", None


    def explain_with_lime(self, img_path):
        if self.model is None:
            return "Model not loaded", None
        
        # LIME expects images in [0, 1] range
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        img = cv2.resize(img, (128, 128))
        img_0_1 = img.astype('double') / 255.0 # Keep this in [0,1] for LIME explainer

        def predict_fn(images):
            # LIME provides images in [0,1], convert to model's expected range [-1,1]
            processed_images = (images * 2) - 1
            return self.model.predict(processed_images)

        explainer = lime.lime_image.LimeImageExplainer()
        
        # Assume the model outputs a single value for probability of being 'Fake'.
        # LIME explainer usually expects a probability for each class.
        # We need to wrap our single-output model to return probabilities for two classes [prob_real, prob_fake].
        def lime_predict_proba(images):
            raw_predictions = predict_fn(images) # This returns a batch of single probabilities for 'fake'
            prob_fake = raw_predictions.flatten()
            prob_real = 1 - prob_fake
            return np.transpose(np.array([prob_real, prob_fake])) # Shape (num_samples, 2)

        try:
            # top_labels=1: We only care about the top predicted label (either real or fake)
            # We need to specify the label for which to generate the explanation.
            # Since our model outputs a single 'fake' probability, we'll explain for the 'fake' class (index 1).
            # If the prediction is 'Real', we still explain for 'Fake' but it will show regions that reduce 'Fake' probability.
            prediction_output = self.predict(img_path)
            target_label_idx = 1 if prediction_output['label'] == 'Fake' else 0 # Explain for 'Fake' (1) or 'Real' (0)

            explanation = explainer.explain_instance(
                img_0_1, # Image in [0,1]
                lime_predict_proba,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
            
            # Get the image and mask for the predicted class
            temp, mask = explanation.get_image_and_mask(
                target_label_idx, # Explain for the predicted class
                positive_only=False, # Show both positive and negative contributions
                num_features=5,
                hide_rest=False
            )
            
            # The output 'temp' from get_image_and_mask is already in [0,1]
            img_boundry = mark_boundaries(temp, mask)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(img_boundry)
            ax.set_title(f"LIME Explanation for {prediction_output['label']}")
            ax.axis('off')
            plt.tight_layout()

            img_base64 = self.fig_to_base64(fig)
            return "LIME explanation generated", img_base64

        except Exception as e:
            print(f"Error during LIME explanation: {e}")
            return f"Error during LIME explanation: {e}", None


    def textual_explanation(self, prediction_result):
        if prediction_result is None:
            return "Prediction result not available."
        
        raw_pred = prediction_result['raw_prediction']
        
        if prediction_result['label'] == 'Fake':
            return (f"The model predicts this image is **{prediction_result['label']}** with **{prediction_result['confidence']}%** confidence. "
                    f"This suggests strong evidence of manipulation. The raw model output was {raw_pred:.4f} (closer to 1.0 indicating fake).")
        else:
            return (f"The model predicts this image is **{prediction_result['label']}** with **{prediction_result['confidence']}%** confidence. "
                    f"This indicates the image appears authentic. The raw model output was {raw_pred:.4f} (closer to 0.0 indicating real).")

    def uncertainty_quantification(self, img_path):
        # For a standard Keras model without dropout at inference or other stochastic elements,
        # repeatedly predicting on the same image will yield the same result and a std_deviation of 0.
        # This method is more relevant for models with inherent uncertainty (e.g., Bayesian NNs, or Dropout enabled during inference).
        # We will keep a simplified version to show the concept, but note its limitation.
        if self.model is None:
            return "Model not loaded", None
        img_array, _ = self.preprocess_image(img_path)
        
        # If the model has dropout layers, you can enable them for inference to estimate uncertainty
        # by passing `training=True` to model.predict in a custom inference function, but this is
        # generally not recommended without careful consideration.
        
        # For a standard deterministic model, we'll just report the single prediction confidence.
        # To get a more meaningful uncertainty, you'd need a different model architecture or methodology.
        
        prediction_result = self.predict(img_path)
        if prediction_result is None:
            return "Could not get prediction for uncertainty quantification.", None

        # Placeholder for more advanced uncertainty:
        # If you were to implement Monte Carlo Dropout:
        # predictions = []
        # for _ in range(num_samples):
        #     # You'd need a model where dropout layers are active during prediction
        #     # e.g., model.predict(img_array, training=True) or a custom Keras model subclass.
        #     pred = self.model.predict(img_array)[0][0]
        #     predictions.append(pred)
        # mean_pred = np.mean(predictions)
        # std_pred = np.std(predictions)
        # return { 'mean_prediction': mean_pred, 'std_deviation': std_pred }

        # For now, we'll just return the prediction confidence as a measure of certainty/uncertainty.
        certainty = prediction_result['confidence'] / 100.0 # [0, 1]
        uncertainty_score = 1.0 - certainty # Lower confidence means higher uncertainty
        
        return {
            'certainty_score': certainty,
            'uncertainty_score': round(uncertainty_score * 100, 2),
            'explanation': "For a deterministic model, uncertainty is derived from the confidence score (1 - confidence). "
                           "Higher scores indicate less certainty in the prediction."
        }


    def analyze_image_with_xai(self, img_path):
        prediction = self.predict(img_path)
        if prediction is None:
            return {"error": "Model not loaded"}
        
        # Generate explanations as base64 strings
        shap_text, shap_img_base64 = self.explain_with_shap(img_path)
        lime_text, lime_img_base64 = self.explain_with_lime(img_path)
        
        textual_exp = self.textual_explanation(prediction)
        uncertainty = self.uncertainty_quantification(img_path)

        return {
            'prediction': prediction,
            'shap_explanation_text': shap_text,
            'shap_explanation_img_base64': shap_img_base64,
            'lime_explanation_text': lime_text,
            'lime_explanation_img_base64': lime_img_base64,
            'textual_explanation': textual_exp,
            'uncertainty': uncertainty
        }

# Instantiate the enhanced model
enhanced_model = EnhancedModel()