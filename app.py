import os
import csv  # For reading CSV files if needed elsewhere
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from transformers import pipeline
from PIL import Image
import joblib
from keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv
import base64  # For encoding images for the Gemini API
import json  # For JSON parsing
import logging
import tensorflow as tf  # For TFLite inference
# Gemini API (GenAI) imports and configuration
import google.generativeai as genai
load_dotenv()
# Configure Gemini API key from environment variables.
GEMINI_API_KEY = os.getenv("HF_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Google API Key not found. Set it as GEMINI_API_KEY in the environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
# Setup the Gemini generative model
model = genai.GenerativeModel('gemini-1.5-flash')

########################
# Flask App Setup
########################
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

########################
# Load ML/DL Models
########################
# Crop Recommendation Model
crop_model = joblib.load(os.path.join(MODELS_DIR, 'crop_model.pkl'))
crop_scaler = joblib.load(os.path.join(MODELS_DIR, 'crop_scaler.pkl'))
crop_label_encoder = joblib.load(os.path.join(MODELS_DIR, 'crop_label.pkl'))

# Fertilizer Model
fertilizer_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
fertilizer_scaler = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_scaler.pkl'))
fertilizer_label_encoder = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_label.pkl'))

# Plant Disease Model (Keras) for non-Wheat crops
try:
    from tensorflow.keras.models import load_model
    plant_disease_model = load_model(os.path.join(MODELS_DIR, 'Plant_Detection_Model.h5'))
except Exception as e:
    plant_disease_model = None
    print("Plant disease model not loaded:", e)

########################
# Fertilizer Data
########################
fertilizer_data_path = os.path.join(BASE_DIR, 'dataset', 'fertilizer_data.csv')
if os.path.exists(fertilizer_data_path):
    fert_df = pd.read_csv(fertilizer_data_path)
    unique_crop_types = sorted(list(fert_df["Crop Type"].unique()))
else:
    print("Fertilizer CSV NOT found. Using fallback list.")
    unique_crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy']

crop_type_mapping = {crop: i for i, crop in enumerate(unique_crop_types)}
soil_type_mapping = {'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4}

########################
# Plant Disease Data & Crop List
########################
plant_disease_class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple__Cedar_apple_rust', 
    'Apple__healthy', 'Blueberry_healthy', 'Cherry__healthy', 
    'Cherry__Powdery_mildew', 'Corn__Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn__Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn__healthy', 
    'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato___healthy'
]

def unify_label(label):
    """
    Converts a label with multiple underscores into a lowercase, space-joined string.
    E.g., 'Apple__Black_rot' -> 'apple black rot'
    """
    parts = label.split('_')
    parts = [p for p in parts if p.strip()]
    return ' '.join(parts).lower().strip()

# Define the list of crops that your Keras model is trained on (excluding wheat and rice).
trained_crop_list = sorted(set([d.split("_")[0] for d in plant_disease_class_names]))
# The overall crop list shown in your dropdown includes your trained crops plus wheat and rice.
crop_list = sorted(set(trained_crop_list + ["Wheat", "Rice"]))
########################

translation_pipeline = pipeline("translation", model="ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
LANG_TOKEN_MAP = {
    "hi": "<2hi>",  # Hindi
    "te": "<2te>",  # Telugu
    # Add more if needed, e.g., "mr": "<2mr>"
}
########################
# Rice model pipeline using Hugging Face (for rice disease detection)
rice_pipeline = pipeline(
    "image-classification",
    model="jkrperson/Beit-for-rice-disease"
)

plant_disease_info = {
    'Apple__Apple_scab': {
        'description': "Dark, rough spots appear on apple leaves and fruits due to a fungal infection.",
        'recommended_fertilizer': "Use fertilizers rich in potassium along with balanced nutrients.",
        'recommended_treatment': "Spray copper-based fungicides before flowering to prevent the infection.",
        'usage_instructions': "Apply the spray early in the morning or late in the evening."
    },
    'Apple_Black_rot': {
        'description': "Brown, sunken spots develop on apples, eventually causing the fruit to rot.",
        'recommended_fertilizer': "Apply potash-rich fertilizers to boost plant immunity.",
        'recommended_treatment': "Remove infected parts and spray sulfur-based fungicides.",
        'usage_instructions': "Spray before the rainy season starts to protect the crop."
    },
    'Apple__Cedar_apple_rust': {
        'description': "Orange spots form on the leaves, weakening the plant and reducing fruit quality.",
        'recommended_fertilizer': "Use balanced fertilizers with nitrogen, phosphorus, and potassium.",
        'recommended_treatment': "Apply fungicides containing myclobutanil early in the season.",
        'usage_instructions': "Spray in early spring when new leaves begin to appear."
    },
    'Apple__healthy': {
        'description': "The apple plant is healthy, with no signs of disease.",
        'recommended_fertilizer': "Maintain balanced nutrition using compost and manure.",
        'recommended_treatment': "No treatment is needed; continue regular care.",
        'usage_instructions': "Keep up with regular watering and pruning for best results."
    },
    'Blueberry_healthy': {
        'description': "The blueberry plant is disease-free and growing well.",
        'recommended_fertilizer': "Use nitrogen-based fertilizers to support good fruit production.",
        'recommended_treatment': "No treatment needed; simply maintain proper care.",
        'usage_instructions': "Ensure proper watering and timely pruning."
    },
    'Cherry__healthy': {
        'description': "The cherry plant is healthy and producing good quality fruits.",
        'recommended_fertilizer': "Use fertilizers rich in potassium to support fruit growth.",
        'recommended_treatment': "No treatment required; follow standard care practices.",
        'usage_instructions': "Maintain regular watering and avoid over-fertilization."
    },
    'Cherry__Powdery_mildew': {
        'description': "A white, powdery fungus appears on the leaves, making them weak.",
        'recommended_fertilizer': "Apply phosphorus-rich fertilizers to enhance plant resistance.",
        'recommended_treatment': "Spray neem oil or sulfur-based fungicides on the affected areas.",
        'usage_instructions': "Use early morning or evening sprays for effective control."
    },
    'Corn__Cercospora_leaf_spot Gray_leaf_spot': {
        'description': "Grey or brown spots form on corn leaves, reducing photosynthesis and yield.",
        'recommended_fertilizer': "Use nitrogen-rich fertilizers to promote robust growth.",
        'recommended_treatment': "Apply fungicides containing azoxystrobin or tebuconazole at early signs.",
        'usage_instructions': "Spray during the early stage of infection for best control."
    },
    'Corn__Common_rust': {
        'description': "Reddish-brown spots develop on corn leaves, which may lower overall production.",
        'recommended_fertilizer': "Apply balanced fertilizers with proper nitrogen levels.",
        'recommended_treatment': "Use mancozeb-based fungicides at the first sign of rust.",
        'usage_instructions': "Apply the fungicide before the rainy season to prevent spread."
    },
    'Corn_Northern_Leaf_Blight': {
        'description': "Long, brown lesions appear on corn leaves, affecting the plant's vitality.",
        'recommended_fertilizer': "Enhance soil fertility with organic compost.",
        'recommended_treatment': "Spray fungicides containing tebuconazole or azoxystrobin.",
        'usage_instructions': "Apply treatment early in the growing season."
    },
    'Corn__healthy': {
        'description': "The corn plant is healthy and free from diseases.",
        'recommended_fertilizer': "Use nitrogen-rich fertilizers for a good yield.",
        'recommended_treatment': "No treatment is needed; maintain good cultural practices.",
        'usage_instructions': "Regular irrigation and weed control are essential."
    },
    'Grape__Black_rot': {
        'description': "Black spots form on grape leaves and fruits, eventually causing rot.",
        'recommended_fertilizer': "Apply potash-based fertilizers to boost plant defenses.",
        'recommended_treatment': "Remove affected leaves and spray copper-based fungicides.",
        'usage_instructions': "Apply before the monsoon to reduce the risk of infection."
    },
    'Grape_Esca(Black_Measles)': {
        'description': "Brown streaks appear on leaves and fruits, leading to reduced yield.",
        'recommended_fertilizer': "Use organic manure along with phosphorus-rich fertilizers.",
        'recommended_treatment': "Prune and remove infected vines, then spray copper fungicide.",
        'usage_instructions': "Spray during dry weather for optimal results."
    },
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': {
        'description': "Brown patches develop on grape leaves, causing them to drop early.",
        'recommended_fertilizer': "Use nitrogen-rich fertilizers to maintain healthy foliage.",
        'recommended_treatment': "Apply Bordeaux mixture or copper-based fungicides.",
        'usage_instructions': "Spray at the first sign of infection."
    },
    'Grape__healthy': {
        'description': "The grape plant is healthy and shows no signs of disease.",
        'recommended_fertilizer': "Maintain balanced nutrition with organic fertilizers.",
        'recommended_treatment': "No treatment required.",
        'usage_instructions': "Regular pruning and pest management will help keep the plant healthy."
    },
    'Orange_Haunglongbing(Citrus_greening)': {
        'description': "Infected orange plants show yellowing leaves and produce small, misshapen fruits.",
        'recommended_fertilizer': "Use micronutrient fertilizers containing zinc and iron.",
        'recommended_treatment': "Remove infected branches and apply organic pesticides.",
        'usage_instructions': "Regularly monitor and control insect vectors like psyllids."
    },
    'Peach___Bacterial_spot': {
        'description': "Small brown spots appear on peach leaves and fruits due to bacterial infection.",
        'recommended_fertilizer': "Use nitrogen-rich fertilizers to support foliage growth.",
        'recommended_treatment': "Spray copper-based bactericides to control the spread.",
        'usage_instructions': "Apply during the early fruiting stage."
    },
    'Peach__healthy': {
        'description': "The peach tree is healthy and bearing good quality fruit.",
        'recommended_fertilizer': "Use phosphorus-rich fertilizers to ensure fruit quality.",
        'recommended_treatment': "No treatment is necessary; follow standard care practices.",
        'usage_instructions': "Ensure proper irrigation and drainage."
    },
    'Pepper,_bell_Bacterial_spot': {
        'description': "Small dark spots develop on bell pepper leaves because of bacterial infection.",
        'recommended_fertilizer': "Use calcium-rich fertilizers to strengthen plant tissues.",
        'recommended_treatment': "Spray copper-based bactericides to control the infection.",
        'usage_instructions': "Apply early in the day for maximum effect."
    },
    'Pepper,_bell__healthy': {
        'description': "The bell pepper plant is healthy and free from bacterial infections.",
        'recommended_fertilizer': "Use balanced fertilizers for optimal growth.",
        'recommended_treatment': "No treatment is needed, just routine care.",
        'usage_instructions': "Maintain regular watering and nutrient supply."
    },
    'Potato__Early_blight': {
        'description': "Brown spots appear on potato leaves, reducing the size and quality of tubers.",
        'recommended_fertilizer': "Use compost and phosphorus-rich fertilizers.",
        'recommended_treatment': "Spray copper-based fungicides at the first signs of infection.",
        'usage_instructions': "Apply after rainfall when the leaves are wet."
    },
    'Potato_Late_blight': {
        'description': "A fast-spreading disease that darkens potato leaves and causes tuber rot.",
        'recommended_fertilizer': "Use balanced fertilizers with sufficient nitrogen and potassium.",
        'recommended_treatment': "Spray fungicides like mancozeb or chlorothalonil immediately.",
        'usage_instructions': "Apply at the first appearance of symptoms and reapply as needed."
    },
    'Potato__healthy': {
        'description': "The potato plant is healthy and developing well.",
        'recommended_fertilizer': "Maintain balanced fertilization with NPK-based fertilizers.",
        'recommended_treatment': "No treatment is required; practice regular field hygiene.",
        'usage_instructions': "Regular irrigation and timely weeding are recommended."
    },
    'Raspberry__healthy': {
        'description': "The raspberry plant is healthy and free from disease.",
        'recommended_fertilizer': "Use organic compost and balanced fertilizers to boost growth.",
        'recommended_treatment': "No treatment is needed; keep monitoring for pests.",
        'usage_instructions': "Maintain proper pruning and regular watering."
    },
    'Soybean_healthy': {
        'description': "The soybean plant is healthy and growing well.",
        'recommended_fertilizer': "Use nitrogen-fixing inoculants or balanced fertilizers.",
        'recommended_treatment': "No treatment is required.",
        'usage_instructions': "Ensure proper irrigation and soil fertility management."
    },
    'Squash__Powdery_mildew': {
        'description': "A white, powdery fungus appears on squash leaves, stunting growth.",
        'recommended_fertilizer': "Apply balanced fertilizers to keep the plant strong.",
        'recommended_treatment': "Spray neem oil or sulfur-based fungicides to control the fungus.",
        'usage_instructions': "Apply the treatment early in the morning."
    },
    'Strawberry__Leaf_scorch': {
        'description': "The edges of strawberry leaves appear brown or scorched, possibly from sunburn or infection.",
        'recommended_fertilizer': "Use organic compost and micronutrient-rich fertilizers.",
        'recommended_treatment': "Increase shading and ensure proper watering; apply fungicides if necessary.",
        'usage_instructions': "Avoid intense midday sun and water during cooler hours."
    },
    'Strawberry_healthy': {
        'description': "The strawberry plant is healthy and producing good fruit.",
        'recommended_fertilizer': "Use balanced organic fertilizers along with compost.",
        'recommended_treatment': "No treatment required; maintain good cultural practices.",
        'usage_instructions': "Regular irrigation and mulching will help sustain growth."
    },
    'Tomato__Bacterial_spot': {
        'description': "Small black spots appear on tomato leaves and fruits due to bacteria.",
        'recommended_fertilizer': "Use calcium-rich fertilizers to strengthen plant tissues.",
        'recommended_treatment': "Spray copper-based bactericides to control the disease.",
        'usage_instructions': "Apply the spray early in the morning."
    },
    'Tomato__Early_blight': {
        'description': "Dark patches form on lower tomato leaves, leading to early leaf drop and weaker fruits.",
        'recommended_fertilizer': "Use compost and nitrogen-based fertilizers to support robust growth.",
        'recommended_treatment': "Spray fungicides containing chlorothalonil to manage the disease.",
        'usage_instructions': "Begin treatment as soon as symptoms appear."
    },
    'Tomato_Late_blight': {
        'description': "A serious fungal infection that causes tomato leaves to turn black and fruits to rot.",
        'recommended_fertilizer': "Use phosphorus-rich fertilizers to enhance plant resistance.",
        'recommended_treatment': "Apply mancozeb-based fungicides immediately at the first sign.",
        'usage_instructions': "Reapply treatment after heavy rains."
    },
    'Tomato__Leaf_Mold': {
        'description': "A moldy, powdery growth forms on the underside of tomato leaves due to high humidity.",
        'recommended_fertilizer': "Use calcium-rich fertilizers to prevent related issues like blossom-end rot.",
        'recommended_treatment': "Spray sulfur-based fungicides to control the mold.",
        'usage_instructions': "Apply treatment early in the day when humidity is lower."
    },
    'Tomato__Septoria_leaf_spot': {
        'description': "Small dark spots appear on tomato leaves, often leading to premature leaf drop.",
        'recommended_fertilizer': "Maintain balanced fertilization to keep plants healthy.",
        'recommended_treatment': "Spray fungicides containing chlorothalonil at the first sign of spots.",
        'usage_instructions': "Begin treatment as soon as symptoms are noticed."
    },
    'Tomato__Spider_mites Two-spotted_spider_mite': {
        'description': "Tiny spider mites cause speckled yellow spots and fine webbing on tomato leaves.",
        'recommended_fertilizer': "Apply balanced fertilizers along with micronutrients to support strong growth.",
        'recommended_treatment': "Use insecticidal soap or neem oil to control spider mites.",
        'usage_instructions': "Spray every few days until the mite population is under control."
    },
    'Tomato__Target_Spot': {
        'description': "Circular brown spots with yellow edges appear on tomato leaves, reducing photosynthesis.",
        'recommended_fertilizer': "Use phosphorus-rich fertilizers to promote strong plant structure.",
        'recommended_treatment': "Apply fungicides targeted at controlling target spot.",
        'usage_instructions': "Start treatment as soon as small spots appear."
    },
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': {
        'description': "Tomato leaves begin to curl and turn yellow due to a viral infection, weakening the plant.",
        'recommended_fertilizer': "Maintain balanced levels of nitrogen and phosphorus.",
        'recommended_treatment': "Control whiteflies (the virus vector) and remove infected plants.",
        'usage_instructions': "Monitor plants regularly and remove affected leaves promptly."
    },
    'Tomato__Tomato_mosaic_virus': {
        'description': "Mottled or mosaic patterns appear on tomato leaves as a result of viral infection.",
        'recommended_fertilizer': "Use balanced fertilizers to support overall plant health.",
        'recommended_treatment': "Remove and destroy infected plants and control insect vectors.",
        'usage_instructions': "Practice crop rotation and maintain field hygiene."
    },
    'Tomato___healthy': {
        'description': "The tomato plant is healthy, showing no signs of disease.",
        'recommended_fertilizer': "Use balanced organic fertilizers for optimal growth and fruiting.",
        'recommended_treatment': "No treatment needed; just ensure regular care.",
        'usage_instructions': "Maintain regular watering, weeding, and monitoring for pests."
    }
}




########################
# Weather API
########################
def get_weather(city="Hyderabad", country_code="IN"):
    API_KEY = os.getenv("API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        weather_data = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rainfall": data.get("rain", {}).get("1h", 0)
        }
        return weather_data
    except Exception as e:
        print("Error fetching weather data:", e)
        return None

########################
# Crop & Fertilizer Prediction Functions
########################
def predict_crop(input_data):
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df = pd.DataFrame([input_data], columns=columns)
    scaled = crop_scaler.transform(df)
    pred = crop_model.predict(scaled)
    crop_name = crop_label_encoder.inverse_transform(pred)[0]
    return crop_name

def predict_crop_with_weather(user_data, city="Hyderabad", country_code="IN"):
    weather = get_weather(city, country_code)
    if weather:
        user_data[3] = weather["temperature"]
        user_data[4] = weather["humidity"]
        if user_data[6] == 0:
            user_data[6] = weather["rainfall"]
    return predict_crop(user_data)

def predict_fertilizer_numeric(input_data):
    df = pd.DataFrame([input_data])
    num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    df[num_cols] = fertilizer_scaler.transform(df[num_cols])
    pred = fertilizer_model.predict(df)
    if hasattr(fertilizer_label_encoder, 'inverse_transform'):
        fertilizer_name = fertilizer_label_encoder.inverse_transform(pred)[0]
    else:
        fertilizer_mapping = {
            0: 'Urea',
            1: 'DAP',
            2: '14-35-14',
            3: '28-28',
            4: '17-17-17',
            5: '20-20',
            6: '10-26-26'
        }
        fertilizer_name = fertilizer_mapping.get(pred[0], "Unknown Fertilizer")
    return fertilizer_name

########################
# Wheat Disease Detection Using TFLite
########################
def preprocess_img(img_path, target_size=(255, 255)):
    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255
    x = np.expand_dims(x, axis=0)
    return x

def classify_wheat_image(img_path):
    """
    Classifies the disease in a wheat image using the TFLite model.
    Returns a tuple: (predicted_label, confidence)
    """
    x = preprocess_img(img_path, target_size=(255, 255))
    tflite_model_path = os.path.join(MODELS_DIR, 'wheat_model.tflite')
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    labels = {
        0: 'Aphid', 1: 'Black Rust', 2: 'Blast', 3: 'Brown Rust', 4: 'Common Root Rot',
        5: 'Fusarium Head Blight', 6: 'Healthy', 7: 'Leaf Blight', 8: 'Mildew', 9: 'Mite',
        10: 'Septoria', 11: 'Smut', 12: 'Stem fly', 13: 'Tan spot', 14: 'Yellow Rust'
    }
    probabilities = output_data[0]
    predicted_class = np.argmax(probabilities)
    predicted_label = labels.get(predicted_class, "Unknown")
    confidence = float(probabilities[predicted_class])
    return predicted_label, confidence

def translate_text(text, target_lang):
    """
    Uses the Gemini API to translate English text into the target language.
    If target_lang is 'en', returns the original text.
    """
    if target_lang == "en":
        return text
    
    # A more explicit prompt asking for a minimal translation only
    prompt = f"""
You are a helpful translation assistant. Please translate the following English text into {target_lang}. 
Do not provide any commentary, explanations, or alternative translations. 
Simply return the translated text in {target_lang}.

English text:
{text}
"""
    response = model.generate_content([prompt])
    
    # The model’s response can still contain extra text, 
    # so you may want to parse or trim it further if needed.
    # For now, we just strip whitespace.
    return response.text.strip()

def translate_disease_details(details, target_lang):
    """
    Translates each field in a disease details dictionary using the Gemini API.
    """
    translated = {}
    for key, value in details.items():
        print(f"DEBUG: Translating key='{key}' -> '{value}'")
        translated_text = translate_text(value, target_lang)
        print(f"DEBUG: Translated result -> '{translated_text}'")
        translated[key] = translated_text
    return translated

########################
# Helper function to get disease details (normalize underscores)
########################
def get_disease_info(predicted_label):
    """
    Normalizes the predicted disease label using unify_label() and matches it
    with the keys in plant_disease_info (also normalized).
    """
    normalized_pred = unify_label(predicted_label)
    print(f"DEBUG: Predicted disease raw='{predicted_label}' -> normalized='{normalized_pred}'")
    
    for key, details in plant_disease_info.items():
        normalized_key = unify_label(key)
        if normalized_key == normalized_pred:
            print(f"DEBUG: Matched dictionary key='{key}' (normalized='{normalized_key}')")
            return details
    print("DEBUG: No matching key found in dictionary, returning fallback info.")
    return {
        'description': "No information available.",
        'recommended_fertilizer': "No recommendation available.",
        'recommended_treatment': "No treatment available.",
        'usage_instructions': "No instructions available."
    }

def generate_short_disease_details(disease_name, crop_name, target_lang="en"):
    """
    Uses the Gemini API to generate short, minimal disease details if they are not
    available in the local dictionary. The prompt requests a concise bullet-point format.
    """
    prompt = f"""
You are a helpful assistant. Provide a short, minimal set of details about the plant disease called '{disease_name}' affecting the crop '{crop_name}'.
Write the response in {target_lang} language.
Include the following sections in bullet points:
1. Description
2. Recommended Fertilizer
3. Recommended Treatment
4. Usage Instructions

Keep it concise and relevant. Do not provide extra commentary.
"""
    try:
        response = model.generate_content([prompt])
        raw_text = response.text.strip()
        
        # Initialize a fallback dictionary structure
        details = {
            'description': "",
            'recommended_fertilizer': "",
            'recommended_treatment': "",
            'usage_instructions': ""
        }
        
        # Naively split the response into lines and assign based on keywords
        lines = raw_text.split('\n')
        current_key = None
        for line in lines:
            line_lower = line.lower()
            if "description" in line_lower:
                current_key = 'description'
                continue
            elif "recommended fertilizer" in line_lower:
                current_key = 'recommended_fertilizer'
                continue
            elif "recommended treatment" in line_lower:
                current_key = 'recommended_treatment'
                continue
            elif "usage instructions" in line_lower:
                current_key = 'usage_instructions'
                continue
            
            if current_key:
                details[current_key] += line.strip() + " "
        
        # Clean up and ensure each field has some content
        for key in details:
            details[key] = details[key].strip()
            if not details[key]:
                details[key] = "Information not available."
        
        return details
    except Exception as e:
        print(f"Error generating details via Gemini: {e}")
        return {
            'description': "No information available.",
            'recommended_fertilizer': "No recommendation available.",
            'recommended_treatment': "No treatment available.",
            'usage_instructions': "No instructions available."
        }
########################
# Keras Model for Non-Wheat Diseases 
########################
# Keras Model for Non-Wheat Diseases
########################
def predict_plant_disease(image_path):
    """
    Uses the Keras model (Plant_Detection_Model.h5) to predict disease for non-wheat crops.
    Returns the predicted disease label.
    """
    if not plant_disease_model:
        return "No plant disease model loaded."
    try:
        image_obj = Image.open(image_path).convert('RGB')
        image_obj = image_obj.resize((224, 224))
        img_array = np.array(image_obj).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = plant_disease_model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        disease_name = plant_disease_class_names[class_idx]
        return disease_name
    except Exception as e:
        return f"Error in disease prediction: {str(e)}"

########################
# Gemini API (GenAI) Integration
########################
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_relevant_feedback(plant_name):
    # Placeholder: In production, retrieve feedback data from a database or other source
    return ""

def analyze_plant_image(image_path, plant_name, language):
    try:
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": encode_image(image_path)
            }
        ]
        feedback_context = get_relevant_feedback(plant_name)
        feedback_instruction = (f"Please consider the following user feedback from similar cases: {feedback_context}"
                                  if feedback_context else "")
        prompt = f"""
Analyze this image of a {plant_name} plant and prioritize determining if it's healthy or has a disease or pest infestation.
If a disease or pest is detected, remember the plant can be healthy too. Provide the following information in JSON format:
{{
    "results": [
        {{
            "type": "disease/pest",
            "name": "Name of disease or pest",
            "probability": "Probability as a percentage",
            "symptoms": "Describe the visible symptoms",
            "causes": "Main causes of the disease or pest"
        }}
    ]
}}
{feedback_instruction}
"""
        response = model.generate_content([prompt] + image_parts)
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > 0:
            json_str = response_text[json_start:json_end]
            result_data = json.loads(json_str)
            return result_data.get("results", [{}])[0].get("name", "Unknown")
        else:
            return {"error": "Failed to parse the API response", "raw_response": response_text}
    except Exception as e:
        return {"error": str(e)}

def predict_plant_disease_with_fallback(image_path, plant_name):
    """
    Uses the Keras model to predict disease for non-wheat crops.
    If the confidence is below a threshold or the disease isn't recognized,
    falls back to the Gemini API.
    Returns a tuple: (disease_name, confidence)
    """
    if not plant_disease_model:
        return ("No plant disease model loaded.", None)
    try:
        image_obj = Image.open(image_path).convert('RGB')
        image_obj = image_obj.resize((224, 224))
        img_array = np.array(image_obj).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = plant_disease_model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][class_idx])
        disease_name = plant_disease_class_names[class_idx]
        CONFIDENCE_THRESHOLD = 0.8
        if confidence < CONFIDENCE_THRESHOLD or disease_name not in plant_disease_info:
            gemini_result = analyze_plant_image(image_path, plant_name, language="en")
            if isinstance(gemini_result, dict) and "error" in gemini_result:
                return (f"Gemini API error: {gemini_result.get('error', 'Unknown error')}", confidence)
            else:
                return (gemini_result, confidence)
        else:
            return (disease_name, confidence)
    except Exception as e:
        return (f"Error in disease prediction: {str(e)}", None)

def predict_plant_disease_only_model(image_path):
    """
    Uses only the Keras model to predict disease for non-wheat crops.
    Returns a tuple: (disease_name, confidence)
    """
    if not plant_disease_model:
        return ("No plant disease model loaded.", None)
    try:
        image_obj = Image.open(image_path).convert('RGB')
        image_obj = image_obj.resize((224, 224))
        img_array = np.array(image_obj).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = plant_disease_model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][class_idx])
        disease_name = plant_disease_class_names[class_idx]
        return (disease_name, confidence)
    except Exception as e:
        return (f"Error in disease prediction: {str(e)}", None)

########################
# Routes
########################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop-prediction', methods=['GET', 'POST'])
def crop_prediction():
    prediction = None
    weather_data = None
    if request.method == 'POST':
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            user_data = [N, P, K, temperature, humidity, ph, rainfall]
            use_weather = request.form.get('use_weather')
            if use_weather == "yes":
                weather_data = get_weather(city="Hyderabad", country_code="IN")
                if weather_data:
                    user_data[3] = weather_data["temperature"]
                    user_data[4] = weather_data["humidity"]
                    if user_data[6] == 0:
                        user_data[6] = weather_data["rainfall"]
            prediction = predict_crop(user_data)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('crop_prediction'))
    return render_template('crop_prediction.html', prediction=prediction, weather=weather_data)

@app.route('/fertilizer-prediction', methods=['GET', 'POST'])
def fertilizer_prediction():
    prediction = None
    weather_data = None
    if request.method == 'POST':
        try:
            soil_str = request.form['Soil Type']
            crop_str = request.form['Crop Type']
            temp = float(request.form['Temparature'])
            hum = float(request.form['Humidity'])
            moist = float(request.form['Moisture'])
            nitrogen = float(request.form['Nitrogen'])
            potassium = float(request.form['Potassium'])
            phosphorous = float(request.form['Phosphorous'])
            use_weather = request.form.get('use_weather')
            if use_weather == "yes":
                weather_data = get_weather(city="Hyderabad", country_code="IN")
                if weather_data:
                    temp = weather_data["temperature"]
                    hum = weather_data["humidity"]
            soil_numeric = soil_type_mapping.get(soil_str)
            crop_numeric = crop_type_mapping.get(crop_str)
            if soil_numeric is None or crop_numeric is None:
                flash("Invalid selection for Soil Type or Crop Type.")
                return redirect(url_for('fertilizer-prediction'))
            input_data = {
                'Temparature': temp,
                'Humidity': hum,
                'Moisture': moist,
                'Soil Type': soil_numeric,
                'Crop Type': crop_numeric,
                'Nitrogen': nitrogen,
                'Potassium': potassium,
                'Phosphorous': phosphorous
            }
            prediction = predict_fertilizer_numeric(input_data)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('fertilizer-prediction'))
    return render_template('fertilizer_prediction.html',
                           prediction=prediction,
                           weather=weather_data,
                           crop_types=unique_crop_types,
                           soil_types=list(soil_type_mapping.keys()))

@app.route('/plant-disease-detection', methods=['GET', 'POST'])
def plant_disease_detection():
    selected_crop = None
    disease_name = None
    disease_details = None
    image_url = None
    confidence = None
    target_lang = request.form.get('target_lang', 'en')

    if request.method == 'POST':
        selected_crop = request.form.get('crop')
        if not selected_crop:
            flash("Please select a crop type.")
        else:
            file = request.files.get('image')
            if not file or file.filename == '':
                flash("No selected file.")
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                # Determine prediction based on selected crop:
                # - For wheat and rice, use their dedicated models.
                # - For crops in your trained model (trained_crop_list), use your h5 model.
                # - Otherwise, use the Gemini API fallback.
                if selected_crop.lower() == "wheat":
                    disease_name, confidence = classify_wheat_image(file_path)
                elif selected_crop.lower() == "rice":
                    disease_name, confidence = predict_rice_disease(file_path)
                elif selected_crop in trained_crop_list:
                    disease_name, confidence = predict_plant_disease_only_model(file_path)
                else:
                    disease_name = analyze_plant_image(file_path, selected_crop, language="en")
                    confidence = "N/A"
                
                disease_details = get_disease_info(disease_name)
                no_info_found = (
                    disease_details.get('description') == "No information available." and
                    disease_details.get('recommended_fertilizer') == "No recommendation available." and
                    disease_details.get('recommended_treatment') == "No treatment available." and
                    disease_details.get('usage_instructions') == "No instructions available."
                )

                if no_info_found:
                    # Provide both the disease name and the selected crop
                    disease_details = generate_short_disease_details(
                        disease_name=disease_name, 
                        crop_name=selected_crop, 
                        target_lang=target_lang
                    )

                # If user selected a non-English language, translate
                if target_lang != "en":
                    disease_details = translate_disease_details(disease_details, target_lang)

                image_url = url_for('static', filename=f'uploads/{filename}')

                # Optional: check if predicted disease starts with the selected crop
                if isinstance(disease_name, str) and not disease_name.lower().startswith(selected_crop.lower()):
                    flash("Warning: The detected disease does not match the selected crop.")

    return render_template('plant_disease.html',
                           crops=crop_list,
                           selected_crop=selected_crop,
                           disease_name=disease_name,
                           disease_details=disease_details,
                           image_url=image_url,
                           confidence=confidence)

def predict_rice_disease(file_path):
    """
    Uses the Hugging Face pipeline for rice disease detection.
    Returns a tuple: (disease_label, confidence)
    """
    image = Image.open(file_path)
    results = rice_pipeline(image)
    if not results:
        return ("Unknown", 0.0)
    top_result = results[0]
    disease_label = top_result.get("label", "Unknown")
    confidence = top_result.get("score", 0.0)
    return (disease_label, confidence)

@app.route('/test-weather')
def test_weather():
    data = get_weather(city="Hyderabad", country_code="IN")
    if data:
        return f"""
        <h2>Weather Test</h2>
        <p>Temperature: {data['temperature']} °C</p>
        <p>Humidity: {data['humidity']} %</p>
        <p>Rainfall: {data['rainfall']} mm</p>
        """
    else:
        return "<h2>Weather Test</h2><p>Failed to fetch weather data.</p>"

if __name__ == '__main__':
    app.run(debug=True)