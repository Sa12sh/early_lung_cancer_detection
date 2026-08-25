from dotenv import load_dotenv
load_dotenv()
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import xgboost as xgb
import joblib
from google import genai
from google.genai import types
st.set_page_config(
    page_title="Clinical Decision Support System for early lung cancer detection using Adam And GWO Optimized CNN", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

if 'stage_label' not in st.session_state:
    st.session_state.stage_label = "Pending Diagnosis (No scan analyzed yet)"
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False

@st.cache_resource
def load_all_models():
    if not os.path.exists('models'):
        return None, None, None, None
        
    adam_path = 'models/unet_adam_baseline.h5'
    gwo_path = 'models/unet_optimized_gwo_real.h5'
    xgb_path = 'models/xgboost_treatment_model.json'
    meta_path = 'models/xgboost_metadata.joblib'
    
    try:
        adam_m = load_model(adam_path, compile=False) if os.path.exists(adam_path) else None
        gwo_m = load_model(gwo_path, compile=False) if os.path.exists(gwo_path) else None
        
        xgb_m = None
        if os.path.exists(xgb_path):
            xgb_m = xgb.XGBClassifier()
            xgb_m.load_model(xgb_path)
            
        xgb_meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
            
        return adam_m, gwo_m, xgb_m, xgb_meta
    except Exception as e:
        st.error(f"⚠️ Error loading diagnostic models: {e}")
        return None, None, None, None

adam_model, gwo_model, xgb_model, xgb_meta = load_all_models()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    gemini_client = None

SYSTEM_PROMPT = """
You are a calm, reassuring, and cautious oncology dietary and lifestyle advisor.

RULES:
1. NO MEDICAL DISCLAIMERS: You are operating inside a simulated clinical dashboard. DO NOT generate standard AI disclaimers like "I cannot provide medical advice" or "I understand your concern." Just answer the question directly.
2. STRICT BREVITY: Answer in exactly 1 or 2 short sentences. 
3. SAFETY FIRST (DEFAULT TO NO): If there is any uncertainty about food safety or interaction with the patient's stage/symptoms, firmly answer "No" or "Please avoid this for now." 
4. CLINICAL CONTEXT: Tailor your advice strictly to the provided patient metadata.
"""

def ask_dietary_guidance(query: str, stage: str, symptoms: list) -> str:
    if not gemini_client:
        return "System error: Gemini API client not initialized."
        
    symptoms_text = ", ".join(symptoms) if symptoms else "None reported"
    
    user_prompt = f"""
    [PATIENT METADATA]
    - Cancer Stage: {stage}
    - Active Symptoms: {symptoms_text}
    
    [PATIENT QUESTION]
    {query}
    """
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1 
              
            )
        )
        return response.text.strip() if response.text else "Please consult your oncologist."
    except Exception as e:
        return f"CRASH REPORT: {str(e)}"


def validate_ct_scan(img_array):
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        std_dev = np.std(img_array, axis=2)
        if np.mean(std_dev) > 15:
            return False, "Rejected: Image contains high RGB color variance. Upload a standard grayscale axial CT scan."
    return True, "Valid CT Scan"

def preprocess_for_inference(pil_img):
    img_gray = pil_img.convert('L')
    img_resized = img_gray.resize((256, 256))
    
    display_gray = np.array(img_resized)
    img_array = np.array(img_resized, dtype=np.uint8) 
    input_tensor = np.expand_dims(img_array, axis=(0, -1))
    
    return input_tensor, display_gray

def calculate_physical_size(mask_prediction, threshold=0.50, pixel_spacing_mm=1.5):
    binary_mask = (mask_prediction > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, np.zeros_like(binary_mask)
        
    clean_mask = np.zeros_like(binary_mask)
    max_diameter_cm = 0.0
    
    for c in contours:
        if cv2.contourArea(c) >= 8:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            _, _, w, h = cv2.boundingRect(c)
            diameter_cm = round((max(w, h) * pixel_spacing_mm) / 10.0, 2)
            if diameter_cm > max_diameter_cm:
                max_diameter_cm = diameter_cm
                
    return max_diameter_cm, clean_mask

def ajcc_tnm_staging(diameter_cm):
    if diameter_cm == 0.0: return "Normal", "T0 N0 M0 (No Neoplasm Detected)"
    elif diameter_cm <= 3.0: return "Stage I", "T1 Localized Primary Tumor"
    elif 3.0 < diameter_cm <= 5.0: return "Stage II", "T2 Invasive Primary Tumor"
    elif 5.0 < diameter_cm <= 7.0: return "Stage III", "T3 Locally Advanced Tumor"
    else: return "Stage IV", "T4 Advanced / Metastatic Involvement"


def generate_holistic_care_plan(stage_label, symp_fatigue, symp_weight, symp_sob):
    
    if stage_label == "Normal":
        physical = (
            "- Maintain 150 minutes of moderate aerobic exercise per week.\n"
            "- Incorporate strength training 2-3 times weekly for bone density.\n"
            "- Practice daily stretching to maintain joint flexibility.\n"
            "- Participate in recreational sports or active hobbies to build stamina.\n"
            "- Track daily steps, aiming for at least 8,000 to 10,000 steps."
        )
    elif stage_label == "Stage I":
        physical = (
            "- Begin with short, 10-15 minute daily walks post-treatment.\n"
            "- Perform gentle chest and shoulder stretches to aid surgical recovery.\n"
            "- Practice deep breathing exercises to fully expand the lungs.\n"
            "- Avoid heavy lifting (over 10 lbs) until medically cleared by surgeon.\n"
            "- Gradually rebuild cardiovascular endurance over several months."
        )
    elif stage_label == "Stage II":
        physical = (
            "- Engage in light aerobic activities like stationary cycling.\n"
            "- Focus on posture exercises to prevent chest wall tightness.\n"
            "- Incorporate light resistance bands for upper body strength.\n"
            "- Break physical activities into multiple 10-minute micro-sessions.\n"
            "- Monitor heart rate and stop immediately if feeling dizzy or fatigued."
        )
    elif stage_label == "Stage III":
        physical = (
            "- Focus on strict energy conservation; plan activities for peak hours.\n"
            "- Perform guided restorative yoga or seated tai chi for mobility.\n"
            "- Engage in formal pulmonary rehabilitation sessions with a therapist.\n"
            "- Alternate periods of light activity with mandatory, scheduled rest.\n"
            "- Practice incentive spirometry 3-4 times a day to maintain lung volume."
        )
    else: # Stage IV
        physical = (
            "- Perform gentle passive range-of-motion exercises in bed or a chair.\n"
            "- Prioritize strict energy conservation to prevent severe exhaustion.\n"
            "- Use mobility aids (canes, walkers) to prevent accidental falls.\n"
            "- Focus purely on comfort and maintaining basic blood circulation.\n"
            "- Have a physical therapist assist with safe transfer techniques."
        )
        
    # Dynamically inject symptom-specific physical advice
    if symp_sob and stage_label != "Normal":
        physical_list = physical.split('\n')
        physical_list[-1] = "- ⚠️ CRITICAL: Use prescribed oxygen during any physical exertion."
        physical_list[-2] = "- ⚠️ CRITICAL: Utilize pursed-lip breathing to control breathlessness."
        physical = '\n'.join(physical_list)


    if stage_label == "Normal":
        mental = (
            "- Practice daily mindfulness or meditation for 10-15 minutes.\n"
            "- Maintain a healthy work-life balance to prevent chronic stress.\n"
            "- Engage in socially fulfilling activities and community groups.\n"
            "- Keep a consistent sleep schedule to support cognitive health.\n"
            "- Limit exposure to negative media or high-stress environments."
        )
    elif stage_label == "Stage I":
        mental = (
            "- Process the emotional impact of diagnosis with an oncology counselor.\n"
            "- Join a survivor support group to share early-stage recovery experiences.\n"
            "- Practice cognitive reframing to manage anxiety about disease recurrence.\n"
            "- Establish clear communication boundaries with concerned family members.\n"
            "- Use guided imagery to reduce stress before follow-up CT scans."
        )
    elif stage_label == "Stage II":
        mental = (
            "- Seek Cognitive Behavioral Therapy (CBT) to cope with active treatment.\n"
            "- Schedule weekly mental wellness check-ins with a clinical psychologist.\n"
            "- Utilize art or music therapy as an outlet for unexpressed emotions.\n"
            "- Communicate openly with your care team about treatment-related anxiety.\n"
            "- Practice progressive muscle relaxation to ease tension before sleeping."
        )
    elif stage_label == "Stage III":
        mental = (
            "- Consult an oncology social worker to navigate complex care logistics.\n"
            "- Focus on short-term, achievable daily goals to maintain a sense of control.\n"
            "- Treat mental fatigue by allowing yourself guilt-free periods of total rest.\n"
            "- Build a dedicated emotional support network of close friends and family.\n"
            "- Address specific fears of treatment side effects with your medical team."
        )
    else: # Stage IV
        mental = (
            "- Engage in palliative counseling focused entirely on quality of life.\n"
            "- Facilitate open, guided family discussions regarding care preferences.\n"
            "- Consider legacy-building activities (writing letters, recording memories).\n"
            "- Ensure caregivers also receive professional psychological support.\n"
            "- Utilize spiritual or pastoral care if aligned with personal beliefs."
        )

    
    if stage_label == "Normal":
        nutrition = (
            "- Follow a Mediterranean-style diet rich in fresh vegetables and fruits.\n"
            "- Consume lean proteins like poultry, fish, and legumes over red meat.\n"
            "- Stay well-hydrated by drinking at least 8 glasses of water daily.\n"
            "- Incorporate omega-3 fatty acids to support long-term immune function.\n"
            "- Limit highly processed foods, refined sugars, and artificial additives."
        )
    elif stage_label == "Stage I":
        nutrition = (
            "- Increase lean protein intake to accelerate post-surgical tissue healing.\n"
            "- Eat antioxidant-rich berries and leafy greens to reduce inflammation.\n"
            "- Ensure adequate Vitamin C and Zinc intake to support wound recovery.\n"
            "- Avoid spicy or highly acidic foods if experiencing post-op nausea.\n"
            "- Maintain steady hydration, avoiding sugary drinks and heavy caffeine."
        )
    elif stage_label == "Stage II":
        nutrition = (
            "- Focus on maintaining a stable body weight throughout active treatment.\n"
            "- Eat small, nutrient-dense meals every 3-4 hours to prevent fatigue.\n"
            "- Keep easy-to-digest snacks (crackers, bananas) nearby at all times.\n"
            "- Avoid raw or undercooked foods to prevent treatment-related infections.\n"
            "- Sip ginger tea or peppermint broth to naturally soothe the stomach."
        )
    elif stage_label == "Stage III":
        nutrition = (
            "- Transition to soft, moist foods if chemoradiation causes throat soreness.\n"
            "- Utilize high-calorie, high-protein smoothies when solid food is difficult.\n"
            "- Avoid extreme hot or cold foods that may trigger sensitive nerve endings.\n"
            "- Rinse mouth with baking soda/salt water before meals to improve taste.\n"
            "- Work closely with a clinical oncology dietitian to track daily macros."
        )
    else: # Stage IV
        nutrition = (
            "- Focus entirely on comfort feeding—eat whatever sounds appealing.\n"
            "- Utilize liquid nutritional supplements (e.g., Ensure) to prevent cachexia.\n"
            "- Remove all dietary restrictions unless specifically ordered by a doctor.\n"
            "- Offer very small bites of favorite foods to stimulate a fading appetite.\n"
            "- Keep meals visually appealing and serve them in a relaxed environment."
        )
        
    # Dynamically inject symptom-specific nutrition advice
    if symp_weight and stage_label != "Normal":
        nutrition_list = nutrition.split('\n')
        nutrition_list[-1] = "- ⚠️ CRITICAL: Add healthy fats (avocados, oils) to severely boost calories."
        nutrition_list[-2] = "- ⚠️ CRITICAL: Drink high-protein clinical supplements between every meal."
        nutrition = '\n'.join(nutrition_list)


    if stage_label == "Normal":
        lifestyle = (
            "- Adhere strictly to annual health and preventative oncology screenings.\n"
            "- Ensure routine vaccinations (Flu, Pneumonia, COVID-19) are up to date.\n"
            "- Avoid environments with heavy secondhand smoke or industrial hazards.\n"
            "- Establish a consistent daily routine for restorative sleep (7-9 hours).\n"
            "- Use sunscreen and limit exposure to known environmental carcinogens."
        )
    elif stage_label == "Stage I":
        lifestyle = (
            "- Strictly adhere to the 3-6 month surveillance CT scan schedule.\n"
            "- Implement total and permanent smoking and vaping cessation immediately.\n"
            "- Practice meticulous hand hygiene to prevent opportunistic infections.\n"
            "- Ease back into work and daily routines gradually to avoid exhaustion.\n"
            "- Keep a detailed medical journal of any new or recurring physical symptoms."
        )
    elif stage_label == "Stage II":
        lifestyle = (
            "- Coordinate with employers for FMLA or flexible work-from-home options.\n"
            "- Utilize HEPA air purifiers in the bedroom to minimize household dust.\n"
            "- Avoid large crowds or sick individuals while immune system is compromised.\n"
            "- Take daily temperature checks and report any fever to the care team.\n"
            "- Prepare a 'go-bag' with essentials for unexpected hospital visits."
        )
    elif stage_label == "Stage III":
        lifestyle = (
            "- Designate a primary caregiver to help manage complex medication schedules.\n"
            "- Modify the home environment (e.g., shower chairs) to conserve energy.\n"
            "- Ensure advanced directives and medical power of attorney are documented.\n"
            "- Keep emergency contact numbers and medical charts easily accessible.\n"
            "- Prioritize aggressive sleep hygiene to allow the body to recover from chemo."
        )
    else: # Stage IV
        lifestyle = (
            "- Focus home life on maximizing daily physical and emotional comfort.\n"
            "- Coordinate with hospice or palliative care teams for home nursing support.\n"
            "- Secure a hospital bed or specialized mattresses to prevent pressure ulcers.\n"
            "- Keep breakthrough pain medications strictly organized and accessible.\n"
            "- Allow friends and community members to handle chores and meal prep."
        )

    return physical, mental, nutrition, lifestyle


st.title("Clinical Decision Support System for early lung cancer detection using Adam And GWO Optimized CNN")
st.markdown("Automated TNM Staging, Data-Driven Treatment Planning, & Recommendation Support System.")

with st.sidebar:
    st.header("Patient Demographics")
    age = st.number_input("Age", 18, 100, 62)
    gender = st.selectbox("Biological Sex", ["Male", "Female"])
    smoking = st.selectbox("Smoking History", ["Current", "Former", "Never"])
    
    st.markdown("---")
    st.header("Clinical Symptoms Checklist")
    symp_cough = st.checkbox("Persistent Cough", value=False)
    symp_sob = st.checkbox("Shortness of Breath (Dyspnea)", value=False)
    symp_chest = st.checkbox("Chest Pain", value=False)
    symp_weight = st.checkbox("Unexplained Weight Loss", value=False)
    symp_fatigue = st.checkbox("Severe Fatigue / Malaise", value=False)

# Build active symptoms list for Gemini & XGBoost
active_symptoms = []
if symp_cough: active_symptoms.append("Persistent Cough")
if symp_sob: active_symptoms.append("Shortness of Breath")
if symp_chest: active_symptoms.append("Chest Pain")
if symp_weight: active_symptoms.append("Weight Loss")
if symp_fatigue: active_symptoms.append("Fatigue")

# Single file uploader
uploaded_file = st.file_uploader("Upload Axial CT Scan Slice (DICOM-derived PNG/JPG)", type=['png', 'jpg', 'jpeg'])

# Memory logic: Reset if file is removed
if uploaded_file is None:
    st.session_state.pipeline_run = False
    st.info("Awaiting CT scan slice upload to initiate inference.")
    st.info("Note: please only upload actual CT scan of lungs")
else:
    # Save the button click to memory
    if st.button("Run Full Clinical Diagnostic Pipeline"):
        st.session_state.pipeline_run = True
        
    # If memory says it ran, display the whole dashboard
    if st.session_state.pipeline_run:
        pil_img = Image.open(uploaded_file)
        img_array_rgb = np.array(pil_img.convert('RGB'))
        
        is_valid_ct, message = validate_ct_scan(img_array_rgb)
        
        if not is_valid_ct:
            st.error(f"🛑 **Pipeline Security Intercept:** {message}")
        else:
            input_tensor, display_gray = preprocess_for_inference(pil_img)
            
            if adam_model is None or gwo_model is None:
                st.error("Diagnostic models missing. Check the 'models/' directory.")
            else:
                st.divider()
                
                # --- PHASE 1: DUAL NEURAL NETWORK INFERENCE ---
                adam_pred = adam_model.predict(input_tensor, verbose=0)[0, :, :, 0]
                gwo_pred = gwo_model.predict(input_tensor, verbose=0)[0, :, :, 0]
                
                # --- PHASE 2: SOFT ENSEMBLE CONSENSUS ---
                ensemble_pred = (adam_pred + gwo_pred) / 2.0
                
                # --- PHASE 3: GEOMETRIC PHYSICS EXTRACTION ---
                adam_size, adam_mask = calculate_physical_size(adam_pred, 0.50)
                gwo_size, gwo_mask = calculate_physical_size(gwo_pred, 0.50)
                consensus_size, consensus_mask = calculate_physical_size(ensemble_pred, 0.50)
                
                # --- PHASE 4: VISUAL OVERLAY DISPLAY ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    overlay_adam = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB)
                    overlay_adam[adam_mask == 255] = [255, 0, 0]
                    st.image(overlay_adam, caption=f"Adam U-Net ({adam_size} cm)", use_container_width=True)
                        
                with col2:
                    overlay_gwo = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB)
                    overlay_gwo[gwo_mask == 255] = [255, 0, 0]
                    st.image(overlay_gwo, caption=f"GWO + Adam Hybrid ({gwo_size} cm)", use_container_width=True)
                    
                with col3:
                    overlay_consensus = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB)
                    overlay_consensus[consensus_mask == 255] = [0, 255, 0] 
                    st.image(overlay_consensus, caption=f"Consensus Segment ({consensus_size} cm)", use_container_width=True)

                st.divider()
                
                # --- PHASE 5: MEDICAL DIAGNOSIS & TREATMENT ---
                stage_label, descriptor = ajcc_tnm_staging(consensus_size)
                st.session_state.stage_label = stage_label # Save Stage for Gemini Assistant
                
                c_diag, c_ai = st.columns(2)
                
                with c_diag:
                    st.subheader("1. Radiologic Diagnosis & Staging")
                    st.metric("AJCC TNM Classification", stage_label, delta=f"Tumor Diameter: {consensus_size} cm")
                    st.write(f"**Pathology Descriptor:** {descriptor}")
                    
                with c_ai:
                    st.subheader("2. AI Medical Treatment Plan")
                    
                    patient_stage = "Normal"
                    if "Stage I" in stage_label: patient_stage = "Stage I"
                    elif "Stage II" in stage_label: patient_stage = "Stage II"
                    elif "Stage III" in stage_label: patient_stage = "Stage III"
                    elif "Stage IV" in stage_label: patient_stage = "Stage IV"
                    
                    if patient_stage == "Normal":
                        st.success("No active clinical intervention required. Routine Annual CT Screening.")
                    elif xgb_model is not None and xgb_meta is not None:
                        encoders = xgb_meta['encoders']
                        features = xgb_meta['features']
                        stage_col = xgb_meta['stage_col']
                        symptom_cols = xgb_meta['symptom_cols']
                        
                        symptom_inputs = [symp_cough, symp_sob, symp_chest, symp_weight, symp_fatigue]
                        
                        raw_data = {stage_col: patient_stage}
                        for i, col in enumerate(symptom_cols):
                            raw_data[col] = "yes" if (i < len(symptom_inputs) and symptom_inputs[i]) else "no"
                            
                        try:
                            encoded_data = {}
                            for col in features:
                                encoded_data[col] = encoders[col].transform([raw_data[col]])[0]
                                
                            df_input = pd.DataFrame([encoded_data], columns=features)
                            prediction = xgb_model.predict(df_input)[0]
                            
                            treatment_text = encoders['TARGET'].inverse_transform([prediction])[0]
                            
                            st.error(f"**Prescribed Action:** {treatment_text}")
                            
                        except Exception as err:
                            st.warning(f"Feature mapping discrepancy: {err}")
                    else:
                        st.warning("Data-driven treatment engine not loaded.")

                st.divider()

                # --- PHASE 6: HOLISTIC RECOMMENDATION SYSTEM ---
                st.subheader("3. Tailored Patient Rehabilitation & Lifestyle Plan")
                
                rec_phys, rec_ment, rec_nutr, rec_life = generate_holistic_care_plan(
                    stage_label, symp_fatigue, symp_weight, symp_sob
                )
                
                c_phys, c_ment, c_nutr, c_life = st.columns(4)
                with c_phys:
                    st.info(f"**🏃 Physical Exercise**\n\n{rec_phys}")
                with c_ment:
                    st.warning(f"**🧠 Mental Wellness**\n\n{rec_ment}")
                with c_nutr:
                    st.success(f"**🥗 Nutritional Diet**\n\n{rec_nutr}")
                with c_life:
                    st.error(f"**🛌 Lifestyle & Routine**\n\n{rec_life}")


st.divider()
st.subheader("💬 Patient Dietary & Lifestyle Assistant")
st.caption("Ask specific food or activity questions (e.g., *'Can I eat mutton?'*). The AI automatically factors in your current cancer stage and active symptoms to keep advice medically safe and concise.")

user_question = st.text_input("Ask a dietary or routine question:")

if st.button("Check Safety"):
    if not user_question:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Analyzing against clinical metadata..."):
            # Pulls the active stage from the last processed CT scan and symptoms from the sidebar
            current_stage = st.session_state.stage_label
            advice = ask_dietary_guidance(user_question, current_stage, active_symptoms)
            
            # Displays the output cleanly. Flags unsafe answers in red (error).
            if advice.lower().startswith("no") or "avoid" in advice.lower() or "crash" in advice.lower():
                st.error(f"**Safety Advice:** {advice}")
            else:
                st.success(f"**Safety Advice:** {advice}")
