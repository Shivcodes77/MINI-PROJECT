"""
drSaab - Hybrid Symptom & Health Advisor for India (v3.2)
=========================================================

WHAT THIS APP DOES
    0. CHAT (new in v3.0): talk to drSaab like you would to a doctor.
       "My lower back hurts since a deadlift session without a belt - is it
       sciatica or a disc bulge?" gets a proper answer: drSaab understands
       plain sentences, Hinglish (bukhar, kamar dard), typos, negations
       ("no fever"), durations ("4 din se"), severity, named conditions and
       even lab reports ("NS1 came positive"), asks ONE focused follow-up
       question at a time, remembers your answers, and warns about
       emergencies before anything else. The full report appears below the
       chat as the conversation develops.
    0b. MEDICAL SCHOOL (new in v3.2): drSaab can now STUDY. Upload medical
       books, PDFs, chapter notes or real doctor-patient conversation
       transcripts in "Medical school" mode, and it reads them like a
       student: it highlights symptoms, conditions, tests and medicines,
       counts which signs co-occur with which condition, and CONCLUDES -
       brand-new conditions are learned from scratch (with urgency, red
       flags, tests), known conditions gain extra signs and tests, and the
       facts feed the ML model too. Everything is remembered in
       medical_school.json between sessions.
    1. You select your symptoms and pick your Indian city (or type any city).
    2. The app fetches LIVE weather + air quality for your city using a
       free API (Open-Meteo - no account, no API key, no payment).
    3. A HYBRID engine predicts what might explain your symptoms:
         - RULES half: a transparent scoring engine checks your symptoms
           against 31 conditions common in India, boosted by the season
           and the live weather ("what is going around right now").
         - LEARNING half: a machine-learning model (scikit-learn) trained
           on patient records - it ships with about 2,000 example records
           generated from the knowledge base, and you can teach it more
           in "Doctor mode" by uploading a CSV of real (anonymised)
           patient records: symptoms -> diagnosis -> tests -> medicines.
         - INTERNET half (new in v2.2): live symptom matching against
           Wikipedia and Wikidata - the same sources doctors read. The
           app downloads which symptoms belong to each disease, caches
           it in internet_knowledge.json, and adds an "internet evidence"
           score for every candidate condition.
       You control the mix of the engines with sliders.
    4. It shows the top conditions with the reasons from ALL THREE engines,
       plus the most similar past patient records.
    5. It SUGGESTS TESTS & SCANS - from the knowledge base AND from what
       similar patients actually had done. If you already have results,
       type them in and the prediction is re-ranked using the findings.
    6. It lists common medicines with typical adult doses (with common
       Indian brand examples) and finds the nearest hospitals/clinics
       on a live map (free OpenStreetMap APIs).

HOW TO RUN IT ON YOUR OWN COMPUTER (100% free)
    1. Install Python 3.9 or newer from https://www.python.org
       (Windows: tick "Add Python to PATH" during install)
    2. Open a terminal / command prompt inside this folder and run:
           pip install -r requirements.txt
           streamlit run app.py
    3. Your browser opens automatically at http://localhost:8501

HOW TO PUT IT ONLINE FOR FREE (Streamlit Community Cloud)
    1. Create a free account at https://github.com and upload this whole
       folder into a new public repository.
    2. Go to https://share.streamlit.io, sign in with GitHub, click
       "New app", choose your repository, set the main file to app.py,
       and click Deploy.
    3. You get a permanent free link like:
           https://your-name-drsaab.streamlit.app
    4. To make your uploaded patient data permanent, download it in
       Doctor mode, name it patient_data.csv, and commit it to the repo -
       drSaab loads that file automatically at startup.

DISCLAIMER
    This is an educational project. It is NOT a medical device, NOT a
    diagnosis, and NOT a substitute for a qualified doctor. Lab findings
    sharpen the ranking but only a doctor can confirm a diagnosis.
    Medicine doses shown are typical ADULT reference doses - always
    confirm with a doctor or pharmacist.

HOW THE CODE IS ORGANISED (for the learner)
    PART 1 - The medical knowledge base (diseases, symptoms, seasons,
             medicines, typical tests, lab findings - all editable dicts)
    PART 2 - The free internet APIs (weather, air quality, nearby hospitals)
    PART 2B - The internet knowledge fetcher (Wikidata SPARQL + Wikipedia
             summaries, saved to a cache file and refreshed weekly)
    PART 3 - The rule-based prediction engine (symptom matching + boosts)
    PART 4 - The learning engine (ML: seed data, CSV import, training,
             prediction, similar patients, lab-findings refinement)
    PART 4B - The conversation brain (chat mode: intent detection, free-text
             symptom extraction, doctor-style replies, follow-up questions)
    PART 5 - The web interface (chat + picker + doctor mode, Streamlit)
"""

from collections import Counter
from datetime import datetime, timezone
import json
import os
import random
import re

import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# PART 4C - the medical school: drSaab's engine for learning from books,
# PDFs and doctor-patient conversations (see medical_school.py).
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import medical_school as ms

# ---------------------------------------------------------------------------
# PART 1 - THE MEDICAL KNOWLEDGE BASE
# ---------------------------------------------------------------------------
# This is the app's "brain". Every symptom below is a string; every disease
# maps those strings to a weight:
#     3 = very typical, 2 = common, 1 = sometimes happens
# To add a new disease, copy one block, change the fields, done.
# ---------------------------------------------------------------------------

# Every symptom the user can pick (also the options in the dropdown).
SYMPTOMS = [
    "Fever", "High fever (39°C / 102°F or above)", "Chills or shivering",
    "Night sweats", "Heavy sweating",
    "Runny nose", "Blocked nose", "Sneezing", "Itchy eyes", "Watery eyes",
    "Red eyes", "Eye discharge or crusty eyelids", "Gritty feeling in the eyes",
    "Sore throat", "Pain when swallowing", "Swollen glands in the neck",
    "White patches on tonsils",
    "Dry cough", "Cough with phlegm", "Cough worse at night",
    "Shortness of breath", "Wheezing", "Chest tightness",
    "Chest pain when breathing",
    "Headache", "One-sided throbbing headache",
    "Band-like pressure around the head", "Stiff or tense neck and shoulders",
    "Light sensitivity", "Sound sensitivity",
    "Visual aura (flashing lights or zigzags)",
    "Body aches / muscle pain", "Joint pain", "Pain behind the eyes",
    "Facial pain or pressure", "Reduced sense of smell",
    "Loss of taste or smell", "Fatigue or weakness", "Dizziness",
    "Nausea", "Vomiting", "Diarrhea", "Constipation",
    "Abdominal pain or cramps", "Bloating",
    "Heartburn / burning behind the chest", "Sour or bitter taste in the throat",
    "Belching or burping",
    "Burning when urinating", "Frequent urination",
    "Cloudy or strong-smelling urine", "Blood in urine", "Lower back pain",
    "Itchy rash", "Rash with small fluid-filled blisters",
    "Ring-shaped scaly rash", "Skin peeling or flaking",
    "Excessive thirst", "Dry mouth", "Dark or little urine",
    "Ear pain", "Ear discharge", "Reduced hearing", "Loss of appetite",
    "Muscle cramps", "Pale, cool or clammy skin",
    # v2.0 additions - extra clues for diseases that are common in India:
    "Coughing up blood", "Unintentional weight loss",
    "Yellow eyes or skin (jaundice)", "Pale (clay-coloured) stools",
    "Severe joint pain (hands and feet)", "Bleeding gums or nose",
    # v3.0 additions - back and nerve symptoms, so patients who describe their
    # problem in their own words (chat) can be understood properly:
    "Pain shooting down the leg (below the knee)",
    "Numbness or tingling in the leg or foot",
    "Weakness in the leg or foot",
    "Back pain worse when sitting or bending forward",
    "Back pain worse when coughing or sneezing",
    "Back pain that started after lifting or twisting",
    "Back stiffness in the morning",
]

# Quick-pick list for the sidebar (v2.0 - India focus). Any other city can
# still be typed in by hand - this list is just a convenience.
INDIA_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Kanpur",
    "Nagpur", "Indore", "Bhopal", "Patna", "Kochi", "Coimbatore",
    "Guwahati", "Chandigarh", "Dehradun", "Varanasi", "Ludhiana", "Visakhapatnam",
]

# Season tags a disease can use:
#   "winter" / "spring" / "summer" / "autumn"  - normal temperate seasons
#       (the engine flips them automatically for the Southern Hemisphere)
#   "monsoon" - rough tropical wet-season months (Jun-Oct north of equator)
#
# Weather boost tags (checked against LIVE weather in PART 3):
#   "cold"       temperature <= 10°C
#   "heat"       temperature >= 35°C
#   "warm_rain"  rain + 20-35°C   (mosquito / contaminated-water illnesses)
#   "humid"      humidity >= 78% and temperature >= 24°C
#   "dry_windy"  humidity <= 45% and wind >= 25 km/h (pollen spreads)
#   "poor_air"   US AQI >= 100 or PM2.5 >= 40 µg/m³

DISEASES = {}

DISEASES["Common Cold"] = {
    "about": "A mild viral infection of the nose and throat - the most common illness on earth.",
    "symptoms": {"Runny nose": 3, "Sneezing": 3, "Blocked nose": 3, "Sore throat": 2,
                 "Dry cough": 2, "Headache": 1, "Fatigue or weakness": 1, "Fever": 1},
    "seasons": ["autumn", "winter"],
    "weather_boosts": [("cold", 0.08)],
    "urgency": "self-care",
    "duration": "Usually settles in 7-10 days",
    "medicines": [
        ("Paracetamol", "500 mg every 4-6 hours as needed (max 3,000 mg/day)", "Fever, headache, aches"),
        ("Cetirizine", "10 mg once daily (may cause mild drowsiness)", "Runny nose, sneezing"),
        ("Saline nasal spray", "2-3 sprays per nostril, as often as needed", "Blocked nose"),
    ],
    "home_care": [
        "Drink plenty of warm fluids",
        "Rest as much as possible",
        "Steam inhalation for a blocked nose",
        "Warm salt-water gargle 2-3 times a day for a sore throat",
        "Honey and lemon in warm water soothes a cough (never give honey to babies under 1 year)",
    ],
    "red_flags": [
        "Fever above 39°C for more than 3 days",
        "Shortness of breath",
        "Symptoms lasting more than 10 days, or worsening after improving",
        "Severe facial pain or swelling",
    ],
}

DISEASES["Influenza (Flu)"] = {
    "about": "A respiratory virus that starts suddenly with fever and body aches; peaks in cold months.",
    "symptoms": {"High fever (39°C / 102°F or above)": 3, "Body aches / muscle pain": 3,
                 "Fatigue or weakness": 3, "Dry cough": 3, "Headache": 2,
                 "Chills or shivering": 2, "Sore throat": 2, "Blocked nose": 1},
    "seasons": ["winter"],
    "weather_boosts": [("cold", 0.10)],
    "urgency": "see-doctor",
    "duration": "Fever 3-5 days; tiredness can linger 1-2 weeks",
    "medicines": [
        ("Paracetamol", "500-1000 mg every 4-6 hours (max 3,000 mg/day)", "Fever, aches, headache"),
        ("Warm fluids + rest", "Throughout the day", "Recovery"),
    ],
    "home_care": [
        "Rest and stay home until fever-free for 24 hours (protects others)",
        "Drink extra fluids - water, soup, oral rehydration solution",
        "Warm salt-water gargles for the sore throat",
        "Antiviral medicine (e.g. oseltamivir) is prescription-only and works best within 48 hours - see a doctor early if you are pregnant, elderly, or have asthma/diabetes",
    ],
    "red_flags": [
        "Difficulty breathing or chest pain",
        "Confusion or extreme drowsiness",
        "Fever that returns after improving",
        "Very little urine output (dehydration)",
    ],
}

DISEASES["COVID-like Viral Illness"] = {
    "about": "A respiratory viral infection; a lost or changed sense of taste/smell is a strong clue.",
    "symptoms": {"Fever": 3, "Dry cough": 3, "Fatigue or weakness": 3,
                 "Loss of taste or smell": 3, "Sore throat": 2, "Headache": 2,
                 "Body aches / muscle pain": 2, "Shortness of breath": 2},
    "seasons": ["winter"],
    "weather_boosts": [("cold", 0.05)],
    "urgency": "see-doctor",
    "duration": "Most people recover in 1-2 weeks",
    "medicines": [
        ("Paracetamol", "500 mg every 4-6 hours as needed (max 3,000 mg/day)", "Fever, aches, sore throat"),
        ("Rapid antigen test", "One test, or two 48 hours apart", "Confirms the cause"),
    ],
    "home_care": [
        "Take a rapid antigen test if you have one",
        "Isolate until fever-free for 24 hours and improving; wear a mask around others",
        "Rest and hydrate well",
        "Monitor your breathing - count how many words you can say in one breath",
    ],
    "red_flags": [
        "Difficulty breathing, chest pressure, or blue lips - emergency",
        "Oxygen monitor reading below 94%",
        "Confusion or inability to stay awake",
    ],
}

DISEASES["Allergic Rhinitis (Hay Fever)"] = {
    "about": "An allergy to pollen or dust - sneezing and itchy eyes without fever.",
    "symptoms": {"Sneezing": 3, "Itchy eyes": 3, "Watery eyes": 3, "Runny nose": 3,
                 "Blocked nose": 2, "Red eyes": 2, "Dry cough": 1},
    "seasons": ["spring", "summer", "autumn"],
    "weather_boosts": [("dry_windy", 0.08)],
    "urgency": "self-care",
    "duration": "Continues while exposure to the allergen continues",
    "medicines": [
        ("Cetirizine", "10 mg once daily (may cause mild drowsiness)", "Sneezing, itchy eyes, runny nose"),
        ("Loratadine", "10 mg once daily (less drowsy)", "Same symptoms, alternative option"),
        ("Antihistamine eye drops", "1 drop per eye, twice daily", "Itchy, watery eyes"),
        ("Saline nasal rinse", "Once or twice daily", "Washes pollen out of the nose"),
    ],
    "home_care": [
        "Shower and change clothes after coming indoors",
        "Keep windows closed on dry, windy days",
        "Sunglasses help keep pollen off the eyes",
        "Vacuum and damp-dust regularly",
    ],
    "red_flags": [
        "Wheezing or chest tightness (could be asthma - see a doctor)",
        "Symptoms not controlled by antihistamines after 2 weeks",
    ],
}

DISEASES["Sinusitis"] = {
    "about": "Inflamed sinus cavities, usually after a cold - facial pressure and thick discharge.",
    "symptoms": {"Facial pain or pressure": 3, "Blocked nose": 3, "Runny nose": 2,
                 "Reduced sense of smell": 2, "Headache": 2, "Cough worse at night": 1,
                 "Fever": 1, "Fatigue or weakness": 1},
    "seasons": ["winter"],
    "weather_boosts": [("cold", 0.05)],
    "urgency": "see-doctor",
    "duration": "Acute sinusitis lasts 2-3 weeks",
    "medicines": [
        ("Saline nasal rinse", "Twice daily", "Clears thick mucus"),
        ("Paracetamol or Ibuprofen", "Paracetamol 500-1000 mg q4-6h / Ibuprofen 400 mg q6-8h with food", "Facial pain, headache"),
        ("Steroid nasal spray", "1-2 sprays per nostril daily (ask pharmacist)", "Reduces sinus swelling"),
    ],
    "home_care": [
        "Steam inhalation 2-3 times a day",
        "Warm compress over the cheeks and forehead",
        "Sleep with the head slightly raised",
        "Drink plenty of fluids to thin the mucus",
    ],
    "red_flags": [
        "Swelling around the eyes or vision changes - urgent",
        "Severe headache with confusion or high fever",
        "Symptoms beyond 10 days or worsening after improving",
    ],
}

DISEASES["Acute Bronchitis"] = {
    "about": "Inflammation of the airway tubes, nearly always viral - a nagging productive cough.",
    "symptoms": {"Cough with phlegm": 3, "Dry cough": 2, "Chest tightness": 2,
                 "Fatigue or weakness": 2, "Wheezing": 2, "Sore throat": 1, "Fever": 1},
    "seasons": ["winter"],
    "weather_boosts": [("cold", 0.06), ("poor_air", 0.10)],
    "urgency": "see-doctor",
    "duration": "The cough often lasts 2-3 weeks",
    "medicines": [
        ("Paracetamol", "500 mg every 4-6 hours as needed", "Fever, chest soreness"),
        ("Warm honey and lemon", "A spoonful as needed (not for babies under 1 year)", "Soothes the cough"),
        ("Dextromethorphan syrup", "As label directs, at bedtime (ask pharmacist)", "Cough that disturbs sleep"),
    ],
    "home_care": [
        "Antibiotics usually do NOT help (cause is viral) - never take leftover antibiotics",
        "Avoid smoke and dusty air",
        "Humidify the room air",
        "Rest and drink warm fluids",
    ],
    "red_flags": [
        "Fever above 39°C for more than 3 days",
        "Breathlessness at rest",
        "Coughing up blood",
        "Cough lasting beyond 3 weeks",
    ],
}

DISEASES["Asthma Flare-Up"] = {
    "about": "Airway narrowing with wheeze and chest tightness, triggered by cold air, dust or pollution.",
    "symptoms": {"Wheezing": 3, "Shortness of breath": 3, "Chest tightness": 3,
                 "Cough worse at night": 2, "Dry cough": 1},
    "seasons": ["winter"],
    "weather_boosts": [("cold", 0.08), ("poor_air", 0.12)],
    "urgency": "see-doctor",
    "duration": "A flare-up needs medical review even if it settles",
    "medicines": [
        ("Salbutamol reliever inhaler", "2 puffs, up to every 20 min in a flare (prescription item)", "Opens the airways fast"),
        ("Your preventer inhaler", "Continue daily as prescribed - do not stop it", "Long-term control"),
    ],
    "home_care": [
        "Sit upright, breathe slowly and steadily",
        "Move away from the trigger (smoke, dust, cold air, strong smells)",
        "If you do not have an inhaler, see a doctor today - do not wait",
    ],
    "red_flags": [
        "Reliever inhaler not helping, or needed more than every 4 hours",
        "Cannot finish a sentence in one breath - EMERGENCY",
        "Lips or face turning blue - call emergency services now",
    ],
}

DISEASES["Pneumonia"] = {
    "about": "A serious lung infection (bacterial or viral) - fever with a wet cough and fast breathing.",
    "symptoms": {"High fever (39°C / 102°F or above)": 3, "Chills or shivering": 3,
                 "Cough with phlegm": 3, "Shortness of breath": 3,
                 "Chest pain when breathing": 3, "Fatigue or weakness": 2, "Night sweats": 1},
    "seasons": ["winter"],
    "weather_boosts": [("cold", 0.06), ("poor_air", 0.08)],
    "urgency": "urgent",
    "duration": "Recovery takes 1-4 weeks with proper treatment",
    "medicines": [
        ("Prescription antibiotics", "A doctor must examine you and choose the medicine - this is not self-treatable", "Treats the infection"),
        ("Paracetamol", "500-1000 mg every 4-6 hours on the way to the doctor", "Fever and chest pain"),
    ],
    "home_care": [
        "Do not delay - pneumonia needs an examination, temperature/pulse/oxygen check and often a chest X-ray",
        "Rest completely; do not go to work or school",
        "Sip fluids steadily",
    ],
    "red_flags": [
        "Fast breathing or breathlessness at rest - go to an emergency department NOW",
        "New confusion, or bluish lips",
        "Chest pain on breathing",
    ],
}

DISEASES["Migraine"] = {
    "about": "A one-sided throbbing headache with nausea and light sensitivity; often runs in families.",
    "symptoms": {"One-sided throbbing headache": 3, "Light sensitivity": 3, "Nausea": 2,
                 "Sound sensitivity": 2, "Visual aura (flashing lights or zigzags)": 2,
                 "Vomiting": 1, "Dizziness": 1},
    "seasons": [],
    "weather_boosts": [],
    "urgency": "see-doctor",
    "duration": "An attack lasts 4-72 hours",
    "medicines": [
        ("Ibuprofen", "400 mg with food, at the FIRST sign of the attack", "Pain - works best early"),
        ("Paracetamol", "1000 mg, may repeat after 4-6 hours", "Alternative painkiller"),
        ("Triptans", "Prescription-only - discuss with a doctor if attacks are frequent", "Specific migraine relief"),
    ],
    "home_care": [
        "Lie down in a dark, quiet room",
        "Cold compress on the forehead or neck",
        "Keep a headache diary to find your triggers (sleep, foods, stress, screens)",
        "Do not use simple painkillers on more than 10 days per month (rebound headaches)",
    ],
    "red_flags": [
        "Worst-ever headache or sudden thunderclap onset - emergency",
        "Headache with fever and a stiff neck - emergency",
        "Weakness, slurred speech or vision loss - emergency",
        "New headache pattern after age 50, or after a head injury",
    ],
}

DISEASES["Tension Headache"] = {
    "about": "A dull, band-like headache caused by stress, posture, screens or poor sleep.",
    "symptoms": {"Band-like pressure around the head": 3, "Stiff or tense neck and shoulders": 2,
                 "Headache": 2, "Fatigue or weakness": 1, "Light sensitivity": 1},
    "seasons": [],
    "weather_boosts": [],
    "urgency": "self-care",
    "duration": "30 minutes to a few hours",
    "medicines": [
        ("Paracetamol", "500-1000 mg, repeat after 4-6 hours if needed (max 3,000 mg/day)", "The pressure and ache"),
        ("Ibuprofen", "400 mg with food", "Alternative if paracetamol alone fails"),
    ],
    "home_care": [
        "Drink water - dehydration is a common trigger",
        "Take screen breaks: every 20 minutes look 20 feet away for 20 seconds",
        "Neck and shoulder stretches; check your chair and screen height",
        "Regular sleep schedule; cut late-day caffeine",
    ],
    "red_flags": [
        "Headaches that wake you from sleep",
        "A steadily worsening pattern over weeks",
        "Fever or neurological symptoms with the headache",
    ],
}

DISEASES["Viral Fever (general)"] = {
    "about": "A catch-all for everyday viral fevers that are not clearly flu, dengue or typhoid.",
    "symptoms": {"Fever": 3, "Fatigue or weakness": 3, "Body aches / muscle pain": 2,
                 "Headache": 2, "Chills or shivering": 1, "Sore throat": 1,
                 "Loss of appetite": 1},
    "seasons": [],
    "weather_boosts": [("warm_rain", 0.06)],
    "urgency": "self-care",
    "duration": "Usually 3-5 days",
    "medicines": [
        ("Paracetamol", "500 mg every 4-6 hours as needed (max 3,000 mg/day)", "Fever and aches"),
        ("ORS / extra fluids", "Sip continuously through the day", "Prevents dehydration"),
    ],
    "home_care": [
        "Rest and drink plenty of fluids",
        "Light, easy-to-digest meals",
        "Lukewarm sponging if the fever is high",
    ],
    "red_flags": [
        "Fever lasting more than 3 days, or above 39.5°C not responding to paracetamol",
        "A rash appears",
        "Severe weakness or confusion",
    ],
}

DISEASES["Strep Throat (Bacterial)"] = {
    "about": "A bacterial throat infection - severe pain with swollen neck glands and no cough.",
    "symptoms": {"Sore throat": 3, "Pain when swallowing": 3, "Swollen glands in the neck": 3,
                 "Fever": 2, "White patches on tonsils": 2, "Headache": 1},
    "seasons": ["winter", "spring"],
    "weather_boosts": [("cold", 0.05)],
    "urgency": "see-doctor",
    "duration": "Improves in 2-3 days once on the right antibiotic",
    "medicines": [
        ("Antibiotics (prescription)", "A doctor confirms with a rapid test/throat swab and prescribes - typically penicillin or amoxicillin", "Cures the infection"),
        ("Paracetamol or Ibuprofen", "Standard doses with food", "Throat pain and fever meanwhile"),
    ],
    "home_care": [
        "Warm salt-water gargles 3-4 times a day",
        "Cold fluids and ice pops surprisingly soothe the throat",
        "Soft foods; rest your voice",
    ],
    "red_flags": [
        "Drooling, muffled voice, or trouble opening the mouth - urgent",
        "Cannot swallow even liquids",
        "Difficulty breathing - emergency",
    ],
}

DISEASES["Gastroenteritis (Stomach Flu / Food Poisoning)"] = {
    "about": "An infection of the stomach and gut from a virus or contaminated food/water.",
    "symptoms": {"Diarrhea": 3, "Nausea": 3, "Vomiting": 3, "Abdominal pain or cramps": 3,
                 "Fatigue or weakness": 1, "Fever": 1},
    "seasons": ["summer"],
    "weather_boosts": [("heat", 0.08), ("warm_rain", 0.05)],
    "urgency": "self-care",
    "duration": "Most cases settle in 1-3 days",
    "medicines": [
        ("ORS - oral rehydration salts", "Sip continuously - this is the main treatment, not a tablet", "Replaces lost water and salts"),
        ("Zinc", "20 mg once daily for 10-14 days (especially children)", "Shortens diarrhea in children"),
        ("Loperamide", "2 mg after each loose stool, ONLY if no fever and no blood in stool (ask pharmacist)", "Reduces stool frequency"),
    ],
    "home_care": [
        "Small, frequent sips of fluid - ORS, watery rice water, weak tea",
        "Light foods as tolerated: banana, rice, toast, crackers",
        "Strict hand hygiene - this is how it spreads",
        "Skip dairy, oily and spicy food until recovered",
    ],
    "red_flags": [
        "Blood in the stool",
        "Signs of dehydration: very dark or little urine, dizziness on standing",
        "High fever or severe abdominal pain",
        "Vomiting everything for more than 12 hours (children) / 24 hours (adults)",
    ],
}

DISEASES["Acid Reflux (GERD)"] = {
    "about": "Stomach acid rising into the food pipe - burning behind the chest after meals.",
    "symptoms": {"Heartburn / burning behind the chest": 3, "Sour or bitter taste in the throat": 3,
                 "Belching or burping": 2, "Bloating": 2, "Dry cough": 1, "Sore throat": 1},
    "seasons": [],
    "weather_boosts": [],
    "urgency": "self-care",
    "duration": "Improves in days to weeks with habits and medicine",
    "medicines": [
        ("Antacid (calcium carbonate)", "10-20 mL after meals and at bedtime", "Instant burning relief"),
        ("Omeprazole", "20 mg once daily, 30 minutes before breakfast, up to 14 days (OTC in many countries)", "Switches down acid production"),
    ],
    "home_care": [
        "Smaller meals; do not lie down within 2-3 hours of eating",
        "Raise the head of the bed by 10-15 cm",
        "Cut back coffee, alcohol, spicy and fatty food; stop smoking",
        "Loose clothes - tight belts press the stomach",
    ],
    "red_flags": [
        "Chest pain spreading to arm/jaw with sweating - treat as a heart attack, call emergency services",
        "Difficulty swallowing or food getting stuck",
        "Unintentional weight loss",
        "Vomiting blood or black, tarry stools",
    ],
}

DISEASES["Urinary Tract Infection (UTI)"] = {
    "about": "A bacterial infection of the bladder - burning urine with a constant urge to go.",
    "symptoms": {"Burning when urinating": 3, "Frequent urination": 3,
                 "Cloudy or strong-smelling urine": 2, "Blood in urine": 2,
                 "Abdominal pain or cramps": 2, "Lower back pain": 2, "Fever": 1},
    "seasons": ["summer"],
    "weather_boosts": [("heat", 0.06)],
    "urgency": "see-doctor",
    "duration": "Improves within 1-3 days on the right antibiotic",
    "medicines": [
        ("Antibiotics (prescription)", "Commonly nitrofurantoin or fosfomycin - a doctor chooses after checking you", "The actual cure"),
        ("Paracetamol", "500-1000 mg as needed", "Pain and fever meanwhile"),
    ],
    "home_care": [
        "Drink 2-3 litres of water a day",
        "Do not hold urine; urinate soon after sex",
        "Wipe front to back",
        "Cranberry products may help prevention but do not cure an infection",
    ],
    "red_flags": [
        "Fever + flank/side back pain + vomiting = possible kidney infection - see a doctor the same day",
        "Infection during pregnancy",
        "Three or more infections in a year",
    ],
}

DISEASES["Dengue Fever"] = {
    "about": "A mosquito-borne viral fever with intense body aches and pain behind the eyes.",
    "symptoms": {"High fever (39°C / 102°F or above)": 3, "Pain behind the eyes": 3,
                 "Body aches / muscle pain": 3, "Fatigue or weakness": 3, "Headache": 2,
                 "Nausea": 2, "Itchy rash": 2, "Joint pain": 2, "Bleeding gums or nose": 2},
    "seasons": ["monsoon"],
    "weather_boosts": [("warm_rain", 0.12), ("humid", 0.05)],
    "urgency": "see-doctor",
    "duration": "Fever 2-7 days; full recovery 1-2 weeks",
    "medicines": [
        ("Paracetamol ONLY", "500 mg every 4-6 hours (max 3,000 mg/day)", "Fever and aches"),
        ("NEVER ibuprofen/aspirin/NSAIDs", "They increase bleeding risk in dengue - avoid completely", "Safety warning"),
    ],
    "home_care": [
        "See a doctor early for a blood test (dengue NS1/platelets)",
        "Aggressive fluids: water, ORS, coconut water, soup",
        "Sleep under a mosquito net while feverish so mosquitoes cannot spread it further",
        "Daily platelet checks if the doctor advises",
    ],
    "red_flags": [
        "Bleeding gums or nose, or dark/black stools - hospital immediately",
        "Severe abdominal pain or persistent vomiting",
        "Drowsiness, restlessness, or cold clammy skin (signs of shock) - emergency",
    ],
}

DISEASES["Typhoid Fever"] = {
    "about": "A bacterial fever from contaminated food or water; rises slowly and lasts long.",
    "symptoms": {"Fever": 3, "Fatigue or weakness": 3, "Abdominal pain or cramps": 2,
                 "Loss of appetite": 2, "Headache": 2, "Constipation": 1, "Diarrhea": 1,
                 "Body aches / muscle pain": 1},
    "seasons": ["monsoon", "summer"],
    "weather_boosts": [("warm_rain", 0.10), ("humid", 0.04)],
    "urgency": "see-doctor",
    "duration": "With proper treatment: 1-2 weeks",
    "medicines": [
        ("Antibiotics (prescription)", "Commonly azithromycin or cefixime - the doctor decides after blood tests", "The only cure"),
        ("Paracetamol", "500 mg every 4-6 hours", "Fever meanwhile"),
    ],
    "home_care": [
        "Get a blood test (typhoid IgM/blood culture) rather than guessing",
        "Drink only clean/boiled water; eat freshly cooked food",
        "Strict hand hygiene - you remain contagious while recovering",
    ],
    "red_flags": [
        "Fever lasting more than 5 days",
        "Confusion or extreme drowsiness",
        "Severe abdominal pain or a swollen belly - emergency (intestine perforation)",
    ],
}

DISEASES["Malaria"] = {
    "about": "A mosquito-borne parasite infection: cyclical fever with violent chills and sweats.",
    "symptoms": {"Chills or shivering": 3, "High fever (39°C / 102°F or above)": 3,
                 "Night sweats": 2, "Headache": 2, "Nausea": 2,
                 "Body aches / muscle pain": 2, "Fatigue or weakness": 2, "Vomiting": 1},
    "seasons": ["monsoon"],
    "weather_boosts": [("warm_rain", 0.12)],
    "urgency": "urgent",
    "duration": "Treated early: full recovery in 1-2 weeks",
    "medicines": [
        ("Antimalarials (prescription)", "Only after a confirmed blood test (RDT or microscopy) - same-day testing is essential", "The cure"),
        ("Paracetamol", "500 mg every 4-6 hours while awaiting results", "Fever"),
    ],
    "home_care": [
        "Get tested the same day - malaria is a medical emergency in some types (falciparum)",
        "Sleep under a treated mosquito net",
        "Fluids and rest",
    ],
    "red_flags": [
        "Any fever within a month of visiting a malaria area deserves a same-day test",
        "Confusion, seizures, or yellow eyes - severe malaria, hospital NOW",
        "Very dark urine with pale skin (severe anemia)",
    ],
}

DISEASES["Chickenpox"] = {
    "about": "A viral illness with itchy fluid blisters that appear in crops over several days.",
    "symptoms": {"Rash with small fluid-filled blisters": 3, "Fever": 2, "Itchy rash": 2,
                 "Fatigue or weakness": 1, "Loss of appetite": 1},
    "seasons": ["winter", "spring"],
    "weather_boosts": [],
    "urgency": "see-doctor",
    "duration": "New spots for 3-5 days; scabs within about a week",
    "medicines": [
        ("Calamine lotion", "Dab on spots as needed for the itch", "Itch relief"),
        ("Paracetamol", "Standard doses for fever - never aspirin or ibuprofen with chickenpox", "Fever"),
        ("Cetirizine", "10 mg at night if itch disturbs sleep", "Stronger itch relief"),
    ],
    "home_care": [
        "Cool baths; loose cotton clothes",
        "Keep nails short and clean (scratching causes scars and skin infection)",
        "Isolate until every blister has scabbed (usually 5-7 days)",
        "Adults with chickenpox should always see a doctor - it hits harder than in children",
    ],
    "red_flags": [
        "Blisters becoming red and pus-filled (skin infection)",
        "Drowsiness, repeated vomiting, or breathlessness",
        "Fever beyond 4 days",
    ],
}

DISEASES["Conjunctivitis (Eye Flu)"] = {
    "about": "Inflammation of the eye surface - red, watery, gritty eyes that spread by touch.",
    "symptoms": {"Red eyes": 3, "Eye discharge or crusty eyelids": 3, "Watery eyes": 2,
                 "Gritty feeling in the eyes": 2, "Itchy eyes": 1, "Light sensitivity": 1},
    "seasons": ["monsoon"],
    "weather_boosts": [("humid", 0.06), ("warm_rain", 0.05)],
    "urgency": "self-care",
    "duration": "Viral type: 7-14 days (most contagious in the first 3-5 days)",
    "medicines": [
        ("Lubricating eye drops", "1-2 drops, 4-6 times a day", "Comfort and washing out irritants"),
        ("Cold compress", "Over closed eyelids, several times daily", "Soothes burning"),
        ("Antibiotic drops", "Only if the discharge is pus-like and a pharmacist/doctor agrees", "Bacterial conjunctivitis"),
    ],
    "home_care": [
        "Do not share towels, pillows or eye makeup",
        "Wash hands frequently - it spreads by touch",
        "No contact lenses until fully healed",
        "Clean lids gently with cooled boiled water",
    ],
    "red_flags": [
        "Real eye pain (not just itch)",
        "Marked light sensitivity or any vision change - see an eye doctor",
        "Symptoms beyond 7 days",
    ],
}

DISEASES["Heat Exhaustion"] = {
    "about": "Overheating after sun/heat exposure - heavy sweating, dizziness and weakness.",
    "symptoms": {"Heavy sweating": 3, "Excessive thirst": 3, "Dizziness": 3,
                 "Fatigue or weakness": 3, "Pale, cool or clammy skin": 3,
                 "Muscle cramps": 2, "Nausea": 2, "Headache": 2},
    "seasons": ["summer"],
    "weather_boosts": [("heat", 0.15)],
    "urgency": "see-doctor",
    "duration": "Improves within 30-60 minutes of cooling",
    "medicines": [
        ("ORS / electrolyte drink", "Sip steadily - this IS the treatment", "Replaces salts lost in sweat"),
        ("No tablets needed", "Cooling and fluids fix it; paracetamol does not help overheating", "Note"),
    ],
    "home_care": [
        "Move to shade or air-conditioning NOW",
        "Lie down with legs slightly raised; loosen clothing",
        "Cool wet cloths on the skin, fan continuously",
        "Sip fluids steadily - do not gulp ice-cold drinks",
    ],
    "red_flags": [
        "Confusion, fainting, or a seizure - HEATSTROKE, call emergency services",
        "Sweating stops and the skin turns hot and dry - emergency",
        "Not improving after 1 hour of cooling - hospital",
    ],
}

DISEASES["Dehydration"] = {
    "about": "Not enough fluid in the body - thirst, dry mouth and dark, scanty urine.",
    "symptoms": {"Excessive thirst": 3, "Dry mouth": 3, "Dark or little urine": 3,
                 "Dizziness": 2, "Fatigue or weakness": 2, "Headache": 1, "Muscle cramps": 1},
    "seasons": ["summer"],
    "weather_boosts": [("heat", 0.12)],
    "urgency": "self-care",
    "duration": "Recovers within hours of proper fluid intake",
    "medicines": [
        ("ORS", "About 1-1.5 litres sipped over 2-4 hours", "The actual treatment"),
        ("Water", "Keep drinking after the ORS until urine turns pale", "Maintenance"),
    ],
    "home_care": [
        "Stay in a cool place",
        "Alternate water with ORS/electrolyte drinks",
        "Avoid alcohol and excess caffeine - they worsen dehydration",
    ],
    "red_flags": [
        "No urine for more than 8 hours",
        "Extreme drowsiness or a racing heartbeat",
        "Fainting - needs medical assessment",
    ],
}

DISEASES["Fungal Skin Infection"] = {
    "about": "Ringworm and similar fungi thriving in sweat and humidity - itchy ring-shaped patches.",
    "symptoms": {"Ring-shaped scaly rash": 3, "Itchy rash": 3, "Skin peeling or flaking": 2},
    "seasons": ["monsoon"],
    "weather_boosts": [("humid", 0.10)],
    "urgency": "self-care",
    "duration": "2-4 weeks of consistent treatment",
    "medicines": [
        ("Clotrimazole 1% cream", "Thin layer twice daily; continue 1 week AFTER the rash clears", "Kills the fungus"),
        ("Terbinafine 1% cream", "Once or twice daily (alternative)", "Faster option in many cases"),
    ],
    "home_care": [
        "Keep the area clean and DRY - dry thoroughly after every bath",
        "Wear loose cotton clothes; change sweat-soaked clothes promptly",
        "Do not share towels, combs or clothes",
        "Wash socks and clothes in hot water",
    ],
    "red_flags": [
        "Rash still spreading after 2 weeks of cream",
        "Scalp or nail involvement (needs prescription medicine)",
        "Diabetes or a weakened immune system - see a doctor",
    ],
}

DISEASES["Ear Infection (Otitis Media)"] = {
    "about": "Fluid and infection behind the eardrum, often following a cold - especially in children.",
    "symptoms": {"Ear pain": 3, "Reduced hearing": 2, "Ear discharge": 2, "Fever": 1,
                 "Headache": 1, "Blocked nose": 1},
    "seasons": ["winter"],
    "weather_boosts": [("cold", 0.05)],
    "urgency": "see-doctor",
    "duration": "Pain often eases in 2-3 days (many cases are viral)",
    "medicines": [
        ("Paracetamol or Ibuprofen", "Standard weight/age-appropriate doses", "Ear pain and fever"),
        ("Antibiotic drops / oral antibiotics", "Prescription-only - the doctor looks in the ear first", "If bacterial"),
    ],
    "home_care": [
        "Warm (not hot) compress against the ear",
        "Keep the ear dry - cotton ball with petroleum jelly while bathing",
        "Never poke the ear with buds or pins",
        "Treat the blocked nose (steam, saline rinse) - the ear drains through it",
    ],
    "red_flags": [
        "Swelling or redness behind the ear - urgent",
        "High fever with severe pain",
        "Dizziness or facial weakness - same-day doctor",
    ],
}

# ---------------------------------------------------------------------------
# PART 1A - FOUR EXTRA CONDITIONS THAT MATTER A LOT IN INDIA (v2.0)
# ---------------------------------------------------------------------------
# Chikungunya, TB, Hepatitis A and Leptospirosis were added because they are
# either India-specific burdens (TB), monsoon companions of dengue
# (chikungunya, leptospirosis) or classic food/water-borne illnesses
# (hepatitis A). Copy any block to add your own condition.

DISEASES["Chikungunya"] = {
    "about": "A mosquito-borne viral fever that circulates together with dengue in the monsoon - famous for severe joint pain in the hands and feet that can linger for weeks.",
    "symptoms": {"Severe joint pain (hands and feet)": 3, "High fever (39°C / 102°F or above)": 3,
                 "Joint pain": 2, "Headache": 2, "Body aches / muscle pain": 2,
                 "Fatigue or weakness": 2, "Nausea": 1, "Itchy rash": 1},
    "seasons": ["monsoon"],
    "weather_boosts": [("warm_rain", 0.10), ("humid", 0.04)],
    "urgency": "see-doctor",
    "duration": "Fever settles in 3-7 days; joint pain can last weeks to months",
    "medicines": [
        ("Paracetamol ONLY", "500-1000 mg every 6 hours (max 3,000 mg/day)", "Fever and joint pain"),
        ("NEVER ibuprofen/aspirin until dengue is ruled out", "Same bleeding-risk rule as dengue - do a blood test first", "Safety warning"),
    ],
    "home_care": [
        "Get tested - chikungunya and dengue look alike early on but are managed differently",
        "Rest the painful joints; warm compresses help",
        "Drink plenty of fluids",
        "Sleep under a mosquito net while feverish so mosquitoes cannot spread it further",
    ],
    "red_flags": [
        "Severe joint pain that prevents walking or self-care",
        "Bleeding gums or nose (think dengue instead - retest)",
        "Confusion or persistent vomiting",
    ],
}

DISEASES["Tuberculosis (TB)"] = {
    "about": "A bacterial lung infection that grows slowly - the classic pattern is a cough lasting more than 2 weeks with night sweats and weight loss. India carries the world's highest TB burden, so this pattern always deserves a test.",
    "symptoms": {"Cough with phlegm": 3, "Coughing up blood": 3, "Night sweats": 3,
                 "Unintentional weight loss": 3, "Fatigue or weakness": 2, "Fever": 2,
                 "Loss of appetite": 2, "Chest pain when breathing": 1},
    "seasons": [],
    "weather_boosts": [],
    "urgency": "urgent",
    "duration": "Months if untreated; contagious until treated",
    "medicines": [
        ("Anti-TB medicines (prescription)", "A 4-6 month course chosen by a doctor after confirmed tests - free at government DOTS centres in India", "The cure"),
        ("Paracetamol", "Standard doses while awaiting evaluation", "Fever and discomfort"),
    ],
    "home_care": [
        "A cough beyond 2 weeks needs a sputum test and chest X-ray - do not wait it out",
        "Cover your mouth and ventilate the room - TB spreads through the air",
        "Eat well; treatment works only if completed fully",
    ],
    "red_flags": [
        "Coughing up blood",
        "A cough lasting more than 2 weeks with weight loss or night sweats",
        "Breathlessness or severe weight loss",
    ],
}

DISEASES["Hepatitis A"] = {
    "about": "A liver infection from contaminated food or water - fever and vomiting followed by yellow eyes, dark urine and pale stools.",
    "symptoms": {"Yellow eyes or skin (jaundice)": 3, "Dark or little urine": 2,
                 "Pale (clay-coloured) stools": 2, "Fatigue or weakness": 2,
                 "Nausea": 2, "Loss of appetite": 2, "Fever": 1,
                 "Abdominal pain or cramps": 1, "Vomiting": 1},
    "seasons": ["monsoon", "summer"],
    "weather_boosts": [("warm_rain", 0.06)],
    "urgency": "see-doctor",
    "duration": "Most recover fully in 2-6 weeks",
    "medicines": [
        ("No specific antiviral", "Rest, fluids and avoiding liver-stressing medicines and alcohol IS the treatment", "Supportive care"),
        ("Paracetamol (carefully)", "Only if fever is high, at the lowest effective dose - ask a doctor first when the liver is involved", "Fever"),
    ],
    "home_care": [
        "Get an LFT blood test to confirm and grade it",
        "Small frequent meals; avoid oily food and alcohol completely",
        "Strict hand hygiene - it spreads through food and water",
        "Hepatitis A vaccine protects close contacts - ask a doctor",
    ],
    "red_flags": [
        "Drowsiness, confusion or a bleeding tendency - hospital immediately",
        "No urine for many hours, or rapidly worsening yellowing",
        "Vomiting everything for more than a day",
    ],
}

DISEASES["Leptospirosis"] = {
    "about": "A bacterial infection spread through water contaminated by animal urine - a monsoon illness classically linked to wading through flooded streets, well known in Mumbai and coastal cities.",
    "symptoms": {"High fever (39°C / 102°F or above)": 3, "Chills or shivering": 3,
                 "Headache": 2, "Body aches / muscle pain": 2, "Red eyes": 2,
                 "Nausea": 2, "Yellow eyes or skin (jaundice)": 2, "Vomiting": 1,
                 "Fatigue or weakness": 1, "Muscle cramps": 1},
    "seasons": ["monsoon"],
    "weather_boosts": [("warm_rain", 0.12)],
    "urgency": "urgent",
    "duration": "Recovers in 1-2 weeks with early antibiotics",
    "medicines": [
        ("Antibiotics (prescription)", "Commonly doxycycline or a penicillin-family medicine - most effective when started EARLY", "The cure"),
        ("Paracetamol", "Standard doses meanwhile", "Fever and aches"),
    ],
    "home_care": [
        "Tell the doctor about flood-water contact - it changes what they test for",
        "Avoid wading through floodwater; cover cuts and wounds if you must",
        "Fluids and rest",
    ],
    "red_flags": [
        "Yellow eyes with very little urine - kidney/liver failure risk, hospital NOW",
        "Coughing up blood or breathlessness (severe leptospirosis)",
        "Confusion or seizures",
    ],
}

# PART 1C - BACK PAIN & NERVE PROBLEMS (v3.0, built for the chat mode)
# ---------------------------------------------------------------------------
# Back pain is one of the most common reasons people open a symptom checker,
# and it is exactly where checkbox-only apps fail: patients describe the
# STORY ("deadlift without a belt", "pain shoots to my calf") rather than
# ticking boxes. These three conditions are separated by nerve involvement:
#   - Muscle strain: pain stays LOCAL, no leg symptoms.
#   - Sciatica (nerve root irritation): pain RADIATES below the knee,
#     often with numbness/tingling.
#   - Disc herniation: the mechanical cause of sciatica - same leg symptoms
#     PLUS mechanical clues (worse sitting/bending/coughing, lifting injury).
# Only a doctor + MRI can truly separate the last two - the app's job is to
# suggest the right tests and the right urgency, not to be an MRI.

DISEASES["Lower Back Muscle Strain"] = {
    "about": "Torn or overstretched back muscles or ligaments - the classic "
             "'pulled back' after lifting, twisting or sudden movement. Pain stays "
             "in the lower back; the legs feel normal.",
    "symptoms": {"Lower back pain": 3, "Back pain that started after lifting or twisting": 3,
                 "Back stiffness in the morning": 2, "Body aches / muscle pain": 1,
                 "Muscle cramps": 1},
    "seasons": [], "weather_boosts": [],
    "urgency": "self-care",
    "duration": "Settles in a few days to 2 weeks with rest and simple care",
    "medicines": [
        ("Paracetamol", "500-1000 mg every 6 hours as needed (max 3,000 mg/day)", "Back pain"),
        ("Ibuprofen (with food)", "200-400 mg every 6-8 hours for up to 3-5 days - avoid with "
         "kidney trouble, gastritis or dengue-like fever", "Pain and inflammation"),
        ("Topical pain gel (diclofenac)", "Apply 3-4 times a day on the painful area", "Local relief"),
    ],
    "home_care": [
        "Keep gently moving within comfort - total bed rest slows healing",
        "Avoid heavy lifting and rounding your back for 1-2 weeks",
        "Ice for the first 48 hours (15 min at a time), then a warm compress",
        "Sleep on your side with a pillow between the knees",
    ],
    "red_flags": [
        "Pain shooting below the knee, or numbness/weakness of the leg or foot - that is "
        "no longer a simple strain; get examined",
        "Loss of bladder or bowel control - emergency, go now",
        "Fever with back pain, or pain that steadily worsens over 2 weeks",
    ],
}

DISEASES["Sciatica (Lumbar Radiculopathy)"] = {
    "about": "Irritation of the sciatic nerve, usually where a disc or tight muscle "
             "presses on it in the lower spine. The signature is pain that shoots from "
             "the back or buttock down BELOW the knee, often with tingling.",
    "symptoms": {"Pain shooting down the leg (below the knee)": 3,
                 "Lower back pain": 2, "Numbness or tingling in the leg or foot": 3,
                 "Weakness in the leg or foot": 2,
                 "Back pain worse when sitting or bending forward": 2,
                 "Back pain worse when coughing or sneezing": 2},
    "seasons": [], "weather_boosts": [],
    "urgency": "see-doctor",
    "duration": "Most improve in 4-6 weeks with the right care; physiotherapy speeds it up",
    "medicines": [
        ("Paracetamol", "500-1000 mg every 6 hours as needed (max 3,000 mg/day)", "Pain relief"),
        ("Ibuprofen (with food)", "200-400 mg every 6-8 hours short-term - see a doctor if "
         "you have gastritis, kidney issues or are on blood thinners", "Nerve inflammation"),
        ("Pregabalin / gabapentin (prescription)", "Nerve-type pain medicines a doctor may "
         "prescribe if simple painkillers do not control the pain", "Nerve pain"),
    ],
    "home_care": [
        "Stay active as tolerated - prolonged bed rest makes sciatica worse",
        "Avoid prolonged sitting and forward bending while pain shoots down the leg",
        "A physiotherapist can teach 'nerve gliding' and core exercises - this is the "
        "single most effective treatment",
        "Alternate warm compress and gentle walking",
    ],
    "red_flags": [
        "Numbness between the legs, or loss of bladder/bowel control (cauda equina "
        "syndrome) - EMERGENCY, go to hospital immediately",
        "Leg weakness that worsens (foot dragging, cannot stand on toes)",
        "Pain with fever, weight loss, or after a major fall - needs same-day assessment",
    ],
}

DISEASES["Lumbar Disc Herniation (Slipped Disc)"] = {
    "about": "The soft cushion between two spine bones (disc) bulges out and presses "
             "on a nerve - classically after heavy lifting with a rounded back (e.g. "
             "deadlifts without a belt). It is the most common cause of sciatica in "
             "young adults; an MRI is what shows it.",
    "symptoms": {"Back pain that started after lifting or twisting": 3,
                 "Pain shooting down the leg (below the knee)": 3,
                 "Lower back pain": 2, "Numbness or tingling in the leg or foot": 2,
                 "Weakness in the leg or foot": 2,
                 "Back pain worse when sitting or bending forward": 3,
                 "Back pain worse when coughing or sneezing": 3,
                 "Back stiffness in the morning": 1},
    "seasons": [], "weather_boosts": [],
    "urgency": "see-doctor",
    "duration": "6-12 weeks for most; only a minority need surgery",
    "medicines": [
        ("Paracetamol + short NSAID course (with food)", "As advised by a doctor - e.g. "
         "paracetamol 650 mg up to TDS plus ibuprofen short-term", "Pain and inflammation"),
        ("Muscle relaxant (prescription, e.g. thiocolchicoside)", "Short courses help the "
         "painful muscle spasm around the spine", "Muscle spasm"),
        ("Pregabalin / gabapentin (prescription)", "For the shooting, electric nerve pain", "Nerve pain"),
    ],
    "home_care": [
        "Do NOT deadlift or load your back until cleared - re-injury pushes the disc further out",
        "See a physiotherapist early: McKenzie extension exercises and core work are the "
        "mainstay of treatment",
        "Avoid prolonged sitting; use a lumbar support and stand up every 20-30 minutes",
        "Keep walking - movement heals discs better than bed rest",
        "Use a lifting belt for heavy sessions once you return to training (after recovery)",
    ],
    "red_flags": [
        "Numbness in the groin/inner thighs, or loss of bladder/bowel control - EMERGENCY "
        "(cauda equina syndrome), go to hospital NOW",
        "Progressive leg weakness or a foot that starts dragging",
        "Pain so severe you cannot sit, sleep or walk even with medicines",
    ],
}

# ---------------------------------------------------------------------------
# PART 1B - TESTS, SCANS & LAB FINDINGS (v2.0 upgrade pack)
# ---------------------------------------------------------------------------
# TYPICAL_TESTS: for every condition, what a doctor in India would typically
# order, and why. Shown in "Step 4 - Suggested tests & scans", and also used
# to generate the example patients that train the learning engine.
#
# CONFIRMING_FINDINGS: plain-language lab results that push a condition's
# score up or down (e.g. the word "ns1" in the text the user types). Points
# are added to the hybrid score - positive = supports, negative = argues
# against. A longer matched phrase automatically overrides a shorter one it
# contains (so "ns1 negative" wins over plain "ns1").
#
# INDIA_BRAND_EXAMPLES / COMMON_LAB_FINDINGS: small helpers for the UI.
# ---------------------------------------------------------------------------

TYPICAL_TESTS = {
    "Common Cold": [("No test needed", "Diagnosed from the symptom picture alone")],
    "Influenza (Flu)": [
        ("Rapid influenza antigen test (nasal or throat swab)", "Confirms flu within about 15 minutes"),
        ("CBC (complete blood count)", "Checks how strong the infection looks if fever runs long"),
    ],
    "COVID-like Viral Illness": [
        ("COVID rapid antigen test", "Confirms the cause quickly, even at home"),
        ("RT-PCR (if rapid test is negative but suspicion stays high)", "The more sensitive laboratory confirmatory test"),
    ],
    "Allergic Rhinitis (Hay Fever)": [
        ("Usually no test needed", "Diagnosed from symptoms and their pattern"),
        ("Serum total IgE or allergy panel (blood)", "Only if identifying the trigger would change management"),
    ],
    "Sinusitis": [
        ("Usually no test needed", "Clinical diagnosis after examination"),
        ("CT scan of the sinuses", "Only for recurrent or not-settling cases"),
    ],
    "Acute Bronchitis": [
        ("Usually no test needed", "Clinical diagnosis"),
        ("Chest X-ray", "To rule out pneumonia if fever or breathlessness appears"),
    ],
    "Asthma Flare-Up": [
        ("Peak flow meter reading", "Measures how narrowed the airways are - instant at any clinic"),
        ("Pulse oximetry (SpO2)", "Checks blood oxygen during a flare-up"),
        ("Spirometry (after recovery)", "Confirms and grades asthma for long-term treatment"),
    ],
    "Pneumonia": [
        ("Chest X-ray", "Shows the infected patch of lung - the key test"),
        ("CBC and CRP (blood)", "Show how active the infection is"),
        ("Pulse oximetry (SpO2)", "Low oxygen means urgent hospital-level care"),
    ],
    "Migraine": [
        ("No test needed for typical attacks", "Diagnosed from the story and examination"),
        ("MRI brain (only with red-flag features)", "For sudden worst-ever or neurological headaches"),
    ],
    "Tension Headache": [("No test needed", "Diagnosed from the typical pressure-like pattern")],
    "Viral Fever (general)": [
        ("CBC (complete blood count)", "A simple first blood test when fever lasts beyond 3 days"),
        ("Dengue NS1 and malaria rapid test", "Standard in India whenever fever persists - rules out the dangerous ones"),
    ],
    "Strep Throat (Bacterial)": [
        ("Rapid strep test or throat swab culture", "Confirms strep before antibiotics"),
    ],
    "Gastroenteritis (Stomach Flu / Food Poisoning)": [
        ("Usually no test needed", "Settles on its own within 1-3 days"),
        ("Stool routine and culture", "If there is blood in stool, high fever, or it lasts beyond 3 days"),
    ],
    "Acid Reflux (GERD)": [
        ("Usually no test needed", "Treated based on symptoms"),
        ("Upper GI endoscopy", "Only for alarm features: difficulty swallowing, weight loss, vomiting blood"),
    ],
    "Urinary Tract Infection (UTI)": [
        ("Urine routine and microscopy", "Shows pus cells and bacteria - quick and inexpensive"),
        ("Urine culture and sensitivity", "Grows the bacteria and tells which antibiotic will work"),
    ],
    "Dengue Fever": [
        ("Dengue NS1 antigen (blood)", "Most accurate in the first 5 days of fever"),
        ("Dengue IgM or IgG (blood)", "Used after day 5 of fever"),
        ("CBC with platelet count (daily while feverish)", "Falling platelets are the danger signal to watch"),
    ],
    "Typhoid Fever": [
        ("Blood culture", "The most reliable test - best in the first week of fever"),
        ("Widal test (blood)", "Widely available in India; needs careful interpretation by a doctor"),
        ("Typhoid IgM rapid test", "A quick supporting test"),
        ("CBC (complete blood count)", "Supporting picture of infection"),
    ],
    "Malaria": [
        ("Malaria rapid diagnostic test (RDT)", "Finger-prick result in 15-20 minutes - do it the same day"),
        ("Peripheral blood smear (microscopy)", "Shows the parasite and its type - the classic confirmatory test"),
        ("CBC (complete blood count)", "Checks for anemia"),
    ],
    "Chickenpox": [
        ("Usually no test needed", "The blister rash is characteristic"),
        ("CBC (complete blood count)", "If fever is high or the patient looks very unwell"),
    ],
    "Conjunctivitis (Eye Flu)": [
        ("No test usually needed", "Diagnosed by examining the eye"),
    ],
    "Heat Exhaustion": [
        ("No test needed in typical cases", "Cooling and ORS fix it"),
        ("Serum electrolytes (sodium and potassium)", "If confusion or cramps do not settle"),
    ],
    "Dehydration": [
        ("Usually no test needed", "Urine colour and frequency tell the story"),
        ("Serum electrolytes and creatinine", "Only if severe or not improving"),
    ],
    "Fungal Skin Infection": [
        ("Skin scraping and KOH mount", "Confirms the fungus if the rash is not typical"),
    ],
    "Ear Infection (Otitis Media)": [
        ("No test usually needed", "The doctor looks into the ear with an otoscope"),
    ],
    "Chikungunya": [
        ("Chikungunya IgM rapid test or RT-PCR (blood)", "Distinguishes it from dengue - important because the two circulate together"),
        ("CBC (complete blood count)", "A supporting test"),
    ],
    "Tuberculosis (TB)": [
        ("Sputum AFB smear (2 samples)", "The standard first test for a cough beyond 2 weeks"),
        ("Chest X-ray", "Shows typical TB changes"),
        ("CBNAAT or GeneXpert (sputum)", "A rapid molecular test - also detects drug resistance"),
    ],
    "Hepatitis A": [
        ("Liver function tests - SGPT and bilirubin (blood)", "Shows liver inflammation and jaundice level"),
        ("Hepatitis A IgM (blood)", "Confirms the specific virus"),
        ("CBC (complete blood count)", "A supporting test"),
    ],
    "Leptospirosis": [
        ("Leptospira IgM or ELISA (blood)", "The specific test when flood-water exposure meets fever"),
        ("CBC (complete blood count)", "Platelet and white cell picture"),
        ("Serum creatinine and LFT (blood)", "Checks kidney and liver involvement"),
    ],
    # v3.0 back-pain trio:
    "Lower Back Muscle Strain": [
        ("Usually no scan needed", "The story (lifting + local pain, normal legs) tells the doctor"),
        ("X-ray of the lumbar spine (only if fracture is suspected)", "After a hard fall or in older adults"),
    ],
    "Sciatica (Lumbar Radiculopathy)": [
        ("MRI lumbar spine (with nerve compression signs)", "Shows exactly which disc and nerve root is involved"),
        ("Straight-leg-raise test (clinical)", "Done in the clinic - a positive test points to nerve irritation"),
        ("X-ray lumbar spine (optional first look)", "Rules out bone problems; MRI is the definitive scan"),
    ],
    "Lumbar Disc Herniation (Slipped Disc)": [
        ("MRI lumbar spine", "THE test - shows the bulging disc and which nerve it is pressing"),
        ("Straight-leg-raise test (clinical)", "Positive when the nerve is irritated - done at examination"),
        ("X-ray lumbar spine", "A cheap first look at the bones; cannot show discs (MRI can)"),
    ],
}

# Merge the test knowledge into the main knowledge base, one loop at startup.
for _name, _tests in TYPICAL_TESTS.items():
    if _name in DISEASES:
        DISEASES[_name]["typical_tests"] = _tests

CONFIRMING_FINDINGS = {
    "Dengue Fever": [
        {"phrase": "ns1 negative", "points": -0.18, "label": "NS1 antigen negative"},
        {"phrase": "ns1", "points": 0.22, "label": "NS1 antigen positive"},
        {"phrase": "dengue igm", "points": 0.15, "label": "Dengue IgM positive"},
        {"phrase": "low platelet", "points": 0.18, "label": "Low platelet count"},
        {"phrase": "thrombocytopenia", "points": 0.18, "label": "Thrombocytopenia (low platelets)"},
        {"phrase": "platelets normal", "points": -0.10, "label": "Platelet count normal"},
        {"phrase": "platelet count normal", "points": -0.10, "label": "Platelet count normal"},
        {"phrase": "all tests normal", "points": -0.08, "label": "All tests normal"},
        {"phrase": "everything normal", "points": -0.08, "label": "All tests normal"},
    ],
    "Malaria": [
        {"phrase": "malaria negative", "points": -0.20, "label": "Malaria test negative"},
        {"phrase": "malaria", "points": 0.25, "label": "Malaria test positive"},
        {"phrase": "vivax", "points": 0.25, "label": "P. vivax seen on smear"},
        {"phrase": "falciparum", "points": 0.25, "label": "P. falciparum seen on smear"},
        {"phrase": "parasite", "points": 0.20, "label": "Parasites seen in blood smear"},
        {"phrase": "all tests normal", "points": -0.08, "label": "All tests normal"},
    ],
    "Typhoid Fever": [
        {"phrase": "widal negative", "points": -0.12, "label": "Widal test negative"},
        {"phrase": "widal", "points": 0.18, "label": "Widal test positive"},
        {"phrase": "typhoid igm", "points": 0.20, "label": "Typhoid IgM positive"},
        {"phrase": "blood culture positive", "points": 0.22, "label": "Blood culture positive"},
        {"phrase": "s. typhi", "points": 0.22, "label": "S. typhi grown in culture"},
        {"phrase": "all tests normal", "points": -0.06, "label": "All tests normal"},
    ],
    "Chikungunya": [
        {"phrase": "chikungunya", "points": 0.25, "label": "Chikungunya test positive"},
    ],
    "Tuberculosis (TB)": [
        {"phrase": "afb negative", "points": -0.15, "label": "Sputum AFB negative"},
        {"phrase": "afb", "points": 0.25, "label": "Sputum AFB positive"},
        {"phrase": "genexpert", "points": 0.25, "label": "GeneXpert positive"},
        {"phrase": "cbnaat", "points": 0.25, "label": "CBNAAT positive"},
        {"phrase": "sputum positive", "points": 0.22, "label": "Sputum test positive"},
        {"phrase": "cavit", "points": 0.15, "label": "Cavity seen on chest X-ray"},
    ],
    "Pneumonia": [
        {"phrase": "chest x-ray normal", "points": -0.12, "label": "Chest X-ray normal"},
        {"phrase": "x-ray clear", "points": -0.12, "label": "Chest X-ray clear"},
        {"phrase": "pneumonia", "points": 0.25, "label": "Pneumonia seen on chest X-ray"},
        {"phrase": "consolidation", "points": 0.22, "label": "Consolidation on X-ray"},
        {"phrase": "infiltrate", "points": 0.20, "label": "Infiltrate on X-ray"},
        {"phrase": "high crp", "points": 0.10, "label": "High CRP"},
    ],
    "Urinary Tract Infection (UTI)": [
        {"phrase": "urine culture negative", "points": -0.15, "label": "Urine culture negative"},
        {"phrase": "pus cells", "points": 0.20, "label": "Pus cells in urine"},
        {"phrase": "urine culture positive", "points": 0.22, "label": "Urine culture positive"},
        {"phrase": "bacteria in urine", "points": 0.18, "label": "Bacteria in urine"},
    ],
    "Strep Throat (Bacterial)": [
        {"phrase": "strep negative", "points": -0.15, "label": "Strep test negative"},
        {"phrase": "strep", "points": 0.25, "label": "Strep test positive"},
        {"phrase": "throat culture", "points": 0.22, "label": "Throat culture positive"},
    ],
    "Influenza (Flu)": [
        {"phrase": "influenza negative", "points": -0.15, "label": "Flu test negative"},
        {"phrase": "influenza positive", "points": 0.25, "label": "Flu test positive"},
        {"phrase": "flu a", "points": 0.22, "label": "Influenza A positive"},
        {"phrase": "flu b", "points": 0.22, "label": "Influenza B positive"},
    ],
    "COVID-like Viral Illness": [
        {"phrase": "covid negative", "points": -0.15, "label": "COVID test negative"},
        {"phrase": "covid", "points": 0.25, "label": "COVID test positive"},
        {"phrase": "rtpcr positive", "points": 0.25, "label": "RT-PCR positive"},
    ],
    "Hepatitis A": [
        {"phrase": "hepatitis a", "points": 0.22, "label": "Hepatitis A IgM positive"},
        {"phrase": "sgpt", "points": 0.20, "label": "High SGPT/ALT (liver enzymes)"},
        {"phrase": "sgot", "points": 0.15, "label": "High SGOT/AST (liver enzymes)"},
        {"phrase": "bilirubin", "points": 0.18, "label": "High bilirubin (jaundice)"},
        {"phrase": "lft", "points": 0.12, "label": "Abnormal liver function tests"},
    ],
    "Leptospirosis": [
        {"phrase": "lepto", "points": 0.25, "label": "Leptospira test positive"},
        {"phrase": "sgpt", "points": 0.10, "label": "High SGPT/ALT (liver enzymes)"},
        {"phrase": "bilirubin", "points": 0.10, "label": "High bilirubin (jaundice)"},
        {"phrase": "creatinine", "points": 0.12, "label": "High creatinine (kidney stress)"},
    ],
    "Asthma Flare-Up": [
        {"phrase": "peak flow", "points": 0.15, "label": "Peak flow reduced"},
        {"phrase": "spirometry", "points": 0.15, "label": "Spirometry shows obstruction"},
        {"phrase": "eosinophil", "points": 0.06, "label": "High eosinophil count"},
    ],
    "Allergic Rhinitis (Hay Fever)": [
        {"phrase": "eosinophil", "points": 0.08, "label": "High eosinophil count"},
        {"phrase": "ige", "points": 0.15, "label": "High IgE (allergy marker)"},
    ],
    "Gastroenteritis (Stomach Flu / Food Poisoning)": [
        {"phrase": "stool culture", "points": 0.15, "label": "Stool culture positive"},
        {"phrase": "entamoeba", "points": 0.20, "label": "Entamoeba seen in stool"},
        {"phrase": "protozoa", "points": 0.18, "label": "Protozoa seen in stool"},
    ],
    "Viral Fever (general)": [
        {"phrase": "all tests normal", "points": 0.05, "label": "All tests normal (fits a simple viral fever)"},
        {"phrase": "everything normal", "points": 0.05, "label": "All tests normal"},
    ],
    "Common Cold": [
        {"phrase": "all tests normal", "points": 0.04, "label": "All tests normal"},
    ],
    # v3.0 back-pain trio:
    "Lower Back Muscle Strain": [
        {"phrase": "mri normal", "points": 0.12, "label": "MRI normal - fits a simple strain"},
        {"phrase": "x-ray normal", "points": 0.06, "label": "X-ray normal"},
        {"phrase": "no disc bulge", "points": 0.10, "label": "No disc bulge on MRI"},
        {"phrase": "disc bulge", "points": -0.10, "label": "Disc bulge seen on imaging"},
    ],
    "Sciatica (Lumbar Radiculopathy)": [
        {"phrase": "l5", "points": 0.12, "label": "L5 nerve root involved"},
        {"phrase": "s1", "points": 0.12, "label": "S1 nerve root involved"},
        {"phrase": "disc bulge", "points": 0.10, "label": "Disc bulge seen on imaging"},
        {"phrase": "straight leg raise positive", "points": 0.15, "label": "Straight-leg-raise test positive"},
        {"phrase": "slr positive", "points": 0.15, "label": "SLR test positive"},
    ],
    "Lumbar Disc Herniation (Slipped Disc)": [
        {"phrase": "disc bulge", "points": 0.22, "label": "Disc bulge seen on imaging"},
        {"phrase": "disc herniation", "points": 0.25, "label": "Disc herniation seen on imaging"},
        {"phrase": "disc prolapse", "points": 0.25, "label": "Disc prolapse seen on imaging"},
        {"phrase": "l4 l5", "points": 0.18, "label": "L4-L5 disc involved"},
        {"phrase": "l5 s1", "points": 0.18, "label": "L5-S1 disc involved"},
        {"phrase": "mri normal", "points": -0.15, "label": "MRI normal - a bulging disc is unlikely"},
        {"phrase": "no disc bulge", "points": -0.15, "label": "No disc bulge on MRI"},
    ],
}

# Quick-pick chips shown under "refine with lab findings".
COMMON_LAB_FINDINGS = [
    "Dengue NS1 antigen positive",
    "Dengue IgM positive",
    "Low platelet count (thrombocytopenia)",
    "Malaria rapid test positive",
    "Widal test positive",
    "Typhoid IgM positive",
    "Chikungunya IgM positive",
    "COVID test positive",
    "Strep throat test positive",
    "Urine test shows pus cells",
    "Chest X-ray shows pneumonia",
    "Sputum test positive for TB (AFB)",
    "High SGPT / ALT (liver enzymes)",
    "Jaundice - high bilirubin",
    "High eosinophil count",
    "All tests came back normal",
]

# Well-known Indian brand examples for the OTC medicines we mention (many
# other brands exist - the generic "salt" name is what actually matters).
INDIA_BRAND_EXAMPLES = {
    "paracetamol": "Crocin / Dolo 650 / Calpol (paracetamol)",
    "cetirizine": "Cetzine / Okacet (cetirizine)",
    "loratadine": "Lorfast (loratadine)",
    "ibuprofen": "Brufen (ibuprofen)",
    "omeprazole": "Omez (omeprazole)",
    "ors": "Electral / Enerzal (ORS)",
    "clotrimazole": "Candid cream (clotrimazole)",
    "terbinafine": "Sebifin cream (terbinafine)",
    "salbutamol": "Asthalin inhaler (salbutamol)",
    "calcium carbonate": "Digene / Gelusil (antacid)",
    "loperamide": "Imodium / Eldoper (loperamide)",
}


def india_brand_caption(medicines):
    """One caption line listing Indian brand examples for a disease's medicines."""
    hits = []
    for med, _, _ in medicines:
        low = med.lower()
        for key, brands in INDIA_BRAND_EXAMPLES.items():
            if key in low and brands not in hits:
                hits.append(brands)
    if not hits:
        return None
    return ("Common Indian brand examples: " + "; ".join(hits) +
            ". Many other brands exist - always match the generic (salt) name.")


# ---------------------------------------------------------------------------
# PART 2 - THE FREE INTERNET APIS (no account, no API key, no payment)
# ---------------------------------------------------------------------------
# 60-second explanation of a web API:
#   1. Our program opens a URL (just like a browser would) with requests.get(...)
#   2. The server replies with JSON - text that looks like {"temperature_2m": 33.4}
#   3. .json() converts that text into a normal Python dictionary
# Every call is wrapped in try/except, so if the internet is down the app
# keeps working - it just runs without weather/hospital data.
# The @st.cache_data decorators cache each answer for a few minutes, so
# clicking around the app does NOT re-download the same data again and again.
# ---------------------------------------------------------------------------

import math
import time

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HTTP_HEADERS = {"User-Agent": "drSaab/2.2 (free educational symptom-checker)"}

# WMO weather codes that Open-Meteo uses, translated to plain English.
WMO_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Light snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Light rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Light snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with light hail", 99: "Thunderstorm with heavy hail",
}


def describe_weather_code(code):
    return WMO_WEATHER_CODES.get(code, f"Unusual conditions (code {code})")


@st.cache_data(ttl=600, show_spinner=False)
def get_city_matches(city_name):
    """Turn a city name into coordinates using Open-Meteo's free geocoding API."""
    try:
        r = requests.get(
            GEOCODING_URL,
            params={"name": city_name, "count": 5, "language": "en", "format": "json"},
            headers=HTTP_HEADERS, timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        return [
            {"name": m["name"], "region": m.get("admin1") or "",
             "country": m.get("country") or "", "lat": m["latitude"], "lon": m["longitude"]}
            for m in results
        ]
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def get_current_weather(lat, lon):
    """Live temperature, humidity, rain, wind and the city's local time."""
    try:
        r = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "current": ("temperature_2m,relative_humidity_2m,apparent_temperature,"
                            "precipitation,weather_code,wind_speed_10m"),
            },
            headers=HTTP_HEADERS, timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        cur = data.get("current") or {}
        return {
            "temp": cur.get("temperature_2m"),
            "feels": cur.get("apparent_temperature"),
            "humidity": cur.get("relative_humidity_2m"),
            "precip": cur.get("precipitation") or 0.0,
            "code": cur.get("weather_code"),
            "wind": cur.get("wind_speed_10m"),
            "local_time": cur.get("time"),  # e.g. "2026-11-13T14:30"
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_air_quality(lat, lon):
    """Live PM2.5 and US AQI (not available for every location - that's ok)."""
    try:
        r = requests.get(
            AIR_QUALITY_URL,
            params={"latitude": lat, "longitude": lon, "current": "pm2_5,us_aqi"},
            headers=HTTP_HEADERS, timeout=15,
        )
        r.raise_for_status()
        cur = r.json().get("current") or {}
        return {"pm25": cur.get("pm2_5"), "aqi": cur.get("us_aqi")}
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in kilometres (great-circle formula)."""
    r_earth = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lam / 2) ** 2
    return r_earth * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _overpass_places(lat, lon, radius_m=8000):
    """PRIMARY hospital finder: the Overpass API queries raw OpenStreetMap data
    and can filter by tag (amenity = hospital / clinic / doctors) inside a circle."""
    query = (
        "[out:json][timeout:25];"
        f'(nwr["amenity"~"^(hospital|clinic|doctors)$"](around:{radius_m},{lat},{lon}););'
        "out center tags 40;"
    )
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HTTP_HEADERS, timeout=40)
            if r.status_code != 200:
                continue
            places = []
            for el in r.json().get("elements", []):
                tags = el.get("tags", {})
                plat = el.get("lat") or (el.get("center") or {}).get("lat")
                plon = el.get("lon") or (el.get("center") or {}).get("lon")
                if plat is None or plon is None:
                    continue
                category = {"hospital": "Hospital", "clinic": "Clinic",
                            "doctors": "Doctor's office"}.get(tags.get("amenity", ""), "Health facility")
                addr = ", ".join(p for p in [tags.get("addr:housenumber"), tags.get("addr:street"),
                                              tags.get("addr:suburb"), tags.get("addr:city")] if p)
                places.append({
                    "name": tags.get("name") or "Unnamed health facility",
                    "category": category,
                    "lat": plat,
                    "lon": plon,
                    "distance_km": haversine_km(lat, lon, plat, plon),
                    "address": addr,
                    "maps_url": f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}",
                })
            if places:
                return places, "OpenStreetMap Overpass API"
        except Exception:
            continue
    return [], None


def _nominatim_places(lat, lon):
    """FALLBACK hospital finder for when Overpass is unreachable or slow.
    Nominatim is OpenStreetMap's geocoder - free, but it asks for at most
    1 request per second, so we politely sleep between the two searches."""
    d = 0.08  # box of about +/- 8 km around the user
    viewbox = f"{lon - d},{lat + d},{lon + d},{lat - d}"
    found, seen = [], set()
    for word in ("hospital", "clinic"):
        try:
            r = requests.get(
                NOMINATIM_URL,
                params={"q": word, "format": "jsonv2", "limit": 15,
                        "bounded": "1", "viewbox": viewbox},
                headers=HTTP_HEADERS, timeout=25,
            )
            r.raise_for_status()
            for item in r.json():
                try:
                    plat, plon = float(item["lat"]), float(item["lon"])
                except (KeyError, ValueError, TypeError):
                    continue
                key = (item.get("name") or "", round(plat, 3), round(plon, 3))
                if key in seen:
                    continue
                seen.add(key)
                display = item.get("display_name", "")
                address = ", ".join(display.split(",")[1:4]).strip() if display else ""
                found.append({
                    "name": item.get("name") or word.title(),
                    "category": word.title(),
                    "lat": plat,
                    "lon": plon,
                    "distance_km": haversine_km(lat, lon, plat, plon),
                    "address": address,
                    "maps_url": f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}",
                })
        except Exception:
            continue
        time.sleep(1.1)  # respect the 1-request-per-second fair-use rule
    return found, "OpenStreetMap Nominatim (fallback)"


@st.cache_data(ttl=900, show_spinner=False)
def get_nearby_healthcare(lat, lon):
    """Try Overpass first; if it fails, fall back to Nominatim.
    Returns (sorted list of places, source label used)."""
    places, source = _overpass_places(lat, lon)
    if not places:
        places, source = _nominatim_places(lat, lon)
    places.sort(key=lambda p: p["distance_km"])
    return places, source


# ---------------------------------------------------------------------------
# PART 2B - INTERNET SYMPTOM MATCHING (Wikidata + Wikipedia, v2.2)
# ---------------------------------------------------------------------------
# The knowledge base in PART 1 was written once by reading medical websites -
# it never changes. This part adds a THIRD source of truth, fetched live:
#
#   1. WIKIDATA - the structured database behind Wikipedia. For every disease
#      it stores which symptoms belong to it (property P780). We ask for all
#      of our 28 conditions with ONE SPARQL query - free, no account.
#      If the official endpoint is busy we retry on a university mirror.
#   2. WIKIPEDIA article summaries - the opening sentences of each disease's
#      article usually list its main symptoms in plain English. The REST API
#      gives us exactly that text, one small request per disease.
#
# Everything is saved to internet_knowledge.json next to app.py, so it is
# downloaded ONCE and then re-used instantly from the cache (refreshed after
# a week, or any time with the sidebar button). If the internet is blocked or
# busy, the app simply runs without this layer - nothing else breaks.
#
# At prediction time (PART 5) every candidate disease gets an "internet
# evidence" score: of the symptoms you reported, how many does the live
# online knowledge confirm for that disease? That becomes a third voice in
# the hybrid score next to the rules engine and the learned ML model.
# ---------------------------------------------------------------------------

# Where to ask for structured disease -> symptom links. All are free,
# key-less SPARQL servers built from Wikipedia's medical content. They are
# tried in order and the FIRST one that answers is used, so a busy or
# blocked service never stops the app.
SPARQL_SOURCES = [
    # (source name, endpoint URL, how symptoms are attached to a disease,
    #  optional extra clause - DBpedia also hands us the Wikipedia URL)
    ("Wikidata", "https://query.wikidata.org/sparql",
     "?disease wdt:P780 ?symptom .", ""),
    ("QLever mirror of Wikidata", "https://qlever.cs.uni-freiburg.de/api/wikidata",
     "?disease wdt:P780 ?symptom .", ""),
    ("DBpedia", "https://dbpedia.org/sparql",
     "?disease <http://dbpedia.org/ontology/symptom> ?symptom .",
     "OPTIONAL { ?disease <http://xmlns.com/foaf/0.1/isPrimaryTopicOf> ?url } "),
]
WIKIPEDIA_REST_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
KNOWLEDGE_CACHE_FILE = "internet_knowledge.json"
KNOWLEDGE_MAX_AGE_HOURS = 24 * 7        # re-download once a week

# Wikipedia article for each condition in our knowledge base. (Article names
# must match Wikipedia's exact titles; anything not listed is simply skipped.)
WIKIPEDIA_TITLES = {
    "Common Cold": "Common cold",
    "Influenza (Flu)": "Influenza",
    "COVID-like Viral Illness": "COVID-19",
    "Allergic Rhinitis (Hay Fever)": "Allergic rhinitis",
    "Sinusitis": "Sinusitis",
    "Acute Bronchitis": "Acute bronchitis",
    "Asthma Flare-Up": "Asthma",
    "Pneumonia": "Pneumonia",
    "Migraine": "Migraine",
    "Tension Headache": "Tension headache",
    "Viral Fever (general)": "Influenza-like illness",
    "Strep Throat (Bacterial)": "Streptococcal pharyngitis",
    "Gastroenteritis (Stomach Flu / Food Poisoning)": "Gastroenteritis",
    "Acid Reflux (GERD)": "Gastroesophageal reflux disease",
    "Urinary Tract Infection (UTI)": "Urinary tract infection",
    "Dengue Fever": "Dengue fever",
    "Typhoid Fever": "Typhoid fever",
    "Malaria": "Malaria",
    "Chickenpox": "Chickenpox",
    "Conjunctivitis (Eye Flu)": "Conjunctivitis",
    "Heat Exhaustion": "Heat exhaustion",
    "Dehydration": "Dehydration",
    "Fungal Skin Infection": "Dermatophytosis",
    "Ear Infection (Otitis Media)": "Otitis media",
    "Chikungunya": "Chikungunya",
    "Tuberculosis (TB)": "Tuberculosis",
    "Hepatitis A": "Hepatitis A",
    "Leptospirosis": "Leptospirosis",
}

# What each condition is called inside Wikidata (lower-case; a condition can
# have several names - we take the symptoms of all of them).
WIKIDATA_DISEASE_LABELS = {
    "Common Cold": ["common cold"],
    "Influenza (Flu)": ["influenza"],
    "COVID-like Viral Illness": ["covid-19", "coronavirus disease 2019"],
    "Allergic Rhinitis (Hay Fever)": ["allergic rhinitis", "hay fever", "pollinosis"],
    "Sinusitis": ["sinusitis"],
    "Acute Bronchitis": ["acute bronchitis", "bronchitis"],
    "Asthma Flare-Up": ["asthma"],
    "Pneumonia": ["pneumonia"],
    "Migraine": ["migraine"],
    "Tension Headache": ["tension headache", "tension-type headache"],
    "Viral Fever (general)": ["influenza-like illness"],
    "Strep Throat (Bacterial)": ["streptococcal pharyngitis", "strep throat"],
    "Gastroenteritis (Stomach Flu / Food Poisoning)": ["gastroenteritis", "infectious diarrhea",
                                                       "food poisoning"],
    "Acid Reflux (GERD)": ["gastroesophageal reflux disease"],
    "Urinary Tract Infection (UTI)": ["urinary tract infection"],
    "Dengue Fever": ["dengue fever", "dengue"],
    "Typhoid Fever": ["typhoid fever", "typhoid", "enteric fever"],
    "Malaria": ["malaria"],
    "Chickenpox": ["chickenpox", "varicella"],
    "Conjunctivitis (Eye Flu)": ["conjunctivitis"],
    "Heat Exhaustion": ["heat exhaustion"],
    "Dehydration": ["dehydration"],
    "Fungal Skin Infection": ["dermatophytosis", "tinea", "ringworm", "dermatomycosis"],
    "Ear Infection (Otitis Media)": ["otitis media"],
    "Chikungunya": ["chikungunya"],
    "Tuberculosis (TB)": ["tuberculosis", "pulmonary tuberculosis"],
    "Hepatitis A": ["hepatitis a"],
    "Leptospirosis": ["leptospirosis"],
    # v3.0 back-pain trio:
    "Lower Back Muscle Strain": ["lumbar strain", "low back pain", "back strain",
                                 "pulled back muscle", "muscle strain"],
    "Sciatica (Lumbar Radiculopathy)": ["sciatica", "lumbar radiculopathy",
                                        "radiculopathy", "sciatic nerve"],
    "Lumbar Disc Herniation (Slipped Disc)": ["lumbar disc herniation", "herniated disc",
                                              "disc herniation", "slipped disc",
                                              "disc protrusion", "disc bulge"],
}

# The internet writes symptoms in medical English ("dysuria"), while the app
# shows friendly words ("Burning when urinating"). For every symptom we keep
# a list of phrases to look for online: the symptom itself plus these
# synonyms. Add a line here whenever a symptom fails to match online.
INTERNET_SYMPTOM_TERMS = {
    "Fever": ["fever", "pyrexia", "high temperature", "elevated temperature", "feverish"],
    "High fever (39°C / 102°F or above)": ["high fever", "fever", "feverish"],
    "Chills or shivering": ["chills", "shivering", "rigors", "rigor"],
    "Night sweats": ["night sweats", "nocturnal sweating"],
    "Heavy sweating": ["excessive sweating", "heavy sweating", "diaphoresis", "sweating"],
    "Runny nose": ["runny nose", "rhinorrhea", "nasal discharge"],
    "Blocked nose": ["blocked nose", "nasal congestion", "stuffy nose", "congested nose"],
    "Sneezing": ["sneezing", "sneeze"],
    "Itchy eyes": ["itchy eyes", "itching of the eyes", "ocular pruritus", "pruritus"],
    "Watery eyes": ["watery eyes", "watering eyes", "epiphora", "lacrimation", "tearing"],
    "Red eyes": ["red eyes", "red eye", "redness of the eyes", "eye redness", "bloodshot"],
    "Eye discharge or crusty eyelids": ["eye discharge", "discharge from the eyes",
                                        "crusty eyelids", "mucous discharge"],
    "Gritty feeling in the eyes": ["gritty", "foreign body sensation", "gritty feeling"],
    "Sore throat": ["sore throat", "throat pain", "pharyngitis", "scratchy throat"],
    "Pain when swallowing": ["pain when swallowing", "painful swallowing", "odynophagia",
                             "difficulty swallowing"],
    "Swollen glands in the neck": ["swollen glands", "swollen lymph nodes", "lymphadenopathy",
                                   "enlarged lymph nodes", "tender lymph nodes"],
    "White patches on tonsils": ["white patches", "white spots", "tonsillar exudate",
                                 "pus on the tonsils"],
    "Dry cough": ["dry cough", "cough", "coughing"],
    "Cough with phlegm": ["productive cough", "cough with phlegm", "sputum", "cough"],
    "Cough worse at night": ["nocturnal cough", "cough at night", "nighttime cough"],
    "Shortness of breath": ["shortness of breath", "dyspnea", "dyspnoea", "breathlessness",
                            "difficulty breathing", "short of breath", "labored breathing"],
    "Wheezing": ["wheezing", "wheeze", "wheezing sound"],
    "Chest tightness": ["chest tightness", "tightness in the chest"],
    "Chest pain when breathing": ["chest pain", "pleuritic", "pain when breathing"],
    "Headache": ["headache", "head pain", "cephalgia"],
    "One-sided throbbing headache": ["unilateral", "throbbing", "one-sided", "pulsating"],
    "Band-like pressure around the head": ["band-like", "pressing", "tightening",
                                           "pressure around the head"],
    "Stiff or tense neck and shoulders": ["stiff neck", "neck stiffness", "tender neck",
                                          "neck and shoulder", "tight neck"],
    "Light sensitivity": ["photophobia", "light sensitivity", "sensitivity to light"],
    "Sound sensitivity": ["phonophobia", "sound sensitivity", "sensitivity to sound"],
    "Visual aura (flashing lights or zigzags)": ["aura", "visual aura", "flashing lights",
                                                 "zigzag", "scintillating"],
    "Body aches / muscle pain": ["body aches", "muscle pain", "myalgia", "muscle aches",
                                 "body pain", "aching muscles"],
    "Joint pain": ["joint pain", "arthralgia", "joint pains", "aching joints"],
    "Pain behind the eyes": ["pain behind the eyes", "retro-orbital", "behind the eyes",
                             "eye pain"],
    "Facial pain or pressure": ["facial pain", "facial pressure", "face pain",
                                "pressure in the face"],
    "Reduced sense of smell": ["reduced sense of smell", "hyposmia", "decreased sense of smell"],
    "Loss of taste or smell": ["loss of taste", "loss of smell", "anosmia", "ageusia",
                               "loss of the sense of smell"],
    "Fatigue or weakness": ["fatigue", "weakness", "tiredness", "malaise", "exhaustion",
                            "feeling tired"],
    "Dizziness": ["dizziness", "lightheadedness", "vertigo", "lightheaded"],
    "Nausea": ["nausea", "feeling sick"],
    "Vomiting": ["vomiting", "emesis", "throwing up"],
    "Diarrhea": ["diarrhea", "diarrhoea", "loose stools", "loose motions", "watery stools"],
    "Constipation": ["constipation"],
    "Abdominal pain or cramps": ["abdominal pain", "stomach pain", "stomach cramps",
                                 "abdominal cramps", "stomach ache", "belly pain"],
    "Bloating": ["bloating", "abdominal bloating", "abdominal distension", "distended"],
    "Heartburn / burning behind the chest": ["heartburn", "acid reflux", "burning chest pain"],
    "Sour or bitter taste in the throat": ["sour taste", "bitter taste", "regurgitation",
                                           "acid taste"],
    "Belching or burping": ["belching", "burping", "eructation"],
    "Burning when urinating": ["dysuria", "burning urination", "burning when urinating",
                               "painful urination", "burning with urination"],
    "Frequent urination": ["frequent urination", "urinary frequency",
                           "frequent passage of urine"],
    "Cloudy or strong-smelling urine": ["cloudy urine", "strong-smelling urine",
                                        "foul-smelling urine"],
    "Blood in urine": ["hematuria", "blood in urine", "blood in the urine", "bloody urine"],
    "Lower back pain": ["lower back pain", "back pain", "flank pain", "lumbago",
                        "low back pain"],
    # v3.0 back / nerve symptoms:
    "Pain shooting down the leg (below the knee)": ["sciatica", "sciatic", "radicular pain",
                                                    "shooting leg pain", "radiating leg pain"],
    "Numbness or tingling in the leg or foot": ["paresthesia", "numbness", "tingling",
                                                "pins and needles"],
    "Weakness in the leg or foot": ["muscle weakness", "leg weakness", "paresis", "foot drop"],
    "Back pain worse when sitting or bending forward": ["worse on sitting", "pain on sitting",
                                                        "flexion pain"],
    "Back pain worse when coughing or sneezing": ["pain on coughing", "pain on sneezing",
                                                  "cough-induced pain"],
    "Back pain that started after lifting or twisting": ["lifting injury", "after lifting",
                                                         "onset after exertion"],
    "Back stiffness in the morning": ["morning stiffness", "stiff back"],
    "Itchy rash": ["itchy rash", "itching", "itchy skin", "pruritus", "skin rash", "rash"],
    "Rash with small fluid-filled blisters": ["fluid-filled blisters", "blisters", "blister",
                                              "vesicles", "vesicular rash", "rash"],
    "Ring-shaped scaly rash": ["ring-shaped", "ringworm", "scaly", "annular", "ring-like"],
    "Skin peeling or flaking": ["skin peeling", "peeling", "flaking", "scaling"],
    "Excessive thirst": ["excessive thirst", "polydipsia", "intense thirst", "thirsty",
                          "thirst"],
    "Dry mouth": ["dry mouth", "xerostomia", "dryness of the mouth"],
    "Dark or little urine": ["dark urine", "dark-colored urine", "reduced urine", "oliguria"],
    "Ear pain": ["ear pain", "earache", "otalgia", "pain in the ear"],
    "Ear discharge": ["ear discharge", "otorrhea", "discharge from the ear"],
    "Reduced hearing": ["hearing loss", "reduced hearing", "deafness", "muffled hearing"],
    "Loss of appetite": ["loss of appetite", "anorexia", "poor appetite",
                         "decreased appetite", "reduced appetite"],
    "Muscle cramps": ["muscle cramps", "muscle spasms", "cramping muscles"],
    "Pale, cool or clammy skin": ["clammy skin", "pale skin", "cool and clammy skin", "clammy"],
    "Coughing up blood": ["hemoptysis", "coughing up blood", "blood in sputum"],
    "Unintentional weight loss": ["weight loss", "unintentional weight loss", "losing weight"],
    "Yellow eyes or skin (jaundice)": ["jaundice", "yellow eyes", "yellow skin",
                                       "yellowing of the skin", "icterus"],
    "Pale (clay-coloured) stools": ["pale stool", "clay-colored stool", "clay-coloured stool",
                                    "acholic stools"],
    "Severe joint pain (hands and feet)": ["severe joint pain", "joint pain", "arthralgia",
                                           "swollen joints"],
    "Bleeding gums or nose": ["bleeding gums", "nosebleed", "epistaxis", "gingival bleeding",
                              "bleeding from the gums"],
}


def _ik_norm(text):
    """Make text comparable before matching: lower-case, drop anything in
    brackets, drop punctuation, squeeze spaces.  'High fever (39°C)' and
    'high fever' should end up identical."""
    t = re.sub(r"\([^)]*\)", " ", str(text or ""))
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return " ".join(t.split())


def symptom_search_terms(symptom):
    """Every phrase worth looking for when matching this symptom against the
    internet knowledge: the symptom itself plus the synonym table above."""
    terms = []
    for t in [_ik_norm(symptom)] + list(INTERNET_SYMPTOM_TERMS.get(symptom, [])):
        t = _ik_norm(t)
        if t and t not in terms:
            terms.append(t)
    return terms


def fetch_structured_symptoms():
    """Ask the linked-data web (Wikidata first, university mirror and DBpedia
    as backups) which symptoms belong to each of our conditions.
    Returns ({condition: set of normalised symptom phrases}, source name,
    {condition: Wikipedia article URL collected along the way})."""
    label_to_name, variants = _disease_label_variants()
    quoted = ", ".join(f'"{v}"' for v in variants)
    rows, used_source = [], None
    for source_name, endpoint, clause, url_clause in SPARQL_SOURCES:
        query = (
            "SELECT DISTINCT ?dl ?sl ?url WHERE { "
            f"{clause} "
            '?disease rdfs:label ?dl FILTER(LANG(?dl) = "en") '
            f"FILTER(LCASE(STR(?dl)) IN ({quoted})) "
            '?symptom rdfs:label ?sl FILTER(LANG(?sl) = "en") '
            f"{url_clause}}} LIMIT 3000"
        )
        try:
            r = requests.get(endpoint, params={"query": query, "format": "json"},
                             headers=HTTP_HEADERS, timeout=45)
            if r.status_code == 200:
                rows = r.json().get("results", {}).get("bindings", [])
                if rows:
                    used_source = source_name
                    break
        except Exception:
            continue
    out, urls = {}, {}
    for row in rows:
        name = label_to_name.get(str(row["dl"]["value"]).strip().lower())
        sym = _ik_norm(row["sl"]["value"])
        if name and sym:
            out.setdefault(name, set()).add(sym)
        if name and "url" in row:
            urls.setdefault(name, str(row["url"]["value"]))
    return out, used_source, urls


def _disease_label_variants():
    """(label -> condition name) map plus the flat lower-case label list,
    shared by the SPARQL queries below."""
    label_to_name, variants = {}, []
    for name, labels in WIKIDATA_DISEASE_LABELS.items():
        for lb in labels:
            key = lb.strip().lower()
            if key and key not in label_to_name:
                label_to_name[key] = name
                variants.append(key)
    return label_to_name, variants


def fetch_wikipedia_summaries():
    """Download the opening summary of each condition's Wikipedia article.
    Returns {condition name: {title, url, summary, text}} - 'text' is the
    normalised copy used for symptom matching, 'summary' is for display."""
    out = {}
    for name, title in WIKIPEDIA_TITLES.items():
        slug = title.strip().replace(" ", "_")
        try:
            r = requests.get(f"{WIKIPEDIA_REST_URL}/{slug}", headers=HTTP_HEADERS, timeout=20)
            if r.status_code == 200:
                data = r.json()
                extract = " ".join((data.get("extract") or "").split())
                if extract:
                    out[name] = {
                        "title": data.get("title") or title,
                        "url": (data.get("content_urls") or {}).get("desktop", {})
                                  .get("page") or f"https://en.wikipedia.org/wiki/{slug}",
                        "summary": extract[:400],
                        "text": _ik_norm(extract),
                    }
            time.sleep(0.15)     # be polite to the free service
        except Exception:
            continue
    return out


def _knowledge_cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), KNOWLEDGE_CACHE_FILE)


def _load_knowledge_file():
    """Read internet_knowledge.json; returns None if missing or damaged."""
    try:
        with open(_knowledge_cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "fetched_at": float(data.get("fetched_at") or 0),
            "structured_source": data.get("structured_source") or "Wikidata",
            "wikidata": {k: set(v) for k, v in (data.get("wikidata") or {}).items()},
            "wikipedia": data.get("wikipedia") or {},
        }
    except Exception:
        return None


def load_internet_knowledge(force_refresh=False):
    """The one entry point the interface uses. Returns the knowledge dict
    (structured symptom links + wikipedia summaries) or None when nothing is
    available. Downloads only when the cache is missing or a week old."""
    if not force_refresh:
        cached = _load_knowledge_file()
        if cached and (time.time() - cached["fetched_at"]) / 3600 < KNOWLEDGE_MAX_AGE_HOURS \
                and (cached["wikidata"] or cached["wikipedia"]):
            return cached
    wikidata, structured_source, wiki_urls = fetch_structured_symptoms()
    wikipedia = fetch_wikipedia_summaries()
    # When the Wikipedia REST API could not be reached, the structured source
    # still gave us the article URLs - keep those so every condition can
    # link out to its Wikipedia page even without the summary text.
    for name, url in (wiki_urls or {}).items():
        if name not in wikipedia and name in WIKIPEDIA_TITLES:
            wikipedia[name] = {"title": WIKIPEDIA_TITLES[name], "url": url,
                               "summary": "", "text": ""}
    if wikidata or wikipedia:                      # something arrived - save it
        data = {"fetched_at": time.time(), "structured_source": structured_source,
                "wikidata": {k: sorted(v) for k, v in wikidata.items()},
                "wikipedia": wikipedia}
        try:
            with open(_knowledge_cache_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass                                   # a read-only disk is fine
        return {"fetched_at": data["fetched_at"], "structured_source": structured_source,
                "wikidata": wikidata, "wikipedia": wikipedia}
    stale = _load_knowledge_file()                 # nothing arrived - an old
    return stale                                    # cache is better than none


def internet_knowledge_summary(knowledge):
    """One line of status text for the sidebar."""
    if not knowledge or not (knowledge.get("wikidata") or knowledge.get("wikipedia")):
        return None
    when = datetime.fromtimestamp(knowledge.get("fetched_at") or 0).strftime("%d %b %Y")
    links = sum(len(v) for v in (knowledge.get("wikidata") or {}).values())
    source = knowledge.get("structured_source") or "Wikidata"
    articles = sum(1 for a in (knowledge.get("wikipedia") or {}).values() if a.get("text"))
    return (f"Online medical knowledge cached on {when}: "
            f"{articles} Wikipedia summaries, "
            f"{len(knowledge.get('wikidata') or {})} conditions with symptom links "
            f"on {source} ({links} links).")


def internet_evidence(symptoms, disease_name, knowledge):
    """Check the user's symptoms against the live online knowledge for ONE
    disease. Returns (fraction 0-1 or None, [(symptom, source), ...]).
    None means we have no online knowledge for this disease at all, and the
    score blending then ignores the internet voice for it."""
    wiki_symptoms = (knowledge.get("wikidata") or {}).get(disease_name)
    article = (knowledge.get("wikipedia") or {}).get(disease_name)
    if wiki_symptoms is None and article is None:
        return None, []
    structured_label = knowledge.get("structured_source") or "Wikidata"
    unique, confirmed = [], []
    for s in symptoms:
        if s not in unique:
            unique.append(s)
    for s in unique:
        terms = symptom_search_terms(s)
        source = None
        if wiki_symptoms and any(t in wiki_symptoms for t in terms):
            source = structured_label
        elif article and article.get("text"):
            for t in terms:
                if len(t) >= 4 and re.search(rf"\b{re.escape(t)}\b", article["text"]):
                    source = "Wikipedia"
                    break
        if source:
            confirmed.append((s, source))
    if not unique:
        return None, []
    return len(confirmed) / len(unique), confirmed


# ---------------------------------------------------------------------------
# PART 3 - THE PREDICTION ENGINE (pure Python, works offline)
# ---------------------------------------------------------------------------
# The idea in one paragraph:
#   For every disease we ask two questions -
#   (a) RECALL: how much of this disease's typical symptom picture did the
#       user report? (weighted - a "3" symptom counts triple)
#   (b) PRECISION: how well does this disease explain ALL the reported
#       symptoms? (penalises diseases that cover only 1 of 8 symptoms)
#   Then we ADD small bonuses when the season or the live weather in the
#   user's city makes that disease more likely right now. The result is a
#   0-1 score shown as a percentage - WITH the reasons spelled out.
# ---------------------------------------------------------------------------

MIN_SCORE_TO_SHOW = 0.15
RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
SEASON_BONUS = 0.08


def month_to_season(month, lat):
    """Calendar season for a month, flipped automatically for the Southern Hemisphere."""
    if lat is not None and lat < 0:
        month = ((month + 5) % 12) + 1  # shift about 6 months for the south
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def is_monsoon_time(month, lat):
    """Rough tropical wet-season approximation: Jun-Oct north of the equator,
    Nov-Mar south of it. Used for dengue/malaria/typhoid-style diseases."""
    southern = lat is not None and lat < 0
    return month in (11, 12, 1, 2, 3) if southern else month in (6, 7, 8, 9, 10)


def month_from_weather(wx):
    """The month AT THE USER'S LOCATION, taken from the weather API's local time."""
    if wx and wx.get("local_time"):
        try:
            return datetime.fromisoformat(wx["local_time"]).month
        except ValueError:
            pass
    return datetime.now(timezone.utc).month  # fallback: current UTC month


def season_reason(disease, month, lat):
    """A human-readable reason if this disease's season is active right now."""
    for tag in disease.get("seasons") or []:
        if tag == "monsoon":
            if is_monsoon_time(month, lat):
                return "It is roughly the tropical wet season in your region"
        elif tag == month_to_season(month, lat):
            return f"It is {tag} in your part of the world"
    return None


def weather_boost_reason(rule, wx):
    """Check one weather rule against the live weather; return a reason or None."""
    if not wx or wx.get("temp") is None:
        return None
    t = wx["temp"]
    h = wx.get("humidity") or 0
    p = wx.get("precip") or 0
    code = wx.get("code") or 0
    wind = wx.get("wind") or 0
    aqi, pm25 = wx.get("aqi"), wx.get("pm25")
    raining = p >= 0.5 or code in RAIN_CODES

    if rule == "cold" and t <= 10:
        return f"Live weather: cold right now ({t:.0f}°C)"
    if rule == "heat" and t >= 35:
        return f"Live weather: very hot right now ({t:.0f}°C)"
    if rule == "warm_rain" and raining and 20 <= t < 35:
        return f"Live weather: warm and rainy ({t:.0f}°C) - conditions that follow rain"
    if rule == "humid" and h >= 78 and t >= 24:
        return f"Live weather: hot and humid ({h:.0f}% humidity)"
    if rule == "dry_windy" and h <= 45 and wind >= 25:
        return f"Live weather: dry and windy ({wind:.0f} km/h) - spreads pollen"
    if rule == "poor_air":
        if (aqi is not None and aqi >= 100) or (aqi is None and pm25 is not None and pm25 >= 40):
            label = f"AQI {aqi:.0f}" if aqi is not None else f"PM2.5 {pm25:.0f}"
            return f"Live air quality is poor ({label}) - irritates airways"
    return None


def score_diseases(selected, wx, month, lat):
    """Score every disease against the selected symptoms.
    Returns the top 3 as a list of dictionaries with score, matches and reasons."""
    results = []
    n = len(selected)
    for name, d in DISEASES.items():
        matched = [s for s in selected if s in d["symptoms"]]
        if not matched:
            continue

        total_w = sum(d["symptoms"].values())
        got_w = sum(d["symptoms"][s] for s in matched)
        recall = got_w / total_w                    # how much of the disease we cover
        precision = len(matched) / n                # how much of the input it explains
        base = 0.55 * recall + 0.45 * precision

        # With very few symptoms every diagnosis is a guess - damp the score.
        if n == 1:
            base *= 0.65
        elif n == 2:
            base *= 0.80

        boost, reasons = 0.0, []
        s_reason = season_reason(d, month, lat)
        if s_reason:
            boost += SEASON_BONUS
            reasons.append(f"Season: {s_reason} (+{int(SEASON_BONUS * 100)}%)")
        for rule, gain in d.get("weather_boosts") or []:
            w_reason = weather_boost_reason(rule, wx)
            if w_reason:
                boost += gain
                reasons.append(f"{w_reason} (+{int(gain * 100)}%)")

        score = min(base + boost, 0.97)
        if score < MIN_SCORE_TO_SHOW:
            continue

        # Key symptoms the user did NOT report - shown as a hint, not a verdict.
        missing = [s for s, w in sorted(d["symptoms"].items(), key=lambda kv: -kv[1])
                   if s not in matched and w >= 2][:3]

        results.append({"name": name, "score": score, "matched": matched,
                        "missing": missing, "reasons": reasons})

    results.sort(key=lambda r: r["score"], reverse=True)
    # v2.0: return up to 6 candidates - the hybrid engine in PART 4 blends
    # these rule scores with the ML probabilities before showing the top 3.
    return results[:6]


def aqi_category(aqi):
    """US EPA bands for the air quality index."""
    if aqi is None:
        return "unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for sensitive groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very unhealthy"
    return "Hazardous"


def match_label(score):
    if score >= 0.70:
        return "Strong match"
    if score >= 0.50:
        return "Good match"
    if score >= 0.30:
        return "Possible match"
    return "Weak match"


def build_trend_notes(wx, month, lat):
    """The 'what is going around right now' panel - season + live weather trends."""
    notes = []
    season = month_to_season(month, lat)
    hemisphere = "Southern" if (lat is not None and lat < 0) else "Northern"
    notes.append(f"It is {season} in your region ({hemisphere} hemisphere).")
    if season == "winter":
        notes.append("Respiratory viruses - flu, common cold, COVID-like illness - circulate most in cold months.")
    elif season == "spring":
        notes.append("Pollen counts rise in spring - hay fever (allergic rhinitis) is common.")
    elif season == "summer":
        notes.append("Hot months bring heat exhaustion, dehydration and food-borne illness.")
    else:
        notes.append("Autumn often brings a rise in colds and viral fevers as temperatures fall.")

    if lat is not None and abs(lat) < 23.5 and is_monsoon_time(month, lat):
        notes.append("This is roughly the tropical wet season - dengue, malaria, typhoid and eye flu circulate more; avoid standing water.")

    if wx and wx.get("temp") is not None:
        t = wx["temp"]
        h = wx.get("humidity") or 0
        code = wx.get("code") or 0
        wind = wx.get("wind") or 0
        raining = (wx.get("precip") or 0) >= 0.5 or code in RAIN_CODES
        if raining and 20 <= t < 35:
            notes.append("Live conditions are warm and rainy - mosquito-borne illnesses rise after rain.")
        if t >= 35:
            notes.append("It is very hot right now - drink extra water and watch for heat exhaustion.")
        if t <= 10:
            notes.append("It is cold right now - flu and cold viruses are more active.")
        if h <= 45 and wind >= 25:
            notes.append("Dry, windy weather spreads pollen - allergies may flare.")
        aqi = wx.get("aqi")
        if aqi is not None and aqi >= 100:
            notes.append(f"Air quality is {aqi_category(aqi).lower()} (AQI {aqi:.0f}) - asthma and bronchitis can worsen.")
    return notes


# ---------------------------------------------------------------------------
# PART 4 - THE LEARNING ENGINE (machine learning on top of the rules)
# ---------------------------------------------------------------------------
# WHY ML ON TOP OF RULES?
#   The rule engine (PART 3) is transparent but rigid: it only knows what
#   somebody wrote down. The learning engine learns from EXAMPLES instead.
#   For every patient record it sees (symptoms -> diagnosis) it adjusts
#   weights, so patterns that appear in YOUR data get predicted even when
#   nobody wrote a rule for them. The final score shown to the user is a
#   BLEND of both engines (you control the mix with a slider). Hand-written
#   knowledge + learned knowledge together is called a HYBRID system - it
#   is what most real triage tools do too.
#
# HOW THE LEARNING WORKS (no math degree needed):
#   1. Every patient becomes a row of 0s and 1s - one column per symptom
#      (1 = had it), plus a few columns for age band, gender and season.
#   2. LogisticRegression (a classic algorithm from scikit-learn) reads
#      thousands of such rows and learns, for every disease, which columns
#      push its probability up and which pull it down.
#   3. At prediction time the user's answers become one such row, and the
#      model outputs a probability for every disease it has ever seen.
#
# WHERE THE TRAINING RECORDS COME FROM:
#   (a) ~2,000 synthetic "example patients" generated from the knowledge
#       base (so the model works out of the box), and
#   (b) CSV files you upload in Doctor mode - real anonymised records.
#   Save them as patient_data.csv next to app.py and they load at startup.
# ---------------------------------------------------------------------------

PATIENT_DATA_FILE = "patient_data.csv"  # auto-loaded if placed next to app.py
SEED_PER_DISEASE = 70                   # synthetic example patients per condition
SYMPTOM_SEP = ";"                       # separator used inside CSV cells
SYMPTOM_SET = set(SYMPTOMS)

# ---- text normalisation helpers -------------------------------------------
# CSV files written by humans contain "Dengue fever", "DENGUE", "dengue!",
# "loose motions"... These helpers map all of that onto the app's canonical
# names so the knowledge base and the model stay consistent.


def _norm_key(text):
    """lowercase + collapse punctuation to spaces: 'Dengue  Fever!' -> 'dengue fever'"""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


_SYMPTOM_LOOKUP = {_norm_key(s): s for s in SYMPTOMS}
_DISEASE_LOOKUP = {_norm_key(d): d for d in DISEASES}
_ALL_TEST_NAMES = [t for tests in TYPICAL_TESTS.values() for t, _ in tests]

# Common Indian shorthand understood automatically during CSV import.
SYMPTOM_ALIASES = {
    "loose motion": "Diarrhea", "loose motions": "Diarrhea", "loose stools": "Diarrhea",
    "vomitting": "Vomiting", "vomitting sensation": "Nausea",
    "running nose": "Runny nose", "body ache": "Body aches / muscle pain",
    "body pains": "Body aches / muscle pain", "muscle pain": "Body aches / muscle pain",
    "weakness": "Fatigue or weakness", "giddiness": "Dizziness",
    "burning micturition": "Burning when urinating",
    "burning urination": "Burning when urinating",
    "pain in abdomen": "Abdominal pain or cramps", "stomach pain": "Abdominal pain or cramps",
    "stomach ache": "Abdominal pain or cramps",
    "breathlessness": "Shortness of breath", "difficulty breathing": "Shortness of breath",
    "weight loss": "Unintentional weight loss",
    "jaundice": "Yellow eyes or skin (jaundice)", "yellowish eyes": "Yellow eyes or skin (jaundice)",
    "head ache": "Headache", "pain in throat": "Sore throat", "throat pain": "Sore throat",
    "rash": "Itchy rash", "itching": "Itchy rash",
    # v4.0 - how people actually talk (no medical words needed):
    "pain in chest": "Chest pain when breathing", "chest hurts": "Chest pain when breathing",
    "my chest hurts": "Chest pain when breathing", "chest pain": "Chest pain when breathing",
    "pain in my chest": "Chest pain when breathing", "in my chest": "Chest pain when breathing",
    "cant breathe": "Shortness of breath", "cant breathe properly": "Shortness of breath",
    "cant breath": "Shortness of breath", "hard to breathe": "Shortness of breath",
    "not able to breathe": "Shortness of breath", "short of breath": "Shortness of breath",
    "breathing problem": "Shortness of breath", "breath shortness": "Shortness of breath",
    "dizzy": "Dizziness", "feeling dizzy": "Dizziness", "im feeling dizzy": "Dizziness",
    "head is spinning": "Dizziness", "head spinning": "Dizziness",
    "tired": "Fatigue or weakness", "im tired": "Fatigue or weakness",
    "very tired": "Fatigue or weakness", "no energy": "Fatigue or weakness",
    "feeling weak": "Fatigue or weakness", "weak": "Fatigue or weakness",
    "feel like vomiting": "Nausea", "feel like puking": "Nausea",
    "nauseous": "Nausea", "feeling sick": "Nausea",
    "throwing up": "Vomiting", "puking": "Vomiting", "vomitted": "Vomiting",
    "acidity": "Heartburn / burning behind the chest", "acid reflux": "Heartburn / burning behind the chest",
    "gas": "Bloating", "bloating": "Bloating", "burping": "Belching or burping",
    "skin rash": "Itchy rash", "rashes": "Itchy rash", "red spots": "Itchy rash",
    "have temperature": "Fever", "got temperature": "Fever", "running a temperature": "Fever",
    "feverish": "Fever", "temperature": "Fever",
    "cough": "Dry cough", "coughing": "Dry cough", "khaansi": "Dry cough",
    "khansi": "Dry cough", "cold and cough": "Dry cough",
    "sneeze": "Sneezing", "sneezing a lot": "Sneezing",
    "chakkar": "Dizziness", "chakkar aa rahe": "Dizziness",
    "pet dard": "Abdominal pain or cramps", "pet mein dard": "Abdominal pain or cramps",
    "bukhar": "Fever", "bukhaar": "Fever", "zukam": "Runny nose",
    "cloudy urine": "Cloudy or strong-smelling urine", "strong smelling urine": "Cloudy or strong-smelling urine",
    "foul smelling urine": "Cloudy or strong-smelling urine",
    "joint pains": "Joint pain", "severe joint pain": "Severe joint pain (hands and feet)",
    "pain behind eyes": "Pain behind the eyes", "high temperature": "Fever",
    "bleeding gums": "Bleeding gums or nose",
    # v3.0 - back / nerve symptoms as people actually say them:
    "back pain": "Lower back pain", "backache": "Lower back pain", "back ache": "Lower back pain",
    "lower backache": "Lower back pain", "pain in lower back": "Lower back pain",
    "lower back ache": "Lower back pain", "waist pain": "Lower back pain",
    "kamar dard": "Lower back pain", "kamar dard hai": "Lower back pain",
    "pain in my back": "Lower back pain", "lower back pain": "Lower back pain",
    "back pain lower": "Lower back pain", "lumbar pain": "Lower back pain",
    "sciatic pain": "Pain shooting down the leg (below the knee)",
    "sciatica pain": "Pain shooting down the leg (below the knee)",
    "leg pain shooting": "Pain shooting down the leg (below the knee)",
    "shooting pain in leg": "Pain shooting down the leg (below the knee)",
    "shooting leg pain": "Pain shooting down the leg (below the knee)",
    "pain radiating to leg": "Pain shooting down the leg (below the knee)",
    "pain radiating down my leg": "Pain shooting down the leg (below the knee)",
    "pain going down my leg": "Pain shooting down the leg (below the knee)",
    "pain down the back of my leg": "Pain shooting down the leg (below the knee)",
    "electric shock like pain in leg": "Pain shooting down the leg (below the knee)",
    "leg pain below knee": "Pain shooting down the leg (below the knee)",
    "pain till my calf": "Pain shooting down the leg (below the knee)",
    "pain till the calf": "Pain shooting down the leg (below the knee)",
    "pain goes to my calf": "Pain shooting down the leg (below the knee)",
    "numbness in leg": "Numbness or tingling in the leg or foot",
    "numbness in foot": "Numbness or tingling in the leg or foot",
    "tingling in leg": "Numbness or tingling in the leg or foot",
    "tingling in my feet": "Numbness or tingling in the leg or foot",
    "pins and needles in leg": "Numbness or tingling in the leg or foot",
    "pins and needles": "Numbness or tingling in the leg or foot",
    "legs feel numb": "Numbness or tingling in the leg or foot",
    "my leg feels numb": "Numbness or tingling in the leg or foot",
    "leg going numb": "Numbness or tingling in the leg or foot",
    "weak leg": "Weakness in the leg or foot", "weak legs": "Weakness in the leg or foot",
    "leg feels weak": "Weakness in the leg or foot",
    "foot drop": "Weakness in the leg or foot",
    "cannot stand on toes": "Weakness in the leg or foot",
    "cant stand on toes": "Weakness in the leg or foot",
    "my leg gives way": "Weakness in the leg or foot",
    "leg feels heavy": "Weakness in the leg or foot",
    "back pain after deadlift": "Back pain that started after lifting or twisting",
    "back pain after lifting": "Back pain that started after lifting or twisting",
    "back pain after lifting heavy": "Back pain that started after lifting or twisting",
    "back pain after gym": "Back pain that started after lifting or twisting",
    "pain after lifting": "Back pain that started after lifting or twisting",
    "hurt my back lifting": "Back pain that started after lifting or twisting",
    "hurt my back deadlifting": "Back pain that started after lifting or twisting",
    "injured my back deadlift": "Back pain that started after lifting or twisting",
    "deadlift": "Back pain that started after lifting or twisting",
    "deadlifts": "Back pain that started after lifting or twisting",
    "deadlifting": "Back pain that started after lifting or twisting",
    "hurt my back": "Back pain that started after lifting or twisting",
    "injured my back": "Back pain that started after lifting or twisting",
    "pulled my back": "Back pain that started after lifting or twisting",
    "lifting injury": "Back pain that started after lifting or twisting",
    "hurt my back at gym": "Back pain that started after lifting or twisting",
    "back pain when sitting": "Back pain worse when sitting or bending forward",
    "pain when sitting": "Back pain worse when sitting or bending forward",
    "pain while sitting": "Back pain worse when sitting or bending forward",
    "back pain when bending": "Back pain worse when sitting or bending forward",
    "pain when bending forward": "Back pain worse when sitting or bending forward",
    "cannot bend forward": "Back pain worse when sitting or bending forward",
    "cant bend forward": "Back pain worse when sitting or bending forward",
    "back pain when coughing": "Back pain worse when coughing or sneezing",
    "pain when i cough": "Back pain worse when coughing or sneezing",
    "back pain when sneezing": "Back pain worse when coughing or sneezing",
    "pain when i sneeze": "Back pain worse when coughing or sneezing",
    "sneeze makes pain worse": "Back pain worse when coughing or sneezing",
    "cough makes pain worse": "Back pain worse when coughing or sneezing",
    "morning back stiffness": "Back stiffness in the morning",
    "stiff back in morning": "Back stiffness in the morning",
    "back is stiff": "Back stiffness in the morning",
}

DIAGNOSIS_ALIASES = {
    "flu": "Influenza (Flu)", "influenza": "Influenza (Flu)",
    "fever": "Viral Fever (general)", "viral fever": "Viral Fever (general)",
    "dengue": "Dengue Fever", "malaria": "Malaria",
    "typhoid": "Typhoid Fever", "enteric fever": "Typhoid Fever",
    "uti": "Urinary Tract Infection (UTI)", "urinary tract infection": "Urinary Tract Infection (UTI)",
    "covid": "COVID-like Viral Illness", "covid 19": "COVID-like Viral Illness",
    "cold": "Common Cold", "common cold": "Common Cold",
    "gerd": "Acid Reflux (GERD)", "acid reflux": "Acid Reflux (GERD)",
    "asthma": "Asthma Flare-Up", "tb": "Tuberculosis (TB)", "tuberculosis": "Tuberculosis (TB)",
    "chikungunya": "Chikungunya", "chickenpox": "Chickenpox", "chicken pox": "Chickenpox",
    "food poisoning": "Gastroenteritis (Stomach Flu / Food Poisoning)",
    "gastroenteritis": "Gastroenteritis (Stomach Flu / Food Poisoning)",
    "loose motion": "Gastroenteritis (Stomach Flu / Food Poisoning)",
    "hepatitis a": "Hepatitis A", "hep a": "Hepatitis A",
    "leptospirosis": "Leptospirosis", "pneumonia": "Pneumonia", "migraine": "Migraine",
    "sinusitis": "Sinusitis", "bronchitis": "Acute Bronchitis",
    "eye flu": "Conjunctivitis (Eye Flu)", "conjunctivitis": "Conjunctivitis (Eye Flu)",
    # v3.0 - so the chat and CSV import understand how people actually name these:
    "sciatica": "Sciatica (Lumbar Radiculopathy)",
    "lumbar radiculopathy": "Sciatica (Lumbar Radiculopathy)",
    "radiculopathy": "Sciatica (Lumbar Radiculopathy)",
    "slipped disc": "Lumbar Disc Herniation (Slipped Disc)",
    "disc herniation": "Lumbar Disc Herniation (Slipped Disc)",
    "disc herniation lumbar": "Lumbar Disc Herniation (Slipped Disc)",
    "herniated disc": "Lumbar Disc Herniation (Slipped Disc)",
    "disc bulge": "Lumbar Disc Herniation (Slipped Disc)",
    "bulging disc": "Lumbar Disc Herniation (Slipped Disc)",
    "disc prolapse": "Lumbar Disc Herniation (Slipped Disc)",
    "lumbar disc herniation": "Lumbar Disc Herniation (Slipped Disc)",
    "muscle strain": "Lower Back Muscle Strain",
    "back strain": "Lower Back Muscle Strain",
    "pulled muscle": "Lower Back Muscle Strain",
    "back muscle strain": "Lower Back Muscle Strain",
    "lumbago": "Lower Back Muscle Strain",
}


def normalize_symptom(text):
    """Map free text from a CSV onto the app's canonical symptom names.
    Returns the canonical name, or a tidied version of the text if it is new."""
    t = _norm_key(text)
    if not t:
        return None
    if t in SYMPTOM_ALIASES:
        return SYMPTOM_ALIASES[t]
    if t in _SYMPTOM_LOOKUP:
        return _SYMPTOM_LOOKUP[t]
    if t.endswith("s") and t[:-1] in _SYMPTOM_LOOKUP:      # simple plural fix
        return _SYMPTOM_LOOKUP[t[:-1]]
    hits = [k for k in _SYMPTOM_LOOKUP if t in k]           # partial match
    if hits:
        return _SYMPTOM_LOOKUP[min(hits, key=len)]
    raw = str(text).strip()
    return raw[:1].upper() + raw[1:] if raw else None       # new symptom, keep it


def normalize_diagnosis(text):
    """Map free text onto the app's disease names, or keep it as a new label."""
    t = _norm_key(text)
    if not t:
        return None
    if t in DIAGNOSIS_ALIASES:
        return DIAGNOSIS_ALIASES[t]
    if t in _DISEASE_LOOKUP:
        return _DISEASE_LOOKUP[t]
    if t.endswith("s") and t[:-1] in _DISEASE_LOOKUP:
        return _DISEASE_LOOKUP[t[:-1]]
    hits = [k for k in _DISEASE_LOOKUP if t in k]
    if hits:
        return _DISEASE_LOOKUP[min(hits, key=len)]
    raw = str(text).strip()
    return raw[:1].upper() + raw[1:] if raw else None


def normalize_test_name(text):
    """Map free text like 'cbc' onto the canonical test names when possible."""
    t = _norm_key(text)
    if not t:
        return None
    for name in _ALL_TEST_NAMES:
        if _norm_key(name) == t:
            return name
    if len(t) >= 3:
        for name in _ALL_TEST_NAMES:
            if t in _norm_key(name):
                return name
    raw = str(text).strip()
    return raw[:1].upper() + raw[1:] if raw else None


def _is_empty(value):
    """True for NaN / None / blank strings (pandas reads blanks as NaN)."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _parse_age(value):
    if _is_empty(value):
        return None
    try:
        age = int(float(value))
        return age if 0 <= age <= 120 else None
    except (TypeError, ValueError):
        return None


def _parse_gender(value):
    if _is_empty(value):
        return None
    v = str(value).strip().lower()
    if v.startswith("m"):
        return "male"
    if v.startswith("f"):
        return "female"
    if v.startswith("o"):
        return "other"
    return None


def _parse_month(value):
    if _is_empty(value):
        return None
    s = str(value).strip().lower()
    try:
        m = int(float(s))
        return m if 1 <= m <= 12 else None
    except (TypeError, ValueError):
        pass
    names = ["january", "february", "march", "april", "may", "june",
             "july", "august", "september", "october", "november", "december"]
    for i, nm in enumerate(names, 1):
        if s.startswith(nm[:3]):
            return i
    return None


def age_to_band(age):
    """Collapse an age into one of 4 bands - a small, useful feature for ML."""
    if age is None:
        return "adult"
    if age < 12:
        return "child"
    if age < 18:
        return "teen"
    if age < 65:
        return "adult"
    return "senior"


def age_band_from_group(age_group):
    """Sidebar age-group string -> the same band codes used in training."""
    g = (age_group or "").lower()
    if g.startswith("child"):
        return "child"
    if g.startswith("teen"):
        return "teen"
    if g.startswith("senior"):
        return "senior"
    return "adult"


# ---- 1) synthetic example patients (the model's "textbook") ---------------


def make_seed_patients(n_per_disease=SEED_PER_DISEASE):
    """Generate example patient records FROM the knowledge base itself, so the
    ML model has something to learn from on day one. random.Random(42) makes
    the generator deterministic - same records every run, reproducible tests."""
    rng = random.Random(42)
    patients = []
    season_months = {"winter": [12, 1, 2], "spring": [3, 4, 5],
                     "summer": [6, 7, 8], "autumn": [9, 10, 11]}
    monsoon_months = [6, 7, 8, 9, 10]
    for idx, (name, d) in enumerate(DISEASES.items()):
        # Tests with real names only ("No test needed" is not a test).
        tests = [t for t, _ in d.get("typical_tests", [])
                 if not _norm_key(t).startswith(("no test", "usually no"))]
        for i in range(n_per_disease):
            # Weight-3 symptoms appear in most synthetic patients, weight-1
            # in only a few - exactly how real diseases behave.
            syms = [s for s, w in d["symptoms"].items()
                    if rng.random() < (w / 3.0) * 0.85 + 0.05]
            if not syms:
                syms = [max(d["symptoms"], key=d["symptoms"].get)]
            if rng.random() < 0.30:                     # a little realistic noise
                extra = rng.choice(SYMPTOMS)
                if extra not in syms:
                    syms.append(extra)
            months = []
            for tag in d.get("seasons") or []:
                months += monsoon_months if tag == "monsoon" else season_months.get(tag, [])
            month = rng.choice(months) if months else rng.randint(1, 12)
            patients.append({
                "id": f"EX-{idx * n_per_disease + i + 1:04d}",
                "age": int(rng.triangular(3, 80, 32)),
                "gender": rng.choice(["male", "female"]),
                "city": "",
                "month": month,
                "symptoms": syms,
                "diagnosis": name,
                "tests_done": rng.sample(tests, k=min(len(tests), rng.randint(1, 2))) if tests else [],
                "test_results": "",
                "medicines": [],
                "outcome": "recovered",
                "source": "example",
            })
    return patients


# ---- 2) CSV import (your real patient records) -----------------------------

TEMPLATE_COLUMNS = ["patient_id", "age", "gender", "city", "visit_month",
                    "symptoms", "diagnosis", "tests_done", "test_results",
                    "medicines", "outcome"]

_COLUMN_ALIASES = {
    "patient_id": ["patient_id", "id", "pid", "patientid"],
    "age": ["age", "years"],
    "gender": ["gender", "sex"],
    "city": ["city", "location", "town"],
    "visit_month": ["visit_month", "month", "visit_date_month"],
    "symptoms": ["symptoms", "symptom", "complaints", "presenting_complaints"],
    "diagnosis": ["diagnosis", "disease", "final_diagnosis", "condition", "outcome_diagnosis"],
    "tests_done": ["tests_done", "tests", "test", "scans", "investigations", "investigations_done"],
    "test_results": ["test_results", "results", "result", "reports"],
    "medicines": ["medicines", "medicine", "drugs", "prescription", "treatment"],
    "outcome": ["outcome", "notes", "remarks", "course"],
}


def _find_column(df, canonical):
    for alias in _COLUMN_ALIASES[canonical]:
        if alias in df.columns:
            return alias
    return None


def _split_cell(value):
    """Split a CSV cell on ; | or newlines (commas are kept - symptom names
    themselves contain commas, e.g. 'Pale, cool or clammy skin')."""
    if _is_empty(value):
        return []
    parts = [p.strip() for p in re.split(r"[;|\n]", str(value))]
    return [p for p in parts if p]


def parse_patient_csv(df):
    """Validate and normalise an uploaded DataFrame of patient records.
    Returns (patients, issues, stats):
      patients - list of clean record dicts ready for training
      issues   - human-readable list of skipped rows and why
      stats    - counts, new symptoms/diagnoses discovered, per-diagnosis tally
    """
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    cols = {canon: _find_column(df, canon) for canon in TEMPLATE_COLUMNS}

    missing_required = [c for c in ("symptoms", "diagnosis") if cols[c] is None]
    if missing_required:
        return [], [f"Missing required column(s): {', '.join(missing_required)}"], {}

    patients, issues = [], []
    new_symptoms, new_diagnoses = set(), set()

    for i, row in df.iterrows():
        rowno = i + 2  # +2: header line + 1-based numbering, like a spreadsheet
        if _is_empty(row.get(cols["symptoms"])) or _is_empty(row.get(cols["diagnosis"])):
            issues.append(f"Row {rowno}: skipped - both 'symptoms' and 'diagnosis' are required")
            continue

        syms = []
        for piece in _split_cell(row.get(cols["symptoms"])):
            canon = normalize_symptom(piece)
            if canon and canon not in syms:
                syms.append(canon)
                if canon not in SYMPTOM_SET:
                    new_symptoms.add(canon)
        if not syms:
            issues.append(f"Row {rowno}: skipped - no usable symptoms found")
            continue

        diagnosis = normalize_diagnosis(row.get(cols["diagnosis"]))
        if not diagnosis:
            issues.append(f"Row {rowno}: skipped - diagnosis could not be read")
            continue
        if diagnosis not in DISEASES:
            new_diagnoses.add(diagnosis)

        pid_raw = row.get(cols["patient_id"]) if cols["patient_id"] else None
        pid = f"UP-{rowno:04d}" if _is_empty(pid_raw) else str(pid_raw).strip()

        tests = []
        if cols["tests_done"]:
            for piece in _split_cell(row.get(cols["tests_done"])):
                canon = normalize_test_name(piece)
                if canon and canon not in tests:
                    tests.append(canon)

        medicines = []
        if cols["medicines"]:
            for piece in _split_cell(row.get(cols["medicines"])):
                if piece not in medicines:
                    medicines.append(piece)

        tres = "" if not cols["test_results"] or _is_empty(row.get(cols["test_results"])) \
            else str(row.get(cols["test_results"])).strip()
        outcome = "" if not cols["outcome"] or _is_empty(row.get(cols["outcome"])) \
            else str(row.get(cols["outcome"])).strip()

        patients.append({
            "id": pid,
            "age": _parse_age(row.get(cols["age"])) if cols["age"] else None,
            "gender": _parse_gender(row.get(cols["gender"])) if cols["gender"] else None,
            "city": ("" if not cols["city"] or _is_empty(row.get(cols["city"]))
                     else str(row.get(cols["city"])).strip()),
            "month": _parse_month(row.get(cols["visit_month"])) if cols["visit_month"] else None,
            "symptoms": syms,
            "diagnosis": diagnosis,
            "tests_done": tests,
            "test_results": tres,
            "medicines": medicines,
            "outcome": outcome,
            "source": "uploaded",
        })

    stats = {
        "total": len(df),
        "ok": len(patients),
        "skipped": len(issues),
        "new_symptoms": sorted(new_symptoms),
        "new_diagnoses": sorted(new_diagnoses),
        "by_diagnosis": Counter(p["diagnosis"] for p in patients),
    }
    return patients, issues, stats


def patients_to_csv(patients):
    """Turn learned patient records back into the standard CSV format, so the
    user can save them as patient_data.csv for permanent learning."""
    rows = [{
        "patient_id": p["id"], "age": p["age"], "gender": p["gender"],
        "city": p["city"], "visit_month": p["month"],
        "symptoms": SYMPTOM_SEP.join(p["symptoms"]),
        "diagnosis": p["diagnosis"],
        "tests_done": SYMPTOM_SEP.join(p["tests_done"]),
        "test_results": p["test_results"],
        "medicines": SYMPTOM_SEP.join(p["medicines"]),
        "outcome": p["outcome"],
    } for p in patients if p.get("source") == "uploaded"]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS).to_csv(index=False)


def build_csv_template():
    """A small example CSV users can download from Doctor mode."""
    rows = [
        {"patient_id": "P-001", "age": 28, "gender": "male", "city": "Mumbai",
         "visit_month": 8,
         "symptoms": "High fever (39°C / 102°F or above);Pain behind the eyes;Body aches / muscle pain;Nausea",
         "diagnosis": "Dengue Fever",
         "tests_done": "Dengue NS1 antigen (blood);CBC with platelet count (daily while feverish)",
         "test_results": "NS1 positive; platelets 95,000",
         "medicines": "Paracetamol;ORS",
         "outcome": "Recovered in 9 days"},
        {"patient_id": "P-002", "age": 41, "gender": "female", "city": "Delhi",
         "visit_month": 10,
         "symptoms": "Fever;Fatigue or weakness;Abdominal pain or cramps;Loss of appetite",
         "diagnosis": "Typhoid Fever",
         "tests_done": "Widal test (blood);CBC (complete blood count)",
         "test_results": "Widal positive 1:160",
         "medicines": "Azithromycin;Paracetamol",
         "outcome": "Recovered in 12 days on antibiotics"},
        {"patient_id": "P-003", "age": 34, "gender": "female", "city": "Pune",
         "visit_month": 5,
         "symptoms": "Burning when urinating;Frequent urination;Cloudy or strong-smelling urine",
         "diagnosis": "Urinary Tract Infection (UTI)",
         "tests_done": "Urine routine and microscopy;Urine culture and sensitivity",
         "test_results": "Pus cells plenty; E. coli sensitive to nitrofurantoin",
         "medicines": "Nitrofurantoin",
         "outcome": "Recovered in 5 days"},
        {"patient_id": "P-004", "age": 62, "gender": "male", "city": "Kolkata",
         "visit_month": 1,
         "symptoms": "Cough with phlegm;Fever;Shortness of breath;Chest pain when breathing",
         "diagnosis": "Pneumonia",
         "tests_done": "Chest X-ray;CBC and CRP (blood)",
         "test_results": "Consolidation right lower lobe; CRP 96",
         "medicines": "Amoxicillin-clavulanate",
         "outcome": "Admitted 4 days, recovered"},
        {"patient_id": "P-005", "age": 23, "gender": "male", "city": "Chennai",
         "visit_month": 11,
         "symptoms": "Severe joint pain (hands and feet);High fever (39°C / 102°F or above);Headache",
         "diagnosis": "Chikungunya",
         "tests_done": "Chikungunya IgM rapid test or RT-PCR (blood);Dengue NS1 antigen (blood)",
         "test_results": "Chikungunya IgM positive; NS1 negative",
         "medicines": "Paracetamol",
         "outcome": "Fever settled in 5 days; joint pain for 3 weeks"},
    ]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS).to_csv(index=False)


# ---- 3) training and prediction --------------------------------------------


def patient_feature_row(symptoms, age_band, gender, month):
    """One patient -> one dictionary of 0/1 columns. Absent keys become 0 later.
    This is 'feature engineering' - deciding what the model is allowed to see."""
    row = {}
    for s in symptoms:
        row[f"sym:{s}"] = 1.0
    if age_band:
        row[f"age:{age_band}"] = 1.0
    if gender:
        row[f"sex:{gender}"] = 1.0
    if month:
        if month in (12, 1, 2):
            row["season:winter"] = 1.0
        elif month in (3, 4, 5):
            row["season:spring"] = 1.0
        elif month in (6, 7, 8):
            row["season:summer"] = 1.0
        else:
            row["season:autumn"] = 1.0
        if month in (6, 7, 8, 9, 10):          # India's monsoon window
            row["season:monsoon"] = 1.0
    return row


def train_ml_model(patients):
    """Train the LogisticRegression classifier on a list of patient records.
    Returns a 'model bundle' dictionary with everything needed to predict."""
    if len(patients) < 20:
        return None
    rows = [patient_feature_row(p["symptoms"], age_to_band(p["age"]),
                                p["gender"], p["month"]) for p in patients]
    X = pd.DataFrame(rows).fillna(0.0)
    y = [p["diagnosis"] for p in patients]

    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced",
                             random_state=42)
    counts = Counter(y)
    if len(counts) >= 2 and min(counts.values()) >= 4 and len(patients) >= 80:
        # Enough data for an honest test: hide 20% and see if the model is right.
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        clf.fit(X_tr, y_tr)
        accuracy = float(clf.score(X_te, y_te))
        acc_note = "measured on 20% of records the model never saw during training"
    else:
        clf.fit(X, y)
        accuracy = float(clf.score(X, y))
        acc_note = "measured on the training records themselves (upload more data for an honest number)"

    return {
        "clf": clf,
        "columns": list(X.columns),
        "n_rows": len(patients),
        "n_uploaded": sum(1 for p in patients if p.get("source") == "uploaded"),
        "labels": list(clf.classes_),
        "accuracy": accuracy,
        "accuracy_note": acc_note,
        "symptom_vocab": sorted({s for p in patients for s in p["symptoms"]}),
    }


def predict_ml(model, symptoms, age_band, gender, month):
    """The user's answers -> {disease: probability} from the learned model."""
    if not model:
        return {}
    row = pd.DataFrame([patient_feature_row(symptoms, age_band, gender, month)]).fillna(0.0)
    row = row.reindex(columns=model["columns"], fill_value=0.0)
    probs = model["clf"].predict_proba(row)[0]
    return {label: float(p) for label, p in zip(model["clf"].classes_, probs)}


def explain_ml_features(model, disease, symptoms):
    """Which of the user's symptoms pushed this disease's probability up?
    Reads the coefficients LogisticRegression learned - this is what makes the
    ML half explainable instead of a black box."""
    try:
        clf = model["clf"]
        classes = list(clf.classes_)
        if disease not in classes:
            return []
        weights = clf.coef_[classes.index(disease)]
        col_index = {c: i for i, c in enumerate(model["columns"])}
        pairs = []
        for s in symptoms:
            key = f"sym:{s}"
            if key in col_index:
                w = float(weights[col_index[key]])
                if w > 0.05:
                    pairs.append((s, w))
        pairs.sort(key=lambda t: -t[1])
        return pairs[:3]
    except Exception:
        return []


# ---- 4) similar patients & learned test suggestions ------------------------


def find_similar_patients(symptoms, patients, top_k=6, min_sim=0.25):
    """Cosine similarity between the user's symptom set and every stored
    patient record. In plain words: 'past patients who complained of almost
    the same things'. Used to explain the ML score and to suggest tests."""
    user = set(symptoms)
    if not user:
        return []
    scored = []
    for p in patients:
        ps = set(p["symptoms"])
        if not ps:
            continue
        overlap = user & ps
        if not overlap:
            continue
        sim = len(overlap) / ((len(user) ** 0.5) * (len(ps) ** 0.5))
        if sim >= min_sim:
            scored.append((sim, p, sorted(overlap)))
    scored.sort(key=lambda t: (-t[0], t[1].get("id", "")))
    return [{"sim": s, "patient": p, "overlap": o} for s, p, o in scored[:top_k]]


def aggregate_tests(similar, diagnosis):
    """Among similar past patients who had this diagnosis, which tests did
    they actually have done? Returns ([(test, count)], n_patients_with_dx)."""
    counts, total = {}, 0
    for item in similar:
        p = item["patient"]
        if p["diagnosis"] != diagnosis:
            continue
        total += 1
        for t in p.get("tests_done") or []:
            if t:
                counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1]), total


# ---- 5) lab findings refinement --------------------------------------------

NEGATION_WORDS = ("negative", "not positive", "absent", "ruled out")


def _phrase_negated(text, phrase):
    """True if a negation word appears shortly AFTER the phrase - so
    'ns1 antigen negative' or 'malaria rapid test: negative' count as
    NEGATIVE findings even though the keyword itself matched."""
    idx = text.find(phrase)
    while idx != -1:
        window = text[idx + len(phrase): idx + len(phrase) + 28]
        if any(w in window for w in NEGATION_WORDS):
            return True
        idx = text.find(phrase, idx + 1)
    return False


def _normalize_lab_text(text):
    """Plain-language tweaks so numbers people actually type match the
    finding phrases: 'platelets are 85,000' or 'platelet count 85' both
    mean LOW platelets, but only 'low platelet' is a lookup phrase."""
    t = re.sub(r"\s+", " ", str(text or "").lower())
    m = re.search(r"platelets?\D{0,15}(\d{1,3}(?:,\d{3})+|\d{4,6}|\d{1,3})", t)
    if m:
        num = float(m.group(1).replace(",", ""))
        if 5 <= num < 150_000:          # '85' means 85,000/µl, '85,000' too
            t += " low platelet"
    return t


def apply_lab_findings(results, findings_text):
    """Adjust the hybrid scores using typed/picked lab findings.
    Each result dict gains: score_final, lab_reasons, lab_confirmed."""
    text = " " + _normalize_lab_text(findings_text) + " "
    for r in results:
        rules = CONFIRMING_FINDINGS.get(r["name"], [])
        matched = [rule for rule in rules if rule["phrase"] in text]
        # Keep only the most specific match: if "ns1 negative" matched, drop
        # the shorter "ns1" that also matched inside it.
        final = []
        for m in matched:
            covered = any(m["phrase"] in o["phrase"] and len(o["phrase"]) > len(m["phrase"])
                          for o in matched if o is not m)
            if not covered:
                final.append(m)
        adjust, reasons = 0.0, []
        for m in final:
            points, label = m["points"], m["label"]
            # A positive keyword reported as negative flips its sign:
            # 'NS1 antigen negative' should penalise, not boost.
            if points > 0 and _phrase_negated(text, m["phrase"]):
                points, label = -points, f"{label} (reported as negative)"
            adjust += points
            reasons.append(f"Lab finding: {label} ({'+' if points >= 0 else ''}{int(points * 100)}%)")
        r["score_final"] = min(max(r["score"] + adjust, 0.02), 0.99)
        r["lab_reasons"] = reasons
        r["lab_confirmed"] = adjust >= 0.15
    results.sort(key=lambda r: r.get("score_final", r["score"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# PART 4B - THE CONVERSATION BRAIN (v3.0 - "talk to drSaab like a human")
# ---------------------------------------------------------------------------
# A checkbox-only app forces patients to translate their story into clean
# labels. Real people say "my lower back hurts since a deadlift session
# without a belt, is it sciatica or a disc bulge?" This part understands
# that sentence:
#
#   1. UNDERSTAND  - intent detection (greeting / symptoms / lab results /
#                    emergency / question about a named condition / thanks),
#                    symptom extraction from free text (alias dictionary +
#                    regex patterns + fuzzy typo matching), negation
#                    ("no fever"), duration ("for 3 days"), severity,
#                    city mentions, and named conditions ("sciatica").
#   2. THINK       - the extracted symptoms feed the SAME hybrid engine as
#                    picker mode (rules + ML + internet evidence), via
#                    chat_turn() below.
#   3. TALK        - doctor-style replies: acknowledge first, show the
#                    reasoning briefly, ask ONE focused follow-up question
#                    (and remember the answer), suggest tests, warn about
#                    red flags, and hand over the full report below the chat.
#
# chat_turn(text, S) works on any MutableMapping S (st.session_state in the
# app, a plain dict in the tests) - that is what keeps it testable.

import difflib

# --- vocabulary -------------------------------------------------------------

# Single words / inflected forms that the alias table misses. Mapped with
# regex, so "dizzy", "nauseous" and "puking" all work.
CHAT_SYMPTOM_PATTERNS = [
    (r"\bdizzy\b|\bdizziness\b|\bgiddiness\b|\blightheaded\b", "Dizziness"),
    (r"\bnauseous\b|\bnauseated\b|\bfeeling sick\b", "Nausea"),
    (r"\bvomiting\b|\bvomited\b|\bvomit\b|\bthrowing up\b|\bthrew up\b|\bpuking\b|\bpuked\b", "Vomiting"),
    (r"\bdiarrhea\b|\bdiarrhoea\b|\bloose motions?\b|\bloose stools?\b", "Diarrhea"),
    (r"\bfeverish\b|\btemperature\b", "Fever"),
    (r"\bcoughing\b|\bcough(ed)?\b", "Dry cough"),
    (r"\bheadaches?\b", "Headache"),
    (r"\bfatigued\b|\btired(ness)?\b|\bexhausted\b|\bno energy\b", "Fatigue or weakness"),
    (r"\bbreathless\b|\bbreathlessness\b|\bshort(ness)? of breath\b|\bdifficulty breathing\b|\bcan ?not breathe\b|\bcant breathe\b|\bstruggling to breathe\b", "Shortness of breath"),
    (r"\bwheez(e|ing)\b", "Wheezing"),
    (r"\bsneeze(d|s)?\b|\bsneezing\b", "Sneezing"),
    (r"\brunny nose\b", "Runny nose"),
    (r"\bblocked nose\b|\bstuffy nose\b|\bcongested?\b", "Blocked nose"),
    (r"\bsore throat\b|\bthroat hurts\b|\bthroat pain\b", "Sore throat"),
    (r"\bbody ach(es|e)\b|\bbody pain\b|\bmuscle ach(es|e)\b|\bmuscle pain\b", "Body aches / muscle pain"),
    (r"\bchills?\b|\bshivering\b", "Chills or shivering"),
    (r"\bnight sweats?\b", "Night sweats"),
    (r"\bchest pain\b|\bchest hurts\b|\bchest tight(ness)?\b", "Chest pain when breathing"),
    (r"\brash(es)?\b", "Itchy rash"),
    (r"\bitch(ing|y)?\b", "Itchy rash"),
    (r"\bthirsty\b|\bexcessive thirst\b", "Excessive thirst"),
    (r"\bheartburn\b|\bacidity\b|\bacid reflux\b", "Heartburn / burning behind the chest"),
    (r"\bbloating\b|\bloated\b", "Bloating"),
    (r"\bbelching\b|\bburping\b|\bburps?\b", "Belching or burping"),
    (r"\bstomach (pain|ache|hurts)\b|\btummy (pain|ache)\b|\babdominal pain\b", "Abdominal pain or cramps"),
    (r"\bloss of appetite\b|\bnot eating\b|\bcannot eat\b", "Loss of appetite"),
    (r"\beye pain\b|\beyes hurt\b", "Pain behind the eyes"),
    (r"\bear (pain|ache|hurts)\b", "Ear pain"),
    (r"\bjoint pain\b|\bjoints hurt\b|\bjoint pains?\b", "Joint pain"),
    (r"\bweak(ness)?\b", "Fatigue or weakness"),
    (r"\bnumb(ness)?\b|\btingling\b|\bpins and needles\b", "Numbness or tingling in the leg or foot"),
    (r"\bback pain\b|\bbackache\b|\bback hurts\b|\bhurt (my|the) back\b", "Lower back pain"),
    (r"\bshooting pain\b|\bshoots down\b|\bradiates? (down|to)\b|\bgoes? down (my|the) leg\b", "Pain shooting down the leg (below the knee)"),
    # Hinglish - how a huge share of Indian patients actually talk:
    (r"\bbukhar\b|\bbukhaar\b", "Fever"),
    (r"\bsar dard\b|\bsar mein dard\b|\bsir dard\b|\bheadache hai\b", "Headache"),
    (r"\bzukam\b|\bzukaam\b", "Runny nose"),
    (r"\bkhansi\b|\bkhansi hai\b", "Dry cough"),
    (r"\bchakkar\b", "Dizziness"),
    (r"\bdast\b", "Diarrhea"),
    (r"\bulti\b|\bulti aa rahi\b", "Vomiting"),
    (r"\bpet dard\b|\bpet mein dard\b", "Abdominal pain or cramps"),
    (r"\bgale mein dard\b|\bgala kharab\b", "Sore throat"),
    (r"\bthakan\b|\bkamzori\b", "Fatigue or weakness"),
    (r"\bbadan dard\b", "Body aches / muscle pain"),
    (r"\bpilia\b|\bpiliya\b", "Yellow eyes or skin (jaundice)"),
]

# Conditions people name in conversation. The bare words "cold" and "fever"
# are excluded on purpose (they are usually symptoms, and "cold weather"
# would false-positive).
CHAT_NAMED_EXCLUDE = {"cold", "fever", "viral fever"}

# Follow-up questions asked in a human voice (one at a time, answer
# remembered). Anything not listed falls back to a generic phrasing.
FOLLOWUP_HINTS = {
    "Pain shooting down the leg (below the knee)":
        "does the pain shoot from your back down below the knee - into your calf or foot?",
    "Numbness or tingling in the leg or foot":
        "any numbness or pins-and-needles in the leg or foot?",
    "Weakness in the leg or foot":
        "does the leg or foot feel weak at all (for example, trouble standing on your toes)?",
    "Back pain worse when sitting or bending forward":
        "is it worse when you sit for a while or bend forward?",
    "Back pain worse when coughing or sneezing":
        "does coughing or sneezing send a jolt of pain through your back?",
    "Fever":
        "have you measured a fever as well?",
    "High fever (39°C / 102°F or above)":
        "how high has the fever been - above 102°F / 39°C?",
    "Dry cough":
        "is there a cough too - dry or with phlegm?",
    "Cough with phlegm":
        "is the cough dry, or are you bringing up phlegm?",
    "Shortness of breath":
        "are you getting breathless - for example on climbing stairs?",
    "Nausea":
        "any nausea or vomiting with it?",
    "Vomiting":
        "have you been vomiting as well?",
    "Body aches / muscle pain":
        "any body aches or muscle pain with it?",
    "Headache":
        "is there a headache too?",
    "Chills or shivering":
        "do you get chills or shivering when the fever spikes?",
    "Loss of taste or smell":
        "has your sense of taste or smell changed?",
    "Sore throat":
        "how does your throat feel - sore or scratchy?",
    "Diarrhea":
        "any loose motions along with it?",
    "Rash with small fluid-filled blisters":
        "any rash or blisters on the skin?",
    "Unintentional weight loss":
        "have you lost weight without trying?",
    "Night sweats":
        "do you sweat a lot at night?",
    "Burning when urinating":
        "any burning when you pass urine?",
}

# Emergency patterns: (regex, plain-language reason). Checked FIRST - if one
# fires, drSaab stops playing symptom-checker and tells the patient to get
# real help now.
EMERGENCY_PATTERNS = [
    (r"chest (pain|pressure|tightness).{0,60}(breathless|short(ness)? of breath|sweat|left arm|can ?not breathe|cant breathe)"
     r"|(breathless|short(ness)? of breath|can ?not breathe|cant breathe).{0,60}chest (pain|pressure|tightness)",
     "Chest pain together with breathlessness or sweating can be a heart attack"),
    (r"\b(can ?not|cant|unable to|struggling to|difficulty) breathe\b|\bsevere breathless\b",
     "Severe breathing difficulty is an emergency"),
    (r"\bunconscious\b|\bfainted\b|\bpassed out\b|\bnot waking up\b|\bblue lips\b",
     "Losing consciousness or fainting needs emergency assessment"),
    (r"\bseizure\b|\bconvulsion\b|\bfits\b|\bfitting\b",
     "A seizure needs emergency care"),
    (r"\bvomiting blood\b|\bblood in vomit\b|\bcoughing (up )?(a lot of )?blood\b|\bhemoptysis\b",
     "Coughing or vomiting blood is an emergency"),
    (r"\bstiff neck\b.{0,50}fever|fever.{0,50}\bstiff neck\b",
     "Fever with a stiff neck can be meningitis - that is an emergency"),
    (r"\b(face drooping|slurred speech|one ?sided weakness|arm is weak|sudden weakness on one side)\b",
     "Sudden one-sided weakness or slurred speech can be a stroke"),
    (r"\b(suicide|suicidal|kill myself|end my life|want to die|hurt myself)\b",
     "What you are feeling right now matters more than any symptom checker"),
    (r"\boverdose\b|\btook too many (pills|tablets)\b",
     "A possible overdose is an emergency"),
    (r"\b(can ?not|cant|lost?|no) (pass |urinate |control (my )?)?(urine|pee|bladder|bowel|stools?)\b"
     r"|\blost control of (my )?(bladder|bowel)s?\b",
     "Losing control of bladder or bowels - with back pain this can be cauda equina syndrome"),
]

GREETING_RE = re.compile(r"^\s*(hi|hii+|hey+|hello+|namaste|namaskar|good (morning|afternoon|evening)|salaam|assalam[ou] ?alaikum)"
                         r"\s*[!.,? ]*$", re.I)
GREETING_WORD_RE = re.compile(r"\b(hi|hey|hello|namaste|good (morning|afternoon|evening)|salaam)\b", re.I)
THANKS_RE = re.compile(r"\b(thanks?|thank you|thx|ty|shukriya|dhanyavad|great|awesome|helpful|nice)\b", re.I)
BYE_RE = re.compile(r"\b(bye|goodbye|see you|good night|take care)\b", re.I)
YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|y|haan|ha|haa|correct|right|true|obviously|i do|i did|it does|sometimes|kind of|kinda)\s*[!.,? ]*$", re.I)
NO_RE = re.compile(r"^\s*(no|nope|nah|nahi|not really|never|false|not at all)\s*[!.,? ]*$", re.I)
LAB_HINT_RE = re.compile(r"\b(ns1|igg|igm|widal|platelets?|cbc|positive|negative|report|results?|x ?ray|mri|scan|usg|ultrasound|blood test|slr|esr|crp|sgpt|sgot|bilirubin|creatinine|culture|stool test|urine test|came back)\b", re.I)
REPORT_REQUEST_RE = re.compile(r"\b(full report|show (me )?(the )?report|show details|details|medicines?|medicine list|hospital|pharmacy|treatment options?)\b", re.I)

# Words that cancel a nearby negation ("no fever BUT body aches").
NEG_BREAKERS = {"but", "however", "though", "although", "except", "yet"}
CHAT_NEGATIONS = {"no", "not", "never", "dont", "doesnt", "didnt", "without",
                  "denies", "deny", "havent", "hasnt", "wasnt", "isnt",
                  "arent", "werent", "none", "nahi", "nope"}


def _chat_norm(text):
    """Lower-case, strip apostrophes (can't -> cant) and punctuation."""
    t = str(text or "").lower()
    t = t.replace("\u2019", "").replace("\u2018", "").replace("'", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _phrase_lookup():
    """alias phrases + canonical symptom names -> canonical symptom."""
    lookup = dict(SYMPTOM_ALIASES)
    for s in SYMPTOMS:
        lookup.setdefault(_chat_norm(s), s)
    return lookup


def _named_lookup():
    """alias phrases + canonical disease names -> disease, minus chatty words."""
    lookup = {k: v for k, v in DIAGNOSIS_ALIASES.items() if k not in CHAT_NAMED_EXCLUDE}
    for name in DISEASES:
        lookup.setdefault(_chat_norm(name), name)
    return lookup


def _find_phrase_spans(text, lookup):
    """Longest-first, non-overlapping phrase matching. Returns [(start, end, value)]."""
    spans = []
    for phrase in sorted(lookup, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(phrase, start)
            if i == -1:
                break
            j = i + len(phrase)
            boundary_ok = (i == 0 or text[i - 1] == " ") and (j >= len(text) or text[j] == " ")
            if boundary_ok and not any(i < e and s < j for s, e, _ in spans):
                spans.append((i, j, lookup[phrase]))
            start = i + 1
    return sorted(spans)


def _has_negation_before(text, pos):
    """True if a negation word sits within ~30 chars before pos (with no
    'but/however' breaking it). 'cannot bend forward' is NOT negation -
    'cannot' is not in the list on purpose."""
    window = text[max(0, pos - 30):pos].split()
    for idx, w in enumerate(window):
        if w in CHAT_NEGATIONS:
            if any(b in window[idx + 1:] for b in NEG_BREAKERS):
                return False
            return True
    return False


def _fuzzy_symptoms(text, covered, lookup):
    """Typo tolerance: 1-2 word windows not already matched get a difflib
    second chance ('dic bulge' -> 'disc bulge', 'nuasea' -> 'nausea')."""
    found = []
    tokens = [(m.start(), m.group()) for m in re.finditer(r"[a-z0-9]+", text)]

    def _try(window):
        # short words need a stricter cutoff, long ones tolerate more editing
        cutoff = 0.85 if len(window) <= 5 else 0.80
        hit = difflib.get_close_matches(window, lookup.keys(), n=1, cutoff=cutoff)
        return lookup[hit[0]] if hit else None

    for k, (pos, tok) in enumerate(tokens):
        if len(tok) < 4 or any(pos < e and s <= pos < e for s, e in covered):
            continue
        candidates = [tok]
        if k + 1 < len(tokens) and tokens[k + 1][0] == pos + len(tok) + 1:
            two = tok + " " + tokens[k + 1][1]
            if len(tok) + len(tokens[k + 1][1]) >= 9:
                candidates.append(two)
        for window in candidates:
            if len(window) < 4:
                continue
            target = _try(window)
            if target and target not in found:
                found.append(target)
                break
    return found


def _extract_symptoms(text):
    """All canonical symptoms mentioned in normalised text, split into
    positive and negated."""
    norm = _chat_norm(text)
    lookup = _phrase_lookup()
    spans = _find_phrase_spans(norm, lookup)
    positive, negated, covered = [], [], [(s, e) for s, e, _ in spans]
    for s, e, name in spans:
        if name in positive or name in negated:
            continue
        (negated if _has_negation_before(norm, s) else positive).append(name)
    # regex patterns on the words not already covered
    uncovered = " ".join(tok for pos, tok in
                         [(m.start(), m.group()) for m in re.finditer(r"[a-z0-9]+", norm)]
                         if not any(pos < e and s <= pos < e for s, e in covered))
    for pattern, name in CHAT_SYMPTOM_PATTERNS:
        m = re.search(pattern, uncovered)
        if m and name not in positive and name not in negated:
            if _has_negation_before(uncovered, max(mm.start() for mm in re.finditer(pattern, uncovered))):
                negated.append(name)
            else:
                positive.append(name)
    # fuzzy pass for typos
    for name in _fuzzy_symptoms(norm, covered, lookup):
        if name not in positive and name not in negated:
            positive.append(name)
    # "fever of 102" / "temperature 102.4" -> high vs normal fever
    m = re.search(r"(?:fever|temperature|temp)\D{0,12}(\d{2,3}(?:\.\d)?)", norm)
    if m and "Fever" in positive:
        v = float(m.group(1))
        if v >= 39 or 100 <= v <= 106:
            positive.remove("Fever")
            if "High fever (39°C / 102°F or above)" not in positive:
                positive.append("High fever (39°C / 102°F or above)")
    return positive, negated


def _extract_duration(text):
    """Duration in days from phrases like '3 days', 'two weeks', 'since
    yesterday', '4 din se', 'a week now'."""
    t = _chat_norm(text)
    m = re.search(r"(\d{1,3})\s*(day|din|dins|days)\b", t)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"(\d{1,2})\s*(week|weeks|hafta|haftay)\b", t)
    if m:
        return max(1, int(m.group(1)) * 7)
    m = re.search(r"(\d{1,2})\s*(month|months|mahina|mahine)\b", t)
    if m:
        return max(1, int(m.group(1)) * 30)
    words = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "couple": 3, "few": 4, "several": 6}
    m = re.search(r"(one|two|three|four|five|couple|few|several|a|an)\s+(day|days|week|weeks)\b", t)
    if m:
        base = words.get(m.group(1), 1)
        return base * 7 if "week" in m.group(2) else base
    if re.search(r"(since )?yesterday|last night", t):
        return 1
    if re.search(r"since (this )?morning|today\b", t):
        return 1
    return None


def _extract_severity(text):
    t = _chat_norm(text)
    if re.search(r"\b(unbearable|excruciating|severe|worst|extreme|horrible|intense|very bad|terrible|agony)\b", t):
        return "severe"
    if re.search(r"\b(mild|slight|little|minor|manageable|not too bad)\b", t):
        return "mild"
    return None


def _extract_city(text):
    t = _chat_norm(text)
    for city in INDIA_CITIES:
        if re.search(rf"\b{_chat_norm(city)}\b", t):
            return city
    return None


def _extract_named_conditions(text):
    norm = _chat_norm(text)
    lookup = _named_lookup()
    named = []
    for _s, _e, name in _find_phrase_spans(norm, lookup):
        if name not in named:
            named.append(name)
    # typo-tolerant pass for condition names ('sicatica', 'herination')
    for w in re.findall(r"[a-z]{5,}", norm):
        hit = difflib.get_close_matches(w, [k for k in lookup if len(k) >= 5], n=1, cutoff=0.86)
        if hit and lookup[hit[0]] not in named:
            named.append(lookup[hit[0]])
    return named[:4]


def _detect_emergency(text):
    t = _chat_norm(text)
    for pattern, reason in EMERGENCY_PATTERNS:
        if re.search(pattern, t):
            return reason
    return None


def understand(text):
    """Turn one human message into a structured understanding."""
    symptoms, negated = _extract_symptoms(text)
    return {
        "raw": text,
        "symptoms": symptoms,
        "negated": negated,
        "named": _extract_named_conditions(text),
        "duration": _extract_duration(text),
        "severity": _extract_severity(text),
        "city": _extract_city(text),
        "emergency": _detect_emergency(text),
        "is_greeting_only": bool(GREETING_RE.match(text.strip())) or
                            (bool(GREETING_WORD_RE.search(text)) and len(text.split()) <= 4
                             and not symptoms and not LAB_HINT_RE.search(text)),
        "is_thanks": bool(THANKS_RE.search(text)) and not symptoms and not negated,
        "is_bye": bool(BYE_RE.search(text)) and not symptoms and not negated,
        "is_lab": bool(LAB_HINT_RE.search(text)) and not symptoms and not negated,
        "wants_report": bool(REPORT_REQUEST_RE.search(text)) and not symptoms,
    }


# --- the thinking part ------------------------------------------------------

def _chat_engine_inputs(S):
    """Read the sidebar/engine settings from the state mapping (works both in
    the app and in tests)."""
    ik = S.get("internet_knowledge")
    net_on = S.get("net_toggle", S.get("Internet knowledge check (Wikipedia + Wikidata)",
                                       ik is not None))
    net_w = (S.get("net_weight_sel", S.get("Weight given to internet evidence", 25))
             if (net_on and ik is not None) else 0)
    return {
        "ml_weight": S.get("ml_weight_sel", S.get("Weight given to the learned model", 50)),
        "model": S.get("ml_model"),
        "patients": S.get("seed_patients", []) + S.get("extra_patients", []),
        "net_weight": net_w if (net_on and ik is not None) else 0,
        "knowledge": ik,
        "school": S.get("school") if isinstance(S, dict) or hasattr(S, "get") else None,
    }


def _chat_build_report(symptoms, S, days=None, city=None):
    eng = _chat_engine_inputs(S)
    report = build_report(
        symptoms,
        city if city is not None else (S.get("city_input", "") or ""),
        S.get("age_sel", "Adult"),
        days if days is not None else S.get("days_sel", 2),
        S.get("gender_sel", "Not saying"),
        eng["ml_weight"], eng["model"], eng["patients"],
        eng["net_weight"], eng["knowledge"], school=eng.get("school"))
    lab = S.get("lab_findings_text", "")
    if lab:
        apply_lab_findings(report["results"], lab)
    S["report"] = report
    return report


def _fmt_pct(x):
    return f"{round(float(x) * 100)}%"


def _one_liner(r, report):
    """One sentence of honest reasoning for a ranked condition."""
    parts = []
    if r["matched"]:
        parts.append("I'm weighing " + ", ".join(s.lower() for s in r["matched"][:3]))
    if r.get("ml_reason"):
        parts.append("the learned model agrees")
    if report.get("internet_on") and r.get("net_conf") is not None:
        n = len(r.get("net_confirmed") or [])
        parts.append(f"{n} of your symptoms are confirmed in live online medical knowledge" if n
                     else "online sources did not confirm this one")
    if r["reasons"]:
        parts.append(r["reasons"][0][0].lower() + r["reasons"][0][1:])
    if not parts:
        parts.append("your overall pattern points this way")
    return "; ".join(parts) + "."


def _top_missing(condition_name, known_symptoms, denied, k=2):
    d = DISEASES.get(condition_name)
    if not d:
        return []
    return [s for s, w in sorted(d["symptoms"].items(), key=lambda kv: -kv[1])
            if s not in known_symptoms and s not in denied and w >= 2][:k]


def _ask_followup(S, symptoms, denied):
    """Pick ONE missing key symptom of the leading condition and phrase it as
    a doctor's question. Remembers it so a bare 'yes'/'no' makes sense."""
    pending = S.get("chat_pending_q")
    if pending and pending.get("symptom") not in symptoms:
        return pending
    month = datetime.now().month
    ranked = score_diseases(symptoms, {"temp": None}, month, None)
    for r in ranked:
        missing = _top_missing(r["name"], symptoms, denied, k=1)
        if missing:
            s = missing[0]
            q = FOLLOWUP_HINTS.get(s, f"are you also experiencing {s.lower()}?")
            S["chat_pending_q"] = {"symptom": s, "question": q, "condition": r["name"]}
            return S["chat_pending_q"]
    return None


def _analysis_reply(S, named, focus_named=True):
    """The doctor-style brief after the hybrid engine has run."""
    report = S.get("report") or {}
    results = report.get("results") or []
    if not results:
        return ("I couldn't narrow that down to anything I know confidently. Could you tell me "
                "a bit more - where exactly it hurts, since when, and anything else you feel "
                "along with it? And if you feel genuinely unwell, please see a doctor regardless "
                "of what I say.")
    symptoms = report["symptoms"]
    n_things = f"{len(symptoms)} thing{'s' if len(symptoms) > 1 else ''}"
    opener = _pick([
        f"Here's my read on it, based on the {n_things} you've told me so far:",
        f"Okay, I've gone through everything. With the {n_things} you've described:",
        f"Alright, I've thought about this. Putting your {n_things} together:",
        f"Right - based on what you've told me ({n_things} so far), here's my thinking:",
    ])
    lines = [opener, ""]
    for i, r in enumerate(results[:3], 1):
        score = r.get("score_final", r["score"])
        badge = "lab-supported" if r.get("lab_confirmed") else match_label(score).lower()
        lines.append(f"**{i}. {r['name']}** - {_fmt_pct(score)} ({badge})")
        lines.append(f"   {_one_liner(r, report)}")
    lines.append("")

    # answer a directly-named question ("is it sciatica or a disc bulge?")
    if focus_named and named:
        top_names = [r["name"] for r in results[:3]]
        for c in named[:2]:
            if c not in DISEASES:
                continue
            pct = next((_fmt_pct(r.get("score_final", r["score"])) for r in results if r["name"] == c), None)
            tests = [t for t, _ in DISEASES[c].get("typical_tests", [])
                     if not _norm_key(t).startswith(("no test", "usually no"))][:1]
            test_txt = f" An **{tests[0]}** is what settles it." if tests else ""
            if pct:
                lines.append(f"You asked about **{c}** - yes, it's on my list ({pct}).{test_txt}")
            else:
                missing = _top_missing(c, symptoms, S.get("chat_denied", []), k=1)
                why = (f" what you've described doesn't strongly point that way yet - classically it needs "
                       f"**{missing[0].lower()}**.") if missing else " your current picture doesn't point that way."
                lines.append(f"You asked about **{c}** -{why}{test_txt}")
        lines.append("")

    # tests + red flags of the leading condition
    top = results[0]
    d = DISEASES.get(top["name"])
    if d:
        tests = [t for t, _ in d.get("typical_tests", [])
                 if not _norm_key(t).startswith(("no test", "usually no"))][:2]
        if tests:
            lines.append("**What would help settle it:** " +
                         "; ".join(f"**{t.lower()}**" for t in tests) + ".")
        if d["red_flags"]:
            lines.append(f"**Watch out for:** {d['red_flags'][0].lower()}")
    lines.append("The full report is just below this chat - medicines, home care and nearby hospitals included.")

    q = _ask_followup(S, symptoms, S.get("chat_denied", []))
    if q and q["symptom"] not in symptoms:
        lines.append("")
        lines.append(_pick([
            "One more question so I can sharpen this: ",
            "To narrow this down further: ",
            "Help me narrow it down - one more question: ",
            "Just one more thing would help me here: ",
        ]) + q["question"])
    return "\n".join(lines)


def _lab_reply(S, text):
    """A lab-result message: remember the findings, re-rank, show the change."""
    prev = [(r["name"], r.get("score_final", r["score"]))
            for r in (S.get("report") or {}).get("results", [])]
    existing = S.get("lab_findings_text", "")
    S["lab_findings_text"] = (existing + "\n" + text).strip()
    if not S.get("chat_symptoms"):
        return ("Noted - I've kept that for later. But first tell me what you're feeling "
                "(your symptoms), so I have something to match the findings against.")
    report = _chat_build_report(S.get("chat_symptoms"), S)
    new = [(r["name"], r.get("score_final", r["score"])) for r in report["results"]]
    lines = ["Got it - findings noted. Here's how that changes my ranking:", ""]
    old_scores = dict(prev)
    for i, (name, score) in enumerate(new[:3], 1):
        old = old_scores.get(name)
        arrow = ""
        if old is not None:
            arrow = " (up from " + _fmt_pct(old) + ")" if score > old + 0.005 else \
                    (" (down from " + _fmt_pct(old) + ")" if score < old - 0.005 else " (unchanged)")
        lines.append(f"**{i}. {name}** - {_fmt_pct(score)}{arrow}")
    lines.append("")
    lines.append("The updated full report is below this chat. Do the findings match what was "
                 "actually tested? Tell me more and I'll refine further.")
    return "\n".join(lines)


def _named_condition_reply(S, named, u):
    """The user asked about a condition by name but hasn't described symptoms."""
    lines = []
    if len(named) == 1:
        c = named[0]
        d = DISEASES.get(c)
        if not d:
            return (f"I don't have **{c}** in my knowledge base yet, so I can't analyse it "
                    "responsibly. A doctor is the right person for that one. "
                    "(If you have a book or notes about it, upload them in **Medical school** "
                    "mode - I'll study it and add it to my knowledge.)")
        short = d["about"].split(". ")[0].rstrip(".") + "."
        tests = [t for t, _ in d.get("typical_tests", [])
                 if not _norm_key(t).startswith(("no test", "usually no"))][:2]
        lines.append(f"**{c}** - here's the short version:")
        lines.append(f"   {short}")
        if tests:
            lines.append(f"   Confirmed by: {'; '.join(t.lower() for t in tests)}.")
        lines.append("")
        lines.append("Now tell me what you're actually feeling - in your own words. For example: "
                     "where it hurts, since when, and anything that travels to another part of the body.")
    else:
        lines.append("Good question - let me separate those for you:")
        for c in named[:2]:
            d = DISEASES.get(c)
            if not d:
                continue
            short = d["about"].split(". ")[0].rstrip(".") + "."
            lines.append(f"- **{c}** - {short}")
        lines.append("")
        lines.append("The tricky part is that these can overlap - the story you tell matters more "
                     "than any single symptom. So: **what exactly are you feeling right now?** "
                     "Where does it hurt, and does anything travel - for example down a leg?")
    return "\n".join(lines)


def _greeting_reply(S):
    variants = [
        "Hello! I'm **drSaab** 🩺 Tell me what's been going on - in your own words, like you'd tell "
        "a doctor. I understand plain sentences, Hinglish, typos and even lab reports.",
        "Hey, good to see you! What's been bothering you? Describe it however feels natural - "
        "'fever and body aches for 3 days' works just as well as a long story.",
        "Namaste! 🙏 I'm listening. What would you like to talk about today - just start typing, "
        "no forms, no checkboxes.",
        "Hello there! 🩺 Go ahead - tell me what's going on, however it comes out. Plain talk, "
        "Hinglish, messy sentences - I can handle all of it.",
    ]
    return variants[len(S.get("chat", [])) % len(variants)]


def _pick(options):
    """Pick one phrasing variant so drSaab doesn't sound like a recording.
    Every variant of a reply keeps the same meaning - only the words differ."""
    return random.choice(options)


_ACK_NORMAL = [
    "I hear you. Let me think it through like a doctor would.",
    "Okay, I'm listening. Give me a moment to think this through.",
    "Alright - let me think about this properly.",
]
_ACK_SEVERE = [
    "That sounds rough - I'm sorry you're dealing with this. Let me think it through properly.",
    "That sounds really rough. Sorry you're going through it - let me think it through carefully.",
]
_ACK_MILD = [
    "Okay - at least it sounds manageable. Let me still think it through properly.",
    "Alright, that doesn't sound too bad - but let me think it through anyway.",
]


def _acknowledge(u, symptoms_total):
    chief = None
    if u["symptoms"]:
        chief = u["symptoms"][0].lower()
    if u["severity"] == "severe":
        base = _pick(_ACK_SEVERE)
    elif u["severity"] == "mild":
        base = _pick(_ACK_MILD)
    else:
        base = _pick(_ACK_NORMAL)
    if chief and symptoms_total >= 2:
        base += _pick([f" Starting with the {chief}.",
                       f" The main thing seems to be the {chief}."])
    return base


def chat_turn(text, S):
    """One full conversation turn. S is any MutableMapping holding the chat
    state (in the app: st.session_state). Returns the assistant's reply."""
    S.setdefault("chat", [])
    S.setdefault("chat_symptoms", [])
    S.setdefault("chat_denied", [])
    u = understand(text)

    # 1. emergencies stop everything
    if u["emergency"]:
        if u["emergency"].startswith("What you are feeling"):
            return ("🚨 **Please stop and read this carefully.** " + u["emergency"] +
                    " that needs a real human right now, not a symptom checker. "
                    "**Call Tele-MANAS 14416** (India's free 24x7 mental-health helpline) or **112** "
                    "immediately, or tell someone near you. You matter.")
        return ("🚨 **This could be serious - please don't wait.** " + u["emergency"] +
                ". **Call 112** (national emergency) or **108** (free ambulance) right now, or get to "
                "the nearest emergency department. I'm an educational tool - this situation needs "
                "real medical help immediately.")

    # 2. update the accumulated picture
    for s in u["negated"]:
        if s in S["chat_symptoms"]:
            S["chat_symptoms"].remove(s)
        if s not in S["chat_denied"]:
            S["chat_denied"].append(s)
    for s in u["symptoms"]:
        if s in S["chat_denied"]:
            S["chat_denied"].remove(s)
        if s not in S["chat_symptoms"]:
            S["chat_symptoms"].append(s)
    if u["duration"]:
        S["chat_days"] = u["duration"]
    if u["city"]:
        S["chat_city"] = u["city"]

    # 3. bare yes / no answers to the pending question
    pending = S.get("chat_pending_q")
    short = len(text.split()) <= 4
    if pending and short and not u["symptoms"] and not u["named"] and not u["is_lab"]:
        if YES_RE.match(text.strip()):
            s = pending["symptom"]
            if s not in S["chat_symptoms"]:
                S["chat_symptoms"].append(s)
            S["chat_pending_q"] = None
            if len(S["chat_symptoms"]) >= 2:
                _chat_build_report(S["chat_symptoms"], S,
                                   days=S.get("chat_days"), city=S.get("chat_city", ""))
                return (_pick(["Noted - that's an important clue. ",
                               "Okay, that's a useful piece of the puzzle. ",
                               "Right, that fits - noted. "]) +
                        _analysis_reply(S, u["named"], focus_named=False))
            q = _ask_followup(S, S["chat_symptoms"], S["chat_denied"])
            if q:
                S["chat_pending_q"] = q
                return _pick(["Noted. One more thing: ",
                              "Okay, got it. One more thing: ",
                              "Alright. And also: "]) + q["question"]
            return "Noted. Tell me anything else you're feeling and I'll analyse the whole picture."
        if NO_RE.match(text.strip()):
            s = pending["symptom"]
            already_had = s in S["chat_symptoms"]
            if already_had:
                S["chat_symptoms"].remove(s)
            if s not in S["chat_denied"]:
                S["chat_denied"].append(s)
            S["chat_pending_q"] = None
            top_before = pending.get("condition")
            if already_had:
                reply = _pick(["Noted - I've removed that from your picture, which makes **",
                               "Okay, taking that out of the picture then - it makes **"]) + \
                        (top_before or "that condition") + _pick(["** less likely. ",
                                                                  "** somewhat less likely. "])
            else:
                reply = _pick(["Okay, good to know - that makes **",
                               "Alright - that makes **",
                               "Fair enough - that points away from **"]) + \
                        (top_before or "that condition") + _pick(["** somewhat less likely. ",
                                                                  "** a bit less likely. "])
            if len(S["chat_symptoms"]) >= 2:
                _chat_build_report(S["chat_symptoms"], S,
                                   days=S.get("chat_days"), city=S.get("chat_city", ""))
                return reply + _analysis_reply(S, u["named"], focus_named=False)
            q = _ask_followup(S, S["chat_symptoms"], S["chat_denied"])
            if q:
                S["chat_pending_q"] = q
                return reply + f"Next question: {q['question']}"
            return reply + "Tell me what else you feel."

    # 4. small talk
    if u["is_greeting_only"]:
        return _greeting_reply(S)
    if u["is_thanks"]:
        return _pick([
            "Anytime! 😊 Remember I'm an educational tool, not a replacement for a real doctor - "
            "anything that worries you deserves a proper examination. Anything else on your mind?",
            "Glad to help! 😊 Just remember - I'm an educational tool, and anything that worries "
            "you deserves a real examination. What else can I help with?",
            "Glad it was useful! 😊 Do see a real doctor for anything that concerns you though. "
            "Anything else you'd like to ask?",
        ])
    if u["is_bye"]:
        return _pick([
            "Take care of yourself! 🌼 Wishing you a speedy recovery. (And remember: **112** "
            "for emergencies anywhere in India.) Bye for now!",
            "Take care - all the best, and get well soon! 🌼 And if anything ever feels like "
            "an emergency, **112** is the number. Bye!",
            "Take care and look after yourself! 🌼 I hope you feel better soon. (**112** for "
            "emergencies, any time.) Bye for now!",
        ])

    # 5. lab results
    if u["is_lab"]:
        return _lab_reply(S, text)

    # 6. symptom talk - the main path
    if S["chat_symptoms"] or u["symptoms"]:
        named = u["named"]
        if named and len(S["chat_symptoms"]) < 2:
            # they named a condition but haven't described enough yet
            q = _ask_followup(S, S["chat_symptoms"], S["chat_denied"])
            reply = _named_condition_reply(S, named, u)
            if q:
                S["chat_pending_q"] = q
            return reply
        if len(S["chat_symptoms"]) >= 2:
            _chat_build_report(S["chat_symptoms"], S,
                               days=S.get("chat_days"), city=S.get("chat_city", ""))
            ack = _acknowledge(u, len(S["chat_symptoms"]))
            S["chat_pending_q"] = None
            return ack + "\n\n" + _analysis_reply(S, named)
        # only one symptom so far -> one focused question before analysing
        q = _ask_followup(S, S["chat_symptoms"], S["chat_denied"])
        if q:
            S["chat_pending_q"] = q
            return (_pick([f"Got it - {S['chat_symptoms'][0].lower()}. Before I analyse, one question: ",
                           f"Okay, {S['chat_symptoms'][0].lower()}. Let me ask you one thing first: ",
                           f"Alright - {S['chat_symptoms'][0].lower()} it is. One question before I go further: "])
                    + q["question"])
        _chat_build_report(S["chat_symptoms"], S,
                           days=S.get("chat_days"), city=S.get("chat_city", ""))
        return _analysis_reply(S, named)

    # 7. asking about a condition by name, no symptoms yet
    if u["named"]:
        return _named_condition_reply(S, u["named"], u)

    # 8. "show me the report / medicines"
    if u["wants_report"] and S.get("report"):
        return ("Of course - the full report is right below this chat: ranked conditions, "
                "medicines, home care, tests and nearby hospitals. Ask me anything about it.")

    # 9. honest fallback
    return ("Hmm, I want to make sure I understand you properly. Could you describe what you're "
            "feeling in your own words - for example *'fever and body aches for 3 days'*, or "
            "*'my lower back hurts since a deadlift session'*. If it's easier, you can also "
            "switch to the picker mode in the sidebar.")


# --- the Streamlit wrapper + chat interface ---------------------------------

CHAT_EXAMPLES = [
    "Hi drSaab!",
    "I have been experiencing lower back pain. It arose due to a deadlift session I was doing "
    "without a belt. Please analyze and tell me if I have sciatica or disc bulge/herniation",
    "Fever and body aches for 3 days, what is going around in Mumbai?",
    "My dengue NS1 antigen came positive and platelets are 85,000",
]


# How drSaab paces a reply, by what kind of reply it is. A human greets
# fast but thinks before giving a diagnosis - so the pause grows with the
# weight of the answer. Emergencies are answered instantly.
THINKING_PAUSES = {
    "emergency": 0.0,
    "greeting": 0.4,
    "smalltalk": 0.5,
    "one_symptom": 1.2,
    "lab": 1.6,
    "followup_short": 0.9,
    "followup_long": 1.8,
    "analysis": 2.6,
    "named_condition": 2.0,
    "fallback": 0.7,
}

# What the typing indicator says while the patient waits - it changes as
# the wait grows, like a real person typing... then checking something.
THINKING_STATUS = [
    "drSaab is typing",
    "drSaab is thinking it through",
    "drSaab is going through your symptoms",
    "drSaab is comparing it with known conditions",
    "drSaab is putting it all together",
]


def classify_reply(reply, user_text):
    """Guess what kind of reply this is, so the pause feels right. Order
    matters: analysis briefs can START with an acknowledgment, so they are
    detected by their content anywhere in the text, not just the opening."""
    if reply.startswith("🚨"):
        return "emergency"
    full = reply.lower()
    head = reply[:90].lower()
    # the doctor's analysis brief - any of its opener phrasings or its
    # tests section marks it, even when an acknowledgment comes first
    if ("here's my read" in full or "i've gone through everything" in full
            or "i've thought about this" in full or "based on what you've told me" in full
            or "what would help settle it" in full):
        return "analysis"
    if "here's the short version" in head or "let me separate those" in head:
        return "named_condition"
    if "findings noted" in head or "kept that for later" in head:
        return "lab"
    # single-symptom transition ("Got it - back pain. Before I analyse...")
    if "before i analyse" in head or "let me ask you one thing" in head:
        return "one_symptom"
    if "one more question" in head or "one more thing" in head or "next question" in head:
        return "followup_long" if len(reply) > 220 else "followup_short"
    if ("anytime" in head or "take care" in head or "glad" in head
            or "all the best" in head or "happy to help" in head
            or "look after yourself" in head):
        return "smalltalk"
    if ("i'm **drsaab**" in head or "i'm listening" in head or "good to see you" in head
            or "hello there" in head):
        return "greeting"
    if ("i hear you" in head or "okay, i'm listening" in head or "let me think" in head
            or "got it" in head):
        return "one_symptom"
    return "fallback"


def human_thinking_pause(reply, user_text, placeholder=None):
    """The pause before drSaab 'starts typing'. While it runs, the status
    line cycles through THINKING_STATUS so the wait feels alive - a doctor
    glancing at notes, not a loading bar."""
    kind = classify_reply(reply, user_text)
    total = THINKING_PAUSES.get(kind, 1.0)
    if total <= 0:
        return kind
    step, waited = 0.45, 0.0
    n_status = max(1, int(total / step))
    for i in range(n_status):
        delay = min(step, total - waited)
        if delay <= 0:
            break
        if placeholder is not None:
            try:
                idx = min(i, len(THINKING_STATUS) - 1)
                placeholder.markdown(
                    f'<div class="thinking-status"><span class="typing-dots"><i></i><i></i><i></i></span>'
                    f'&nbsp; {THINKING_STATUS[idx]}...</div>', unsafe_allow_html=True)
            except Exception:
                pass
        time.sleep(delay)
        waited += delay


def _stream_reply(text):
    """Type the reply out word by word the way a person types: short bursts
    of 2-4 words, a beat at sentence ends, a longer one between paragraphs.
    Long replies speed up gradually so a full diagnosis never takes more
    than ~15-20 seconds to type. Emergencies stream instantly."""
    is_emergency = text.startswith("🚨")
    paragraphs = text.split("\n\n")
    n_words = len(text.split())
    # pace per sentence-end so total typing time stays reasonable
    n_breaks = max(1, text.count("\n\n") + sum(
        1 for w in text.split() if w.endswith((".", "!", "?", ":", ";"))))
    fast = n_words > 90
    sentence_sleep = (0.06, 0.16) if fast else (0.14, 0.30)
    para_sleep = (0.12, 0.26) if fast else (0.30, 0.55)
    word_sleep = (0.02, 0.06) if fast else (0.05, 0.12)
    for p_idx, para in enumerate(paragraphs):
        words = para.split(" ")
        burst = []
        for i, w in enumerate(words):
            burst.append(w)
            end_of_sentence = w.endswith((".", "!", "?", ":", ";")) or w.endswith("**")
            if len(burst) >= 3 or end_of_sentence or i == len(words) - 1:
                chunk = " ".join(burst)
                # keep real paragraph breaks so markdown renders paragraphs
                # while typing too (not only after the message is stored)
                if i == len(words) - 1 and p_idx < len(paragraphs) - 1:
                    chunk += "\n\n"
                else:
                    chunk += " "
                yield chunk
                burst = []
                if is_emergency:
                    continue
                if i == len(words) - 1 and p_idx < len(paragraphs) - 1:
                    time.sleep(random.uniform(*para_sleep))      # new paragraph
                elif end_of_sentence:
                    time.sleep(random.uniform(*sentence_sleep))
                else:
                    time.sleep(random.uniform(*word_sleep))
        if burst:
            yield " ".join(burst) + " "


def process_chat_message(text):
    """The Streamlit-facing wrapper: runs one turn on st.session_state and
    returns the assistant's reply. Learned conditions are re-merged first so
    that everything studied in THIS session (Medical school mode) is usable
    in the chat immediately, without restarting the app."""
    school = st.session_state.get("school")
    if school:
        apply_learned_conditions(school)
    return chat_turn(text, st.session_state)


def render_chat():
    """The chat interface: message history, example chips, free typing, and
    the full hybrid report below once there is something to show."""
    chat = st.session_state.setdefault("chat", [])
    if not chat:
        chat.append({
            "role": "assistant",
            "text": ("Namaste! 🙏 I'm **drSaab**. Tell me what's been going on - plain sentences, "
                     "Hinglish, typos or lab reports all work. (Educational tool, not a doctor - "
                     "emergencies: **112**.)"),
        })
        st.session_state["chat"] = chat

    prompt = st.chat_input("Tell me what's been bothering you...")

    if not prompt and len(chat) <= 1:
        st.caption("Try one:")
        cols = st.columns(len(CHAT_EXAMPLES))
        for col, example in zip(cols, CHAT_EXAMPLES):
            label = example if len(example) <= 42 else example[:40] + "..."
            if col.button(label, key=f"chat_ex_{hash(example) % 10_000}"):
                prompt = example

    # Compute the turn and store BOTH messages in history BEFORE any of the
    # slow human-pacing visuals begin. If the patient sends a new message
    # while drSaab is still "typing", Streamlit restarts this script - the
    # reply would be lost if it were only appended at the end.
    live_pair = None
    if prompt:
        reply = process_chat_message(prompt)
        st.session_state["chat"].append({"role": "user", "text": prompt})
        st.session_state["chat"].append({"role": "assistant", "text": reply})
        live_pair = (prompt, reply)

    # history: everything except the pair we are about to animate live
    for m in (chat if live_pair is None else chat[:-2]):
        with st.chat_message(m["role"]):
            st.markdown(m["text"])

    if live_pair:
        prompt, reply = live_pair
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # pace the reveal like a human: a thinking pause with a live
            # status line, then word-by-word typing
            placeholder = st.empty()
            human_thinking_pause(reply, prompt, placeholder)
            placeholder.empty()
            st.write_stream(_stream_reply(reply))

    c1, c2 = st.columns([1, 5])
    if c1.button("Start a fresh chat"):
        for key in ("chat", "chat_symptoms", "chat_denied", "chat_pending_q",
                    "chat_city", "chat_days", "report", "lab_findings_text"):
            st.session_state.pop(key, None)
        st.rerun()
    c2.caption("Symptoms I have noted so far: " +
               (", ".join(st.session_state.get("chat_symptoms", [])) or "none yet"))

    # the full hybrid report appears once the chat has produced one
    if st.session_state.get("report"):
        st.divider()
        st.markdown("### Full report from our conversation")
        render_report(st.session_state["report"])


# ---------------------------------------------------------------------------
# PART 5 - THE WEB INTERFACE (Streamlit)
# ---------------------------------------------------------------------------
# Streamlit's mental model: it runs this file from top to bottom EVERY time
# the user clicks something. Widgets (buttons, boxes) remember their values
# automatically, and st.session_state lets us keep the analysis result alive
# between clicks. The UI code lives inside main(), which Streamlit calls.
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Compact skin: tight paddings, smaller fonts, denser blocks */
.app-header {padding: 0.35rem 0 0.45rem 0; border-bottom: 2px solid #0d9488; margin-bottom: 0.6rem;}
.app-header h1 {color: #0f766e; margin: 0; font-size: 1.45rem; letter-spacing: -0.5px; display: inline;}
.app-header p {color: #475569; margin: 0.15rem 0 0 0; font-size: 0.85rem;}
.disclaimer {background: #fffbeb; border: 1px solid #fcd34d; border-left: 4px solid #d97706;
             color: #78350f; padding: 5px 10px; border-radius: 6px; font-size: 0.8rem; margin-bottom: 0.6rem;}
.trend-box {background: #f0fdfa; border: 1px solid #99f6e4; border-left: 4px solid #0d9488;
            color: #134e4a; padding: 7px 12px; border-radius: 6px; font-size: 0.85rem;}
.urgency-self-care  {background: #ecfdf5; border-left: 4px solid #059669; color: #064e3b;
                     padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;}
.urgency-see-doctor {background: #fffbeb; border-left: 4px solid #d97706; color: #78350f;
                     padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;}
.urgency-urgent     {background: #fef2f2; border-left: 4px solid #dc2626; color: #7f1d1d;
                     padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;}
.emergency {background: #fef2f2; border: 2px solid #dc2626; color: #7f1d1d;
            padding: 7px 12px; border-radius: 8px; font-size: 0.85rem; margin-top: 0.6rem;}
.hybrid-bar {color: #475569; font-size: 0.8rem; margin: 0.1rem 0 0.25rem 0;}
.hybrid-bar b {color: #0f766e;}
.lab-box {background: #eef2ff; border: 1px solid #c7d2fe; border-left: 4px solid #4f46e5;
          color: #312e81; padding: 6px 10px; border-radius: 6px; font-size: 0.82rem;
          margin-bottom: 0.6rem;}
.badge-lab {background: #4f46e5; color: #ffffff; padding: 0 8px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; margin-left: 6px;}
.badge-net {background: #0d9488; color: #ffffff; padding: 0 8px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; margin-left: 6px;}
.badge-book {background: #b45309; color: #ffffff; padding: 0 8px; border-radius: 999px;
             font-size: 0.72rem; font-weight: 700; margin-left: 6px;}
/* v3.1 - the "drSaab is typing" status line while he thinks */
.thinking-status {color: #64748b; font-size: 0.82rem; padding: 4px 2px;
                  animation: fade-in 0.25s ease;}
.typing-dots i {display: inline-block; width: 6px; height: 6px; margin-right: 3px;
                border-radius: 50%; background: #0d9488; opacity: 0.3;
                animation: dot-bounce 1.1s infinite;}
.typing-dots i:nth-child(2) {animation-delay: 0.18s;}
.typing-dots i:nth-child(3) {animation-delay: 0.36s;}
@keyframes dot-bounce {
  0%, 60%, 100% {transform: translateY(0); opacity: 0.35;}
  30% {transform: translateY(-4px); opacity: 0.9;}
}
@keyframes fade-in {from {opacity: 0;} to {opacity: 1;}}
/* Compact global density: smaller body font, tighter Streamlit blocks */
.main .block-container {padding-top: 0.8rem; padding-bottom: 1rem; max-width: 1080px;}
.main p, .main li {font-size: 0.92rem; line-height: 1.45;}
.main h1 {font-size: 1.5rem;}
.main h2 {font-size: 1.15rem; margin-top: 0.7rem; margin-bottom: 0.3rem;}
.main h3 {font-size: 1.02rem; margin-top: 0.6rem; margin-bottom: 0.25rem;}
.main .stMetric {padding: 0.25rem 0.4rem;}
.main .stMetric label p {font-size: 0.78rem !important;}
.main .stMetric value {font-size: 1.05rem;}
.main [data-testid="stCaptionContainer"] {font-size: 0.78rem; margin-top: 0.1rem;}
.main [data-testid="stChatMessage"] {padding: 0.45rem 0.8rem; margin-bottom: 0.3rem;}
.main [data-testid="stExpander"] {margin-top: 0.2rem;}
.main [data-testid="stVerticalBlock"] {gap: 0.45rem;}
.main section[data-testid="stElementContainer"] {margin-top: 0.15rem;}
/* Sidebar density */
section[data-testid="stSidebar"] .block-container {padding-top: 0.7rem; gap: 0.4rem;}
section[data-testid="stSidebar"] {font-size: 0.88rem;}
section[data-testid="stSidebar"] p {font-size: 0.85rem;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {font-size: 0.75rem;}
section[data-testid="stSidebar"] h2 {font-size: 0.98rem; margin: 0.2rem 0;}
</style>
"""

URGENCY_BANNERS = {
    "self-care": ("urgency-self-care",
                  "SELF-CARE LIKELY OK - most cases settle at home. See a doctor if any red flag below appears."),
    "see-doctor": ("urgency-see-doctor",
                   "SEE A DOCTOR - this condition usually needs a professional check or a prescription."),
    "urgent": ("urgency-urgent",
               "SEE A DOCTOR URGENTLY - this condition can get worse fast. Do not wait."),
}


def build_report(symptoms, city, age_group, days, gender, ml_weight, model, all_patients,
                 internet_weight=0, knowledge=None, school=None):
    """Run the whole analysis (weather + trends + RULES + ML + internet
    evidence) and return one dictionary that the interface can draw. Kept
    separate so you could later turn it into a JSON API endpoint or save it
    to a file."""
    report = {
        "symptoms": list(symptoms), "age_group": age_group, "days": days,
        "gender": gender, "ml_weight": ml_weight, "internet_weight": internet_weight,
        "city_tried": bool(city.strip()), "city_label": None,
        "lat": None, "lon": None, "weather": None, "air": None,
        "month": None, "trend_notes": [], "results": [],
        "similar": [], "model_info": None, "internet_on": False,
        "w_rule": 0.0, "w_ml": 0.0, "w_net": 0.0,
    }
    if city.strip():
        matches = get_city_matches(city.strip())
        if matches:
            m = matches[0]
            report["lat"], report["lon"] = m["lat"], m["lon"]
            label_parts = [m["name"]]
            if m["region"] and m["region"] != m["name"]:
                label_parts.append(m["region"])
            if m["country"]:
                label_parts.append(m["country"])
            report["city_label"] = ", ".join(label_parts)
            report["weather"] = get_current_weather(m["lat"], m["lon"])
            report["air"] = get_air_quality(m["lat"], m["lon"])

    # Merge air quality into the weather dict the engine reads.
    wx = dict(report["weather"]) if report["weather"] else {"temp": None}
    if report["air"]:
        wx["aqi"] = report["air"].get("aqi")
        wx["pm25"] = report["air"].get("pm2_5")

    report["month"] = month_from_weather(report["weather"])
    report["trend_notes"] = build_trend_notes(wx, report["month"], report["lat"])

    # ---- RULES half (PART 3) -------------------------------------------
    rule_results = score_diseases(symptoms, wx, report["month"], report["lat"])

    # ---- LEARNING half (PART 4) ----------------------------------------
    ml_probs = {}
    if model:
        gender_code = {"Female": "female", "Male": "male", "Other": "other"}.get(gender)
        ml_probs = predict_ml(model, symptoms, age_band_from_group(age_group),
                              gender_code, report["month"])

    # ---- BLEND the three voices: rules + learned model + internet -----
    # The internet weight comes "off the top"; whatever is left is split
    # between rules and ML exactly like the sidebar slider asks.
    w_net = min(internet_weight, 100) / 100.0 if (knowledge and internet_weight > 0) else 0.0
    w_ml = (ml_weight / 100.0) * (1.0 - w_net)
    w_rule = (1.0 - ml_weight / 100.0) * (1.0 - w_net)
    report["w_rule"], report["w_ml"], report["w_net"] = w_rule, w_ml, w_net
    report["internet_on"] = w_net > 0
    combined = {}
    for r in rule_results:
        combined[r["name"]] = {"rule": r["score"], "ml": ml_probs.get(r["name"], 0.0), "r": r}
    if w_ml > 0 or w_net > 0:
        for name, p in ml_probs.items():
            if name not in combined and p >= 0.12:  # a condition only the ML knows
                combined[name] = {"rule": 0.0, "ml": float(p), "r": None}
    results = []
    if school is None:
        try:
            school = st.session_state.get("school") or {}
        except Exception:
            school = {}      # tests import build_report without a Streamlit runtime
    learned_names = set(school.get("learned_profiles") or {})
    book_facts = {(f["condition"], t["sign"]): t["strength"]
                  for f in (school.get("conclusions") or {}).get("facts", [])
                  for t in f["top_signs"]}
    for name, c in combined.items():
        r = c["r"] or {"name": name, "score": 0.0, "matched": [], "missing": [], "reasons": []}
        r["rule_score"] = c["rule"]
        r["ml_prob"] = c["ml"]
        # Books evidence (PART 4C): which of the user's symptoms the study
        # material itself links to this condition, and how strongly.
        r["book_confirmed"] = [s for s in r.get("matched", [])
                               if (name, s) in book_facts]
        r["learned_condition"] = name in learned_names
        # Internet evidence (PART 2B): of the reported symptoms, how many do
        # the live Wikipedia / Wikidata sources confirm for this condition?
        if w_net > 0:
            conf, confirmed = internet_evidence(symptoms, name, knowledge)
        else:
            conf, confirmed = None, []
        r["net_conf"], r["net_confirmed"] = conf, confirmed
        r["wiki"] = (knowledge or {}).get("wikipedia", {}).get(name)
        # A condition learned from the books gets a modest evidence bonus
        # proportional to how many of its book-linked signs the user has.
        book_bonus = 0.0
        if learned_names and name in learned_names and r["book_confirmed"]:
            book_bonus = min(0.06 * len(r["book_confirmed"]), 0.12)
        if conf is None:
            # No online knowledge for this condition (e.g. it was learned
            # from uploaded records) - blend the two voices we do have and
            # re-scale, so it is not punished just for being new.
            denom = w_rule + w_ml
            score = (w_rule * c["rule"] + w_ml * c["ml"]) / denom if denom > 0 else c["rule"]
        else:
            score = w_rule * c["rule"] + w_ml * c["ml"] + w_net * conf
        r["score"] = min(score + book_bonus, 0.97)
        if r["score"] >= MIN_SCORE_TO_SHOW or c["rule"] >= MIN_SCORE_TO_SHOW:
            results.append(r)
    results.sort(key=lambda r: r["score"], reverse=True)
    report["results"] = results[:4]

    # ---- evidence: similar past patients + learned test suggestions ----
    report["similar"] = find_similar_patients(symptoms, all_patients)
    for r in report["results"]:
        sims_for = [s for s in report["similar"] if s["patient"]["diagnosis"] == r["name"]]
        r["similar_count"] = len(sims_for)
        r["learned_tests"], r["learned_tests_n"] = aggregate_tests(report["similar"], r["name"])
        r["ml_features"] = explain_ml_features(model, r["name"], symptoms) if model else []
        if sims_for:
            r["ml_reason"] = (f"{len(sims_for)} of the {len(report['similar'])} closest past patient "
                              f"records (best match {sims_for[0]['sim']:.0%}) were diagnosed with {r['name']}")
        elif model and r["ml_prob"] > 0:
            r["ml_reason"] = (f"The learned model gives this a {r['ml_prob']:.0%} probability based on "
                              f"{model['n_rows']:,} training records")
    if model:
        report["model_info"] = {k: model[k] for k in
                                ("n_rows", "n_uploaded", "labels", "accuracy", "accuracy_note")}
    return report


def render_report(report):
    """Draw the full analysis: weather, trends, ranked conditions, medicines,
    home care, and nearby hospitals/clinics."""

    # ----- Step 2: live weather ------------------------------------------
    st.subheader("Step 2 - Live conditions in your city")
    if report["weather"]:
        wx, air = report["weather"], report["air"]
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperature", f"{wx['temp']:.0f}°C", f"feels like {wx['feels']:.0f}°C")
            c2.metric("Condition", describe_weather_code(wx.get("code")))
            c3.metric("Humidity", f"{wx['humidity']:.0f}%")
            if air and air.get("aqi") is not None:
                c4.metric("Air quality (US AQI)", f"{air['aqi']:.0f}", aqi_category(air["aqi"]))
            else:
                c4.metric("Air quality", "not available")
            st.caption(f"{report['city_label']} | {wx.get('local_time', 'n/a')} | Open-Meteo (free)")
    elif report["city_tried"]:
        st.warning("Could not fetch weather for that city - the analysis below runs without live "
                   "weather context.")
    else:
        st.info("No city entered - running without live weather and season context. "
                "Add your city in the sidebar for sharper results.")

    if report["trend_notes"]:
        st.markdown('<div class="trend-box"><b>What is going around right now</b><br>' +
                    "<br>".join("- " + n for n in report["trend_notes"]) + "</div>",
                    unsafe_allow_html=True)

    # ----- Step 3: ranked conditions (hybrid) ------------------------------
    engines = ("rules + learned model + internet check" if report.get("internet_on")
               else "rules + learned model")
    st.subheader(f"Step 3 - Most likely conditions (hybrid: {engines})")
    if not report["results"]:
        st.warning("drSaab could not narrow your symptoms down to any condition it knows. "
                   "Try adding a few more symptoms - and if you feel unwell, see a doctor regardless.")
    else:
        lab_text = st.session_state.get("lab_findings_text", "")
        results = report["results"]
        if lab_text:
            apply_lab_findings(results, lab_text)

        if report.get("internet_on"):
            mix = (f"Engine mix: {report['w_rule']:.0%} rules + {report['w_ml']:.0%} learned model "
                   f"+ {report['w_net']:.0%} internet evidence (live Wikipedia + Wikidata)")
        else:
            mix = f"Engine mix: {100 - report['ml_weight']}% rules + {report['ml_weight']}% learned model"
        st.caption("Analyzing: " + ", ".join(report["symptoms"]) + "  |  " + mix)
        if lab_text:
            st.markdown('<div class="lab-box"><b>Lab findings applied.</b> The ranking below has been '
                        're-calculated using the test results you entered in Step 4.</div>',
                        unsafe_allow_html=True)

        for i, r in enumerate(results[:3], 1):
            d = DISEASES.get(r["name"])
            known = d is not None
            score = r.get("score_final", r["score"])
            with st.container(border=True):
                col1, col2 = st.columns([4, 1.3])
                with col1:
                    title = f"**{i}. {r['name']}** - {match_label(score)}"
                    if r.get("lab_confirmed"):
                        title += ' <span class="badge-lab">lab-supported</span>'
                    if r.get("learned_condition"):
                        title += ' <span class="badge-book">from books</span>'
                    if report.get("internet_on") and r.get("net_conf") is not None:
                        title += f' <span class="badge-net">online {r["net_conf"]:.0%}</span>'
                    st.markdown(title, unsafe_allow_html=True)
                    if known:
                        st.caption(f"{d['about']} Typical course: {d['duration'].lower()}.")
                        if r.get("wiki"):
                            st.markdown(f"*Live source: [{r['wiki']['title']} - Wikipedia]"
                                        f"({r['wiki']['url']})*")
                    else:
                        st.caption("This condition was learned from your study material or uploaded "
                                   "patient records - it is not in the built-in knowledge base, so "
                                   "treat it as a pattern match.")
                with col2:
                    st.metric("Match", f"{score:.0%}")
                st.progress(min(score, 1.0))
                if report.get("internet_on") and r.get("net_conf") is not None:
                    st.markdown('<div class="hybrid-bar">Rules engine: <b>' + f"{r['rule_score']:.0%}" +
                                '</b> &nbsp;&middot;&nbsp; Learned model: <b>' + f"{r['ml_prob']:.0%}" +
                                '</b> &nbsp;&middot;&nbsp; Internet evidence: <b>' + f"{r['net_conf']:.0%}" +
                                '</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="hybrid-bar">Rules engine: <b>' + f"{r['rule_score']:.0%}" +
                                '</b> &nbsp;&middot;&nbsp; Learned model: <b>' + f"{r['ml_prob']:.0%}" +
                                '</b></div>', unsafe_allow_html=True)

                if known:
                    cls, text = URGENCY_BANNERS[d["urgency"]]
                else:
                    cls, text = URGENCY_BANNERS["see-doctor"]
                st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

                # Evidence order = how the engine actually works: the internet
                # symptom lists are checked FIRST, then cross-checked against
                # drSaab's own database of conditions.
                why = []
                if report.get("internet_on") and r.get("net_conf") is not None:
                    n_net = len(r["net_confirmed"])
                    if n_net:
                        names_net = ", ".join(s for s, _ in r["net_confirmed"])
                        srcs_net = " and ".join(sorted({src for _, src in r["net_confirmed"]}))
                        why.append(f"Internet check first ({srcs_net}): {n_net} of your "
                                   f"{len(report['symptoms'])} symptoms are listed for {r['name']} "
                                   f"in live online medical knowledge ({names_net}).")
                    else:
                        why.append(f"Internet check first: none of your symptoms could be confirmed for "
                                   f"{r['name']} in live online medical knowledge.")
                if r["matched"]:
                    db_note = (" - the internet findings agree with our own database"
                               if report.get("internet_on") and r.get("net_confirmed") else "")
                    why.append(f"Cross-check with our own database{db_note}: "
                               f"you reported {len(r['matched'])} of its typical signs: "
                               f"{', '.join(r['matched'])}.")
                if r["missing"]:
                    why.append(f"Usually also involves (you did not report): {', '.join(r['missing'])}.")
                why.extend(r["reasons"])
                if r.get("book_confirmed"):
                    why.append(f"Books: {'; '.join(r['book_confirmed'])} - exactly the signs "
                               "the study material links to this condition.")
                if r.get("learned_condition"):
                    why.append("This condition was learned from the books you uploaded in "
                               "Medical school mode - not from the built-in knowledge base.")
                if r.get("ml_reason"):
                    why.append(f"Machine learning: {r['ml_reason']}.")
                if r.get("ml_features"):
                    why.append("The learned model keyed on: " +
                               ", ".join(s for s, _ in r["ml_features"]) + ".")
                why.extend(r.get("lab_reasons") or [])
                st.markdown("**Why this showed up**")
                st.markdown("\n".join(f"- {w}" for w in why))

                with st.expander("Common medicines (typical ADULT doses)"):
                    if known:
                        st.table(pd.DataFrame(d["medicines"],
                                              columns=["Medicine", "Typical adult dose", "Helps with"]))
                        caption = india_brand_caption(d["medicines"])
                        if caption:
                            st.caption(caption)
                    else:
                        meds = sorted({m for s in report["similar"]
                                       if s["patient"]["diagnosis"] == r["name"]
                                       for m in s["patient"].get("medicines") or []})
                        if meds:
                            st.markdown("Medicines recorded for similar past patients: " +
                                        ", ".join(meds[:8]) + ".")
                        else:
                            st.caption("No medicine information learned for this condition yet.")
                    st.caption("Doses are general adult references, not a prescription. Children, pregnancy, "
                               "other medicines and long-term conditions change the right choice - "
                               "always confirm with a doctor or pharmacist.")

                c_home, c_red = st.columns(2)
                with c_home:
                    st.markdown("**Home care**")
                    if known:
                        st.markdown("\n".join(f"- {h}" for h in d["home_care"]))
                    else:
                        st.markdown("- Follow the advice of the doctor who treats you\n"
                                    "- Track your symptoms and note anything that worsens\n"
                                    "- Rest and stay hydrated")
                with c_red:
                    st.markdown("**See a doctor if**")
                    if known:
                        st.markdown("\n".join(f"- {f}" for f in d["red_flags"]))
                    else:
                        st.markdown("- Symptoms worsen quickly or new severe symptoms appear\n"
                                    "- Fever persists beyond 3 days\n"
                                    "- You cannot eat, drink or stay awake normally")

        if report["similar"]:
            with st.expander(f"The {len(report['similar'])} most similar past patient records", expanded=False):
                rows = []
                for s in report["similar"]:
                    p = s["patient"]
                    rows.append({
                        "Record": p["id"],
                        "Source": "Your upload" if p.get("source") == "uploaded" else "Example data",
                        "Similarity": f"{s['sim']:.0%}",
                        "Shared symptoms": len(s["overlap"]),
                        "Profile": f"{p.get('age') or '?'}y {p.get('gender') or '-'}",
                        "Month": p.get("month") or "-",
                        "Diagnosis": p["diagnosis"],
                        "Tests done": "; ".join(p["tests_done"][:3]) or "-",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption("Example records are generated from the built-in knowledge base; 'Your upload' "
                           "rows come from CSV files added in Doctor mode.")

    # Age and duration advice
    if report["results"]:
        if report["age_group"].startswith("Child"):
            st.warning("You selected a CHILD: all doses above are adult doses. Children must be dosed by a "
                       "doctor or pharmacist based on weight. A child with fever should be seen promptly if "
                       "under 3 months old, feverish beyond 3 days, very drowsy, or drinking very little.")
        elif report["age_group"].startswith("Senior"):
            st.info("For older adults, symptoms can be subtler and complications more common - "
                    "consider seeing a doctor earlier.")
        if report["days"] >= 14:
            st.warning("Symptoms lasting two weeks or more really deserve a professional opinion.")
        elif report["days"] >= 7:
            st.info("Symptoms lasting a week or more are worth getting checked by a doctor.")

    # ----- Step 4: suggested tests & scans (v2.0) --------------------------
    st.subheader("Step 4 - Suggested tests & scans")
    if report["results"]:
        st.caption("Two sources: the medical knowledge base, and what similar past patients actually had "
                   "done. Get the relevant ones done, bring the results back here and enter them below - "
                   "the prediction re-ranks using your findings. A doctor still makes the final call.")
        for r in report["results"][:3]:
            d = DISEASES.get(r["name"])
            with st.expander(f"Suggested for: {r['name']}"):
                if d and d.get("typical_tests"):
                    st.table(pd.DataFrame(d["typical_tests"], columns=["Test / scan", "Why it helps"]))
                else:
                    st.caption("No built-in test list for this condition (it was learned from your data) - "
                               "the learned suggestions below show what similar patients had done.")
                if r.get("learned_tests_n", 0) > 0 and r.get("learned_tests"):
                    n = r["learned_tests_n"]
                    st.markdown(f"**From the learned records** - {n} similar patient record(s) with this "
                                "diagnosis had:")
                    st.markdown("\n".join(f"- {t} - {c} of {n} ({c / n:.0%})"
                                          for t, c in r["learned_tests"][:6]))

        with st.container(border=True):
            st.markdown("**Already have some test results? Refine the prediction**")
            st.caption("Pick common findings or type your own (one per line, e.g. 'NS1 antigen: positive' "
                       "or 'Platelets: 85,000'). drSaab reads them and re-ranks Step 3.")
            chips = st.multiselect("Quick-pick findings", COMMON_LAB_FINDINGS, key="lab_chips")
            free = st.text_area("Or type results in your own words (one per line)", key="lab_free",
                                placeholder="NS1 antigen: positive\nPlatelets: 85,000\nWidal: 1:160")
            hints = []
            for r in report["results"][:2]:
                for rule in CONFIRMING_FINDINGS.get(r["name"], [])[:4]:
                    if rule["points"] > 0:
                        hints.append(rule["label"])
            if hints:
                st.caption("Examples of what I can read for your top conditions: " +
                           "; ".join(hints[:6]) + ".")
            b1, b2 = st.columns(2)
            if b1.button("Refine predictions with these findings", type="primary"):
                txt = "\n".join(chips + ([free] if free.strip() else []))
                if txt.strip():
                    st.session_state["lab_findings_text"] = txt
                    st.rerun()
                else:
                    st.warning("Pick or type at least one finding first.")
            if b2.button("Clear lab findings"):
                st.session_state.pop("lab_findings_text", None)
                st.rerun()

    # ----- Step 5: nearby hospitals & clinics ------------------------------
    st.subheader("Step 5 - Nearest hospitals & clinics")
    if report["lat"] is None:
        st.info("Enter your city in the sidebar and press Analyze again to find hospitals and clinics near you.")
    else:
        with st.spinner("Searching OpenStreetMap for hospitals and clinics near you..."):
            places, source = get_nearby_healthcare(report["lat"], report["lon"])
        if places:
            top = places[:12]
            st.map(pd.DataFrame({
                "lat": [report["lat"]] + [p["lat"] for p in top],
                "lon": [report["lon"]] + [p["lon"] for p in top],
            }))
            st.caption("The cluster shows your location together with the closest facilities. "
                       "Map data (c) OpenStreetMap contributors.")
            rows = ["| Facility | Type | Distance | Address | Directions |",
                    "|---|---|---|---|---|"]
            for p in top:
                name = p["name"].replace("|", "/")
                addr = (p["address"] or "-").replace("|", "/")
                rows.append(f'| {name} | {p["category"]} | {p["distance_km"]:.1f} km | {addr} '
                            f'| [Open in Maps]({p["maps_url"]}) |')
            st.markdown("\n".join(rows))
            st.caption(f"Found via {source}. Distances are straight-line, not road distance.")
        else:
            st.info("No facilities found within about 8 km, or the place service is unreachable "
                    "right now. Try again in a moment.")

    st.markdown('<div class="emergency"><b>Emergency in India?</b> Call <b>112</b> (national emergency '
                'number) or <b>108</b> (free ambulance in most states), or go straight to the nearest '
                'government hospital or emergency department. drSaab cannot detect emergencies.</div>',
                unsafe_allow_html=True)


# ---- shared session-state helpers (used by both modes) ----------------------


def boot_learning():
    """Runs once per session: build the example patients, load any saved
    patient_data.csv sitting next to app.py, and train the model first time."""
    if st.session_state.get("boot_done"):
        return
    st.session_state["boot_done"] = True
    st.session_state["seed_patients"] = make_seed_patients()
    extra = []
    if os.path.exists(PATIENT_DATA_FILE):
        try:
            df = pd.read_csv(PATIENT_DATA_FILE)
            extra, _issues, _stats = parse_patient_csv(df)
            st.session_state["boot_load"] = (f"Loaded {len(extra)} patient records from "
                                             f"{PATIENT_DATA_FILE} at startup")
        except Exception as e:
            st.session_state["boot_load"] = f"Could not read {PATIENT_DATA_FILE}: {e}"
    st.session_state["extra_patients"] = extra
    retrain_model()


def retrain_model():
    """(Re)train the ML model on example patients + everything learned so far."""
    patients = st.session_state.get("seed_patients", []) + st.session_state.get("extra_patients", [])
    try:
        st.session_state["ml_model"] = train_ml_model(patients)
    except Exception as e:
        st.session_state["ml_model"] = None
        st.session_state["ml_error"] = str(e)


def boot_internet_knowledge():
    """Runs once per session: load the cached internet knowledge, or download
    it from Wikipedia + Wikidata the very first time (then it lives in
    internet_knowledge.json and every later start is instant)."""
    if st.session_state.get("ik_boot_done"):
        return
    st.session_state["ik_boot_done"] = True
    cached = _load_knowledge_file()
    fresh = cached is not None and (time.time() - cached["fetched_at"]) / 3600 < KNOWLEDGE_MAX_AGE_HOURS \
        and (cached["wikidata"] or cached["wikipedia"])
    if fresh:
        st.session_state["internet_knowledge"] = cached
        return
    with st.spinner("Downloading current medical knowledge from Wikipedia and Wikidata "
                    "(this happens once, then it is cached)..."):
        knowledge = load_internet_knowledge(force_refresh=True)
    if knowledge is None or not (knowledge.get("wikidata") or knowledge.get("wikipedia")):
        knowledge = None          # nothing arrived and no old cache - run without
    st.session_state["internet_knowledge"] = knowledge


# ---- PART 4C hook: the medical school (learning from books & conversations) --

def boot_medical_school():
    """Runs once per session: load medical_school.json (everything drSaab
    learned from books, PDFs and conversations in past sessions) and merge
    the learned condition profiles into the live knowledge base, so the
    picker, the chat and the hybrid engine can all use them."""
    if st.session_state.get("school_boot_done"):
        return
    st.session_state["school_boot_done"] = True
    school = ms.load_school()
    st.session_state["school"] = school if school else ms.new_school()
    apply_learned_conditions(st.session_state["school"])


def apply_learned_conditions(school):
    """Merge learned profiles into DISEASES (in place) + add their symptom
    vocabulary + remember which conditions came from study (for honest
    labelling in the UI). Learned SYMPTOM phrases (e.g. 'Koplik spots') are
    registered as first-class symptoms: they appear in the picker options,
    the chat vocabulary and the ML model features. Returns (n_conditions,
    n_additions)."""
    n_new = 0
    for name, profile in (school.get("learned_profiles") or {}).items():
        if name not in DISEASES:
            DISEASES[name] = profile
            _DISEASE_LOOKUP[_norm_key(name)] = name
            _ALL_TEST_NAMES.extend(t for t, _ in profile.get("typical_tests", []) or [])
            n_new += 1
    # NEW symptom phrases the books themselves defined ('Koplik spots inside
    # the mouth are a specific sign of measles') become real options
    for key, entry in (school.get("learned_symptoms") or {}).items():
        name = entry.get("name") or entry.get("full")
        if not name or name in SYMPTOM_SET:
            continue
        SYMPTOMS.append(name)
        SYMPTOM_SET.add(name)
        _SYMPTOM_LOOKUP[_norm_key(name)] = name
        if _norm_key(entry.get("full") or name) not in _SYMPTOM_LOOKUP:
            _SYMPTOM_LOOKUP[_norm_key(entry["full"])] = name
        for cond_key in (entry.get("conditions") or {}):
            # attach to the matching condition (known or learned)
            target = _DISEASE_LOOKUP.get(cond_key) or next(
                (n for n in DISEASES if _norm_key(n) == cond_key), None)
            if target and name not in DISEASES[target]["symptoms"]:
                DISEASES[target]["symptoms"][name] = 3   # a defining sign
        # let the internet matcher see the phrase as the book wrote it too
        INTERNET_SYMPTOM_TERMS.setdefault(name, [name.lower()])
        full = entry.get("full")
        if full and full != name:
            INTERNET_SYMPTOM_TERMS[name].append(str(full).lower())
    # extra symptoms the books linked to conditions we already knew
    for cond, extra in (school.get("conclusions") or {}).get("additions", {}).items():
        d = DISEASES.get(cond)
        if d:
            for sym, strength in extra.items():
                if sym not in d["symptoms"]:
                    d["symptoms"][sym] = 2 if strength >= 0.5 else 1
                    if sym not in SYMPTOM_SET:
                        SYMPTOMS.append(sym)
                        SYMPTOM_SET.add(sym)
                        _SYMPTOM_LOOKUP[_norm_key(sym)] = sym
    # extra tests the books suggested
    for cond, tests in (school.get("conclusions") or {}).get("tests", {}).items():
        d = DISEASES.get(cond)
        if d:
            known = {t for t, _ in d.get("typical_tests", [])}
            for t, n in tests.items():
                if t not in known:
                    d.setdefault("typical_tests", []).append(
                        (t, f"Suggested by the study material ({n} mentions)"))
    return n_new, len((school.get("conclusions") or {}).get("additions", {}))


def school_lexicon():
    """The study lexicon built from the CURRENT (possibly merged) knowledge
    base - used when the doctor uploads new books in Medical School mode."""
    return ms.build_lexicon(
        DISEASES, SYMPTOMS,
        {t for ts in TYPICAL_TESTS.values() for t, _ in ts} |
        {t for d in DISEASES.values() for t, _ in (d.get("typical_tests") or [])},
        SYMPTOM_ALIASES, DIAGNOSIS_ALIASES)


def study_book_upload(uploaded_file):
    """Study one uploaded book/PDF/text in Medical School mode. Returns the
    study report dict (ok/error + what was learned)."""
    school = st.session_state.get("school") or ms.new_school()
    data = uploaded_file.read()
    result = ms.study_document(school, data, uploaded_file.name, school_lexicon(), DISEASES)
    if result["ok"]:
        n_new, n_add = apply_learned_conditions(school)
        retrain_model()
        st.session_state["school"] = school
        result["conditions_added"] = n_new
        result["conditions_augmented"] = n_add
    return result


def study_conversation_upload(uploaded_file):
    """Learn from one uploaded doctor-patient transcript in Medical School
    mode. The conversation feeds the study corpus AND, when a diagnosis was
    stated, creates a patient record that trains the ML model too."""
    school = st.session_state.get("school") or ms.new_school()
    text = uploaded_file.read().decode("utf-8", errors="replace")
    normalizers = {"symptom": normalize_symptom, "diagnosis": normalize_diagnosis}
    result = ms.learn_from_conversation(school, text, uploaded_file.name,
                                        school_lexicon(), DISEASES, normalizers)
    n_new, n_add = apply_learned_conditions(school)
    if result.get("record"):
        existing = st.session_state.get("extra_patients", [])
        existing.append(result["record"])
        st.session_state["extra_patients"] = existing
        retrain_model()
    st.session_state["school"] = school
    result["conditions_added"] = n_new
    return result


def symptom_options():
    """Built-in symptoms + any new symptoms learned from uploaded records
    or from books studied in Medical school mode (re-merged every rerun so
    the picker reflects everything learned in THIS session immediately)."""
    school = st.session_state.get("school")
    if school:
        apply_learned_conditions(school)
    opts = list(SYMPTOMS)
    known = set(opts)
    for p in st.session_state.get("extra_patients", []):
        for s in p["symptoms"]:
            if s not in known:
                opts.append(s)
                known.add(s)
    return opts


def _quick_city_changed():
    """Copy the quick-pick Indian city into the free-text city box."""
    st.session_state["city_input"] = st.session_state["quick_city_pick"]


def render_medical_school_page():
    """The 'teach me from books' screen: upload medical books/PDFs and
    doctor-patient transcripts; drSaab studies them, draws conclusions and
    remembers everything between sessions."""
    st.subheader("Medical school - teach drSaab from books & conversations")
    st.caption("Upload books, PDFs or doctor-patient transcripts. drSaab studies them: new conditions "
               "are learned from scratch, known ones gain extra signs and tests. Everything is saved "
               "in `medical_school.json` and remembered forever.")

    school = st.session_state.get("school") or {}
    corpus = school.get("corpus") or {}
    concl = school.get("conclusions") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Books studied", len(school.get("books", [])))
    c2.metric("Conversations", len(school.get("conversations", [])))
    c3.metric("Sentences read", f"{corpus.get('sentences', 0):,}")
    c4.metric("Conditions learned", len(school.get("learned_profiles", {})))
    c5.metric("Facts concluded", concl.get("facts_total", 0))
    if school.get("books") or school.get("conversations"):
        st.caption("Everything below is remembered in medical_school.json next to app.py.")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("**📚 Study a book** (PDF or text - textbook chapters, class notes)")
        book = st.file_uploader("Upload a book / PDF / notes", type=["pdf", "txt"],
                                key="school_book_up", label_visibility="collapsed")
        if book is not None:
            if st.button(f"Study '{book.name}'", type="primary", key="school_study_btn"):
                with st.spinner("Studying... reading, highlighting the medical facts, "
                                "counting associations, drawing conclusions"):
                    result = study_book_upload(book)
                if result["ok"]:
                    st.success(f"Studied **{book.name}** - {result['pages']} pages, "
                               f"{result['sentences']} sentences read.")
                    if result.get("new_conditions"):
                        st.info("🆕 New conditions learned: " +
                                ", ".join(result["new_conditions"]) +
                                ". They now appear in the picker, the chat and the analysis engine.")
                    if result.get("new_symptoms"):
                        st.info("➕ New symptoms learned from this book: " +
                                ", ".join(result["new_symptoms"]) +
                                ". They are now options in the symptom picker and the chat "
                                "understands them.")
                    if result.get("additions"):
                        aug = "; ".join(f"{c}: +{', '.join(s[:28] for s in syms[:3])}"
                                        for c, syms in result["additions"].items())
                        st.caption("Known conditions gained extra signs: " + aug)
                    if result.get("conditions_added") or result.get("new_symptoms"):
                        st.rerun()
                else:
                    st.error(result["error"])
    with right:
        st.markdown("**🩺 Learn from a consultation** (transcript with Doctor:/Patient: turns)")
        conv = st.file_uploader("Upload a conversation transcript", type=["txt"],
                                key="school_conv_up", label_visibility="collapsed")
        if conv is not None:
            if st.button(f"Learn from '{conv.name}'", type="primary", key="school_conv_btn"):
                with st.spinner("Reading the conversation - complaints, diagnosis, advice"):
                    result = study_conversation_upload(conv)
                st.success(f"Learned from **{conv.name}** - {result['turns']} turns read.")
                if result["diagnosis"]:
                    st.info(f"Doctor's diagnosis detected: **{result['diagnosis']}**. "
                            f"Symptoms noted: {', '.join(result['symptoms'][:6])}. "
                            "A synthetic training record was added to the learning engine.")
                else:
                    st.caption("No explicit diagnosis was stated - the complaints still "
                               "joined the study corpus.")

    st.divider()

    learned = school.get("learned_profiles") or {}
    if learned:
        st.markdown(f"**🎓 Conditions drSaab learned from books** ({len(learned)})")
        for name, p in learned.items():
            with st.expander(f"{name} - learned from {p.get('mentions', '?')} mentions"):
                top = sorted(p.get("symptoms", {}).items(), key=lambda kv: -kv[1])[:8]
                st.markdown("- " + "\n- ".join(f"{s} (weight {w})" for s, w in top))
                st.caption(p.get("about", ""))

    additions = concl.get("additions") or {}
    if additions:
        with st.expander(f"➕ Extra signs the books added to known conditions ({len(additions)})"):
            for cond, syms in additions.items():
                st.markdown(f"- **{cond}**: " + ", ".join(s for s in syms))

    learned_syms = school.get("learned_symptoms") or {}
    if learned_syms:
        with st.expander(f"🔤 New symptom phrases learned from books ({len(learned_syms)}) "
                         "- these are now options in the picker"):
            for e in learned_syms.values():
                conds = ", ".join(sorted(e.get("conditions", {})))
                st.markdown(f"- **{e['name']}** (from \"{e.get('full', e['name'])}\", "
                            f"a sign of: {conds})")

    confusions = concl.get("confusions") or []
    if confusions:
        with st.expander(f"🔀 Conditions the books often discuss together ({len(confusions)})"):
            for c in confusions:
                st.markdown(f"- **{c['a']}** and **{c['b']}** "
                            f"(mentioned together {c['co_mentions']} times)")

    facts = concl.get("facts") or []
    if facts:
        st.markdown("**🧠 What drSaab concluded** (strongest sign associations)")
        fact_rows = []
        for f in facts[:12]:
            signs = ", ".join(f"{t['sign']} ({t['strength']:.2f})" for t in f["top_signs"][:3])
            fact_rows.append({"Condition": f["condition"],
                              "Known before?": "built-in" if f["known"] else "learned",
                              "Mentions": f["mentions"], "Strongest signs": signs})
        st.dataframe(pd.DataFrame(fact_rows), width="stretch", hide_index=True)
        st.caption("Sign strength = pointwise mutual information: how much more often the sign "
                   "appears with this condition than chance would predict. It is honest "
                   "statistics, not a black box.")

    if not school.get("books") and not school.get("conversations"):
        st.info("Nothing studied yet. Upload a medical PDF or a consultation transcript above - "
                "or try the sample textbook and conversations in the `study_material/` folder.")

    st.divider()
    if st.button("🗑️ Forget everything from the medical school"):
        st.session_state["school"] = ms.new_school()
        ms.save_school(st.session_state["school"])
        st.rerun()


def render_doctor_page():
    """The 'teach the app' screen: upload patient CSVs, validate, retrain."""
    st.subheader("Doctor mode - teach drSaab from real patient records")
    st.caption("Upload anonymised patient CSVs and drSaab retrains its ML model on the spot - "
               "matching patients then get the same test and medicine suggestions.")

    model = st.session_state.get("ml_model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training records", f"{model['n_rows']:,}" if model else "0")
    c2.metric("From your uploads", f"{model['n_uploaded']:,}" if model else "0")
    c3.metric("Conditions the model knows", len(model["labels"]) if model else 0)
    c4.metric("Model accuracy", f"{model['accuracy']:.0%}" if model else "n/a")
    if model:
        st.caption(f"Accuracy is {model['accuracy_note']}.")

    if st.session_state.get("boot_load"):
        st.info(st.session_state["boot_load"])
    if st.session_state.get("ml_error"):
        st.error(f"Model training failed: {st.session_state['ml_error']}")

    st.warning("Privacy first: upload only records you are allowed to use, with names, phone "
               "numbers, addresses and hospital numbers removed.")

    upload = st.file_uploader("Upload patient records (CSV or Excel)", type=["csv", "xlsx"])
    with st.expander("What the file should look like (+ free template)"):
        st.markdown(
            "| Column | Required | Example |\n"
            "|---|---|---|\n"
            "| patient_id | optional | P-001 |\n"
            "| age | optional | 34 |\n"
            "| gender | optional | female |\n"
            "| city | optional | Pune |\n"
            "| visit_month | optional (1-12 or month name) | 8 |\n"
            "| symptoms | **YES** - separate with ; | High fever (39°C / 102°F or above);Pain behind the eyes |\n"
            "| diagnosis | **YES** | Dengue Fever |\n"
            "| tests_done | optional - separate with ; | Dengue NS1 antigen (blood);CBC (complete blood count) |\n"
            "| test_results | optional | NS1 positive |\n"
            "| medicines | optional - separate with ; | Paracetamol;ORS |\n"
            "| outcome | optional | Recovered in 9 days |")
        st.download_button("Download the CSV template (5 example rows)",
                           build_csv_template(), file_name="patient_data_template.csv",
                           mime="text/csv")
        st.caption("Common short forms are understood automatically: 'fever' becomes Viral Fever, "
                   "'loose motion' becomes Diarrhea, 'uti' becomes Urinary Tract Infection, 'dengue' "
                   "becomes Dengue Fever, and so on. Unknown symptoms and diagnoses are kept and "
                   "learned as new entries.")

    if upload is not None:
        try:
            if str(upload.name).lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(upload)          # Excel files need the openpyxl package
            else:
                df = pd.read_csv(upload)
        except UnicodeDecodeError:
            df = pd.read_csv(upload, encoding="latin-1", sep=None, engine="python")
        st.caption(f"File loaded: {len(df)} rows, {len(df.columns)} columns. First 10 rows:")
        st.dataframe(df.head(10), width="stretch")

        if st.button("Validate, add to the learning data and retrain", type="primary"):
            patients, issues, stats = parse_patient_csv(df)
            if stats:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows read", stats["total"])
                m2.metric("Usable records", stats["ok"])
                m3.metric("Skipped rows", stats["skipped"])
                m4.metric("New symptoms learned", len(stats["new_symptoms"]))
                if stats["new_symptoms"]:
                    st.caption("New symptoms added to the vocabulary: " +
                               ", ".join(stats["new_symptoms"][:15]))
                if stats["new_diagnoses"]:
                    st.caption("New conditions learned (not in the built-in knowledge base): " +
                               ", ".join(stats["new_diagnoses"][:15]))
                if stats["by_diagnosis"]:
                    series = pd.Series(dict(stats["by_diagnosis"])).sort_values(ascending=False).head(15)
                    st.bar_chart(series)
                    st.caption("Records per condition (top 15).")
            if issues:
                with st.expander(f"Skipped rows and warnings ({len(issues)})"):
                    st.markdown("\n".join(f"- {i}" for i in issues[:40]))
            if patients:
                existing = st.session_state.get("extra_patients", [])
                seen = {(p["id"], p["diagnosis"], tuple(sorted(p["symptoms"]))) for p in existing}
                added = 0
                for p in patients:
                    key = (p["id"], p["diagnosis"], tuple(sorted(p["symptoms"])))
                    if key not in seen:
                        existing.append(p)
                        seen.add(key)
                        added += 1
                st.session_state["extra_patients"] = existing
                retrain_model()
                model = st.session_state.get("ml_model")
                if added:
                    st.success(f"Added {added} new records and retrained the model "
                               f"({model['n_rows']:,} total training records now). "
                               "Switch to 'Check my symptoms' to see the effect.")
                else:
                    st.info("All rows in this file were already in the learning data - nothing new added.")
            elif stats:
                st.error("No usable records were found in this file. Check the required columns.")

    st.divider()
    extra = st.session_state.get("extra_patients", [])
    if extra:
        st.download_button("Download all learned patient records (CSV)",
                           patients_to_csv(extra), file_name="patient_data.csv", mime="text/csv")
    st.caption("Persistence trick: save the downloaded file as patient_data.csv in the same folder as "
               "app.py (or commit it to your GitHub repo when the app is on Streamlit Cloud) - drSaab "
               "loads it automatically every time it starts.")


def main():
    st.set_page_config(page_title="drSaab - Hybrid Symptom Checker for India", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="app-header"><h1>drSaab</h1> '
        "<p>Symptom checker for India - explainable rules + learning model + live internet check, "
        "weather and hospital finder.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer"><b>Educational tool - not medical advice.</b> Always confirm with '
        "a qualified doctor before taking any medicine. Emergencies: call <b>112</b>.</div>",
        unsafe_allow_html=True,
    )

    boot_learning()
    boot_internet_knowledge()
    boot_medical_school()

    with st.sidebar:
        st.radio("Mode", ["Chat with drSaab", "Check my symptoms (picker)",
                          "Medical school - teach me from books",
                          "Doctor mode - teach the app"], key="mode")
        if st.session_state["mode"] == "Chat with drSaab":
            st.caption("Plain sentences, Hinglish, typos, lab reports - the full report appears "
                       "below the chat.")
        city = st.text_input("City (weather + hospitals)", key="city_input", placeholder="e.g. Mumbai")
        st.selectbox("Quick-pick", ["(type my own below)"] + INDIA_CITIES,
                     key="quick_city_pick", on_change=_quick_city_changed)
        c1, c2 = st.columns(2)
        with c1:
            age_group = st.selectbox("Age", ["Adult", "Teen", "Child", "Senior"], key="age_sel",
                                     help="Adult 18-64 · Teen 12-17 · Child under 12 · Senior 65+")
        with c2:
            gender = st.selectbox("Gender", ["Not saying", "Female", "Male", "Other"], key="gender_sel")
        days = st.slider("Days with symptoms", 1, 30, 2, key="days_sel")
        st.caption("Weather: Open-Meteo · Places: OpenStreetMap · Knowledge: Wikipedia + Wikidata. "
                   "Free, no API keys.")
        st.divider()
        ml_weight = st.slider("Learned model weight", 0, 100, 50, format="%d%%", key="ml_weight_sel",
                              help="Rules = written medical logic (fully explainable). "
                                   "ML = patterns learned from patient records.")
        ik = st.session_state.get("internet_knowledge")
        net_on = st.toggle("Internet check (Wikipedia + Wikidata)",
                           value=ik is not None, disabled=ik is None, key="net_toggle",
                           help="Symptoms are matched against live medical knowledge FIRST, then "
                                "cross-checked against drSaab's own database.")
        net_weight = st.slider("Internet weight", 5, 40, 25, format="%d%%", key="net_weight_sel",
                               disabled=not (net_on and ik is not None))
        ik_status = internet_knowledge_summary(ik)
        st.caption(ik_status if ik_status else
                   "Internet knowledge not downloaded - running on rules + ML only.")
        if st.button("Refresh internet knowledge"):
            with st.spinner("Downloading fresh medical knowledge from Wikipedia + Wikidata..."):
                fresh_ik = load_internet_knowledge(force_refresh=True)
            if fresh_ik is not None and not (fresh_ik.get("wikidata") or fresh_ik.get("wikipedia")):
                fresh_ik = None
            st.session_state["internet_knowledge"] = fresh_ik
            st.rerun()

    if st.session_state["mode"] == "Doctor mode - teach the app":
        render_doctor_page()
    elif st.session_state["mode"] == "Medical school - teach me from books":
        render_medical_school_page()
    elif st.session_state["mode"] == "Check my symptoms (picker)":
        options = symptom_options()
        extra_n = len(options) - len(SYMPTOMS)
        help_text = (f"{len(SYMPTOMS)} built-in symptoms"
                     + (f" + {extra_n} learned from your uploaded data" if extra_n else ""))
        symptoms = st.multiselect("Select your symptoms", options, placeholder="Pick at least 2 symptoms",
                                  help=help_text)
        can_analyze = len(symptoms) >= 2
        analyze_btn = st.button("Analyze my symptoms", type="primary", disabled=not can_analyze)

        if analyze_btn:
            with st.spinner(f"Scoring {len(DISEASES)} conditions with the hybrid engine "
                            "(rules + learned model + internet evidence)..."):
                st.session_state["report"] = build_report(
                    symptoms, city, age_group, days, gender, ml_weight,
                    st.session_state.get("ml_model"),
                    st.session_state.get("seed_patients", []) + st.session_state.get("extra_patients", []),
                    net_weight if (net_on and ik is not None) else 0,
                    st.session_state.get("internet_knowledge"),
                    school=st.session_state.get("school"))

        if "report" not in st.session_state:
            st.info("Pick your symptoms above and press Analyze - the full report appears here.")
        else:
            report = st.session_state["report"]
            if set(symptoms) and set(symptoms) != set(report["symptoms"]):
                st.info("Your symptom selection changed since this analysis - click Analyze again to refresh.")
            render_report(report)
    else:
        render_chat()

    with st.expander("How does this app work? (guide for the builder)"):
        st.markdown(
            "- **Knowledge base:** editable Python dictionaries cover 31 conditions common in India - "
            "symptoms with weights, seasons, weather triggers, medicines, tests and lab findings.\n"
            "- **Free APIs:** live weather (Open-Meteo) and nearby hospitals (OpenStreetMap); one "
            "SPARQL query + one summary request per disease pull Wikipedia/Wikidata symptom lists, "
            "cached in internet_knowledge.json and blended into every prediction as a third voice.\n"
            "- **Rule engine:** each condition gets a RECALL score (how much of its symptom picture "
            "you reported) and a PRECISION score (how much of your input it explains), plus season "
            "and weather bonuses.\n"
            "- **Learning engine:** LogisticRegression trained on ~2,000 generated patients PLUS CSV "
            "records uploaded in Doctor mode; similar past patients explain results and drive test "
            "suggestions; typed lab findings re-rank conditions.\n"
            "- **Conversation brain (v3.0):** chat_turn() turns free text into intents (greeting / "
            "symptoms / lab / emergency / named condition), extracts symptoms via aliases + regex + "
            "fuzzy matching + negation, understands Hinglish and durations, asks ONE follow-up at a "
            "time, and reuses the same hybrid engine as the picker. Emergencies are detected first "
            "and answered with real helpline numbers.\n"
            "- **This interface:** Streamlit re-runs the file top-to-bottom on every click; "
            "st.session_state keeps the report, learned records and trained model alive."
        )

    st.caption("drSaab v3.1 (India edition) - Python + Streamlit + scikit-learn. Weather/air: "
               "Open-Meteo · Places: OpenStreetMap · Knowledge: Wikipedia + Wikidata. Free forever. "
               "Educational project - not medical advice.")


# Streamlit runs the script as __main__, which launches the interface.
# Importing the file (for testing) does NOT launch it - that is why main()
# is a function sitting behind this guard.
if __name__ == "__main__":
    main()
