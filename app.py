import base64
import datetime
import io
import json
import os
import re
import sqlite3
import time
import traceback
import uuid
import chromadb
import numpy as np
import pandas as pd
import pypdf
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

# Optional PyTesseract import for offline OCR
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Optional Google Generative AI Import for Streaming & Fallback Vision
try:
    import google.generativeai as genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Import custom ARGO service if present
try:
    from argo_service import fetch_region_data
except ImportError:
    def fetch_region_data():
        return pd.DataFrame()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "floatchat_full_multi_intent_routing_2026")
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(days=30)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".csv", ".json", ".dat"}

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and len(GROQ_API_KEY) > 10) else None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ==============================================================================
# 1. SQLITE RELATIONAL DB
# ==============================================================================
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "floatchat_users.db")

def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

def init_relational_db():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"DB Init Exception: {e}")
    finally:
        if conn:
            conn.close()

init_relational_db()

# ==============================================================================
# 2. CHROMADB VECTOR DATABASE
# ==============================================================================
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
history_collection = chroma_client.get_or_create_collection(name="floatchat_multilingual_history")

def save_to_chromadb(prompt: str, reply: str, chart_json: str, timestamp_str: str, user: str = "guest", share_id: str = None):
    try:
        entry_id = share_id if share_id else str(uuid.uuid4())
        created_at = int(time.time())
        metadata = {
            "reply": reply,
            "chart_json": chart_json if chart_json else "",
            "time": timestamp_str,
            "created_at": created_at,
            "user": user,
            "share_id": entry_id
        }
        history_collection.upsert(
            documents=[prompt],
            metadatas=[metadata],
            ids=[entry_id]
        )
        return entry_id
    except Exception as e:
        print(f"ChromaDB Save Error: {e}")
        return None

def get_history_from_chromadb(current_user: str = "guest"):
    try:
        results = history_collection.get()
        sessions = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i]
                if meta.get("user", "guest") == current_user:
                    sessions.append({
                        "id": results["ids"][i],
                        "prompt": results["documents"][i],
                        "reply": meta.get("reply", ""),
                        "chart": meta.get("chart_json", None) if meta.get("chart_json") else None,
                        "time": meta.get("time", ""),
                        "created_at": meta.get("created_at", 0),
                        "share_id": meta.get("share_id", results["ids"][i])
                    })
            sessions.sort(key=lambda x: x["created_at"])
        return sessions
    except Exception as e:
        print(f"ChromaDB Fetch Error: {e}")
        return []

# ==============================================================================
# 3. GLOBAL CLIMATOLOGY MATRIX & BASIN CLASSIFIER
# ==============================================================================
GLOBAL_FLOAT_DATASET = [
    {"id": "5906001", "name": "Float 5906001 (Equatorial Indian Ocean)", "lat": 1.5, "lon": 65.4, "basin": "Equatorial Indian Ocean", "sst": 29.1, "sal": 35.1, "doxy": 64.0, "last_obs": "2025-11-14"},
    {"id": "5906002", "name": "Float 5906002 (South Equatorial Indian)", "lat": -3.2, "lon": 68.0, "basin": "Equatorial Indian Ocean", "sst": 28.7, "sal": 35.2, "doxy": 78.5, "last_obs": "2025-12-22"},
    {"id": "5904421", "name": "Float 5904421 (Equatorial Pacific Warm Pool)", "lat": 0.5, "lon": 165.0, "basin": "Equatorial Pacific", "sst": 29.8, "sal": 34.4, "doxy": 180.2, "last_obs": "2026-02-02"},
    {"id": "5904422", "name": "Float 5904422 (Eastern Equatorial Pacific)", "lat": -1.8, "lon": -110.5, "basin": "Equatorial Pacific", "sst": 25.4, "sal": 35.0, "doxy": 110.4, "last_obs": "2025-10-18"},
    {"id": "6902910", "name": "Float 6902910 (Tropical / Equatorial Atlantic)", "lat": 2.4, "lon": -28.0, "basin": "Equatorial Atlantic", "sst": 27.8, "sal": 35.7, "doxy": 165.0, "last_obs": "2026-01-11"},
    {"id": "5905081", "name": "Float 5905081 (North Arabian Sea)", "lat": 21.4, "lon": 64.2, "basin": "North Arabian Sea", "sst": 28.6, "sal": 36.8, "doxy": 4.2, "last_obs": "2026-03-10"},
    {"id": "5905082", "name": "Float 5905082 (Central Arabian Sea)", "lat": 16.8, "lon": 66.5, "basin": "Central Arabian Sea", "sst": 28.1, "sal": 36.4, "doxy": 6.8, "last_obs": "2026-04-04"},
    {"id": "5905083", "name": "Float 5905083 (Lakshadweep Basin)", "lat": 11.2, "lon": 72.4, "basin": "Lakshadweep Sea", "sst": 28.9, "sal": 35.8, "doxy": 22.4, "last_obs": "2026-05-29"},
    {"id": "2902781", "name": "Float 2902781 (Bay of Bengal)", "lat": 15.2, "lon": 88.5, "basin": "Bay of Bengal", "sst": 29.4, "sal": 33.2, "doxy": 18.2, "last_obs": "2026-05-15"},
    {"id": "4903215", "name": "Float 4903215 (North Pacific Gyre)", "lat": 32.5, "lon": -145.0, "basin": "North Pacific Ocean", "sst": 18.2, "sal": 34.6, "doxy": 195.0, "last_obs": "2025-11-30"},
    {"id": "3901920", "name": "Float 3901920 (South Pacific Gyre)", "lat": -28.0, "lon": -110.0, "basin": "South Pacific Ocean", "sst": 21.0, "sal": 35.4, "doxy": 210.0, "last_obs": "2025-09-12"},
    {"id": "6901840", "name": "Float 6901840 (Gulf Stream Extension)", "lat": 38.5, "lon": -55.0, "basin": "North Atlantic Ocean", "sst": 17.5, "sal": 36.1, "doxy": 235.0, "last_obs": "2026-04-01"},
    {"id": "5906800", "name": "Float 5906800 (Antarctic Circumpolar)", "lat": -58.5, "lon": 20.0, "basin": "Southern Ocean", "sst": 1.4, "sal": 34.1, "doxy": 310.0, "last_obs": "2025-12-25"},
    {"id": "4901500", "name": "Float 4901500 (Fram Strait / Arctic)", "lat": 78.0, "lon": 8.0, "basin": "Arctic Ocean", "sst": -0.5, "sal": 32.8, "doxy": 340.0, "last_obs": "2026-03-04"}
]

OCEAN_LANDMARKS = [
    {"name": "Mariana Trench (Challenger Deep)", "lat": 11.35, "lon": 142.20, "type": "Abyssal Trench", "depth": "10,994 m", "temp": 2.1},
    {"name": "Oman Upwelling (Arabian Sea OMZ)", "lat": 21.40, "lon": 64.20, "type": "Severe Hypoxic Dead Zone", "depth": "3,400 m", "temp": 28.6},
    {"name": "Galapagos Equatorial Front", "lat": 0.00, "lon": -90.50, "type": "Thermal Divergence Upwelling", "depth": "2,800 m", "temp": 22.4},
    {"name": "Agulhas Retroflection Current", "lat": -38.50, "lon": 18.00, "type": "Mesoscale Eddy Ring Corridor", "depth": "4,200 m", "temp": 16.8},
    {"name": "Drake Passage Jet", "lat": -58.50, "lon": -65.00, "type": "Antarctic Circumpolar Chokepoint", "depth": "4,100 m", "temp": 1.8}
]

def resolve_hydrography_at_coords(lat: float, lon: float):
    distances = [((f["lat"] - lat)**2 + (f["lon"] - lon)**2) for f in GLOBAL_FLOAT_DATASET]
    nearest_idx = int(np.argmin(distances))
    return GLOBAL_FLOAT_DATASET[nearest_idx]

def get_ocean_basin_name(lat: float, lon: float, lang: str = "en") -> str:
    if lat < -50.0:
        return "दक्षिणी महासागर" if lang == "hi" else ("दक्षिण महासागर" if lang == "mr" else "Southern Ocean")
    if lat > 65.0:
        return "आर्कटिक महासागर" if lang in ["hi", "mr"] else "Arctic Ocean"
    if -50.0 <= lat <= 30.0 and 45.0 <= lon <= 100.0:
        if 10.0 <= lat <= 30.0 and 50.0 <= lon <= 78.0:
            return "अरब सागर" if lang == "hi" else ("अरबी समुद्र" if lang == "mr" else "Arabian Sea")
        elif 5.0 <= lat <= 25.0 and 80.0 <= lon <= 100.0:
            return "बंगाल की खाड़ी" if lang == "hi" else ("बंगालचा उपसागर" if lang == "mr" else "Bay of Bengal")
        return "भूमध्य हिन्द महासागर" if lang == "hi" else ("विषुववृत्तीय हिंदी महासागर" if lang == "mr" else "Equatorial Indian Ocean")
    if -50.0 <= lat <= 65.0 and (-70.0 <= lon <= 20.0 or lon <= -100.0 and lat > 10.0):
        return ("उत्तरी अटलांटिक महासागर" if lat >= 0 else "दक्षिणी अटलांटिक महासागर") if lang == "hi" else (
            ("उत्तर अटलांटिक महासागर" if lat >= 0 else "दक्षिण अटलांटिक महासागर") if lang == "mr" else (
                "North Atlantic Ocean" if lat >= 0 else "South Atlantic Ocean"
            )
        )
    return ("उत्तरी प्रशांत महासागर" if lat >= 0 else "दक्षिणी प्रशांत महासागर") if lang == "hi" else (
        ("उत्तर पॅसिफिक महासागर" if lat >= 0 else "दक्षिण पॅसिफिक महासागर") if lang == "mr" else (
            "North Pacific Ocean" if lat >= 0 else "South Pacific Ocean"
        )
    )

# ==============================================================================
# 4. LATITUDE & DEPTH-DEPENDENT OCEANOGRAPHIC FORMULAS
# ==============================================================================
def calc_temperature_at_depth(depth_dbar: float, lat: float = 18.0, lon: float = 65.0) -> float:
    abs_lat = abs(lat)
    if abs_lat <= 15.0:
        sst = 29.5 - (abs_lat * 0.08)
    elif abs_lat <= 40.0:
        sst = 28.3 - ((abs_lat - 15.0) * 0.45)
    elif abs_lat <= 60.0:
        sst = 17.0 - ((abs_lat - 40.0) * 0.70)
    else:
        sst = 3.0 - ((abs_lat - 60.0) * 0.18)

    deep_t = 1.5 if abs_lat > 50.0 else 2.1
    if depth_dbar >= 4000.0:
        return float(np.round(deep_t + ((depth_dbar - 4000.0) * 0.00004), 2))

    if abs_lat > 60.0:
        if depth_dbar <= 100.0:
            val = sst - (depth_dbar * 0.012)
        else:
            val = (sst - 1.2) - ((depth_dbar - 100.0) * (sst - 1.2 - deep_t) / 1900.0)
    elif abs_lat > 35.0:
        if depth_dbar <= 40.0:
            val = sst - (depth_dbar * 0.018)
        elif depth_dbar <= 150.0:
            val = (sst - 0.72) - ((depth_dbar - 40.0) * (sst - 0.72 - 12.5) / 110.0)
        elif depth_dbar <= 500.0:
            val = 12.5 - ((depth_dbar - 150.0) * (12.5 - 6.2) / 350.0)
        else:
            val = 6.2 - ((depth_dbar - 500.0) * (6.2 - deep_t) / 1500.0)
    else:
        if depth_dbar <= 30.0:
            val = sst - (depth_dbar * 0.015)
        elif depth_dbar <= 100.0:
            t_100 = 22.16 + ((18.0 - abs_lat) * 0.35)
            val = (sst - 0.45) - ((depth_dbar - 30.0) * ((sst - 0.45) - t_100) / 70.0)
        elif depth_dbar <= 200.0:
            t_100 = 22.16 + ((18.0 - abs_lat) * 0.35)
            val = t_100 - ((depth_dbar - 100.0) * (t_100 - 15.20) / 100.0)
        elif depth_dbar <= 500.0:
            val = 15.20 - ((depth_dbar - 200.0) * (15.20 - 11.40) / 300.0)
        elif depth_dbar <= 1000.0:
            val = 11.40 - ((depth_dbar - 500.0) * (11.40 - 5.80) / 500.0)
        else:
            val = 5.80 - ((depth_dbar - 1000.0) * (5.80 - deep_t) / 3000.0)

    return float(np.round(val, 2))

def calc_salinity_at_depth(depth_dbar: float, lat: float = 18.0, lon: float = 65.0) -> float:
    abs_lat = abs(lat)
    if 10.0 <= lat <= 26.0 and 50.0 <= lon <= 75.0:
        base_sal = 36.45
    elif 5.0 <= lat <= 25.0 and 80.0 <= lon <= 100.0:
        base_sal = 33.20
    elif abs_lat > 55.0:
        base_sal = 33.80
    else:
        base_sal = 35.20

    if depth_dbar <= 100.0:
        return float(np.round(base_sal - (depth_dbar * 0.003), 2))
    elif depth_dbar <= 500.0:
        return float(np.round(base_sal - 0.30 - ((depth_dbar - 100.0) * 0.45 / 400.0), 2))
    else:
        return float(np.round(34.80 - ((depth_dbar - 500.0) * 0.15 / 1500.0), 2))

def calc_doxy_at_depth(depth_dbar: float, lat: float = 18.0, lon: float = 65.0) -> float:
    if 10.0 <= lat <= 26.0 and 50.0 <= lon <= 75.0:
        if depth_dbar <= 80.0:
            return float(np.round(210.0 - (depth_dbar * 1.8), 2))
        elif depth_dbar <= 900.0:
            return float(np.round(4.2 + (depth_dbar * 0.005), 2))
        else:
            return float(np.round(10.0 + ((depth_dbar - 900.0) * 0.12), 2))
    else:
        if depth_dbar <= 100.0:
            return float(np.round(225.0 - (depth_dbar * 0.4), 2))
        elif depth_dbar <= 800.0:
            return float(np.round(180.0 - ((depth_dbar - 100.0) * 0.15), 2))
        else:
            return float(np.round(75.0 + ((depth_dbar - 800.0) * 0.11), 2))

# ==============================================================================
# 5. DYNAMIC CONFIDENCE SCORE ENGINE
# ==============================================================================
def compute_hydrographic_confidence(lat: float, lon: float, depth_dbar: float, param_type: str = "temperature", is_ocr: bool = False) -> float:
    nearest = resolve_hydrography_at_coords(lat, lon)
    spatial_dist = np.sqrt((nearest["lat"] - lat)**2 + (nearest["lon"] - lon)**2)
    spatial_score = max(68.0, 99.4 - (spatial_dist * 4.2))
    
    if depth_dbar <= 40.0:
        depth_score = 98.6
    elif depth_dbar <= 200.0:
        depth_score = 95.8
    elif depth_dbar <= 1000.0:
        depth_score = 93.2
    else:
        depth_score = 90.0

    param_uncertainty = {
        "temperature": 1.0,
        "salinity": 0.985,
        "doxy": 0.925,
        "thermocline": 0.965,
        "comprehensive": 0.975,
        "abyssal": 0.960,
        "equator": 0.975
    }.get(param_type.lower(), 1.0)

    ocr_mod = 0.985 if is_ocr else 1.0
    raw_confidence = ((0.55 * spatial_score) + (0.45 * depth_score)) * param_uncertainty * ocr_mod
    return float(np.round(np.clip(raw_confidence, 65.0, 99.4), 1))

# ==============================================================================
# 6. OCR & MULTILINGUAL PARSER
# ==============================================================================
def extract_all_coordinates(text: str):
    coord_pattern = r'(-?\d+(?:\.\d+)?)\s*(?:°|deg)?\s*([NSns])\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:°|deg)?\s*([EWew])'
    matches = re.findall(coord_pattern, text)
    coords = []
    for lat_val, lat_dir, lon_val, lon_dir in matches:
        lat = -abs(float(lat_val)) if lat_dir.upper() == 'S' else abs(float(lat_val))
        lon = -abs(float(lon_val)) if lon_dir.upper() == 'W' else abs(float(lon_val))
        coords.append((lat, lon))
    
    if not coords:
        pair_pattern = r'(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)'
        for lat_val, lon_val in re.findall(pair_pattern, text):
            lat = float(lat_val)
            lon = float(lon_val)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords.append((lat, lon))
    return coords

def extract_coordinates_and_depth(text: str, default_lat: float = 18.0, default_lon: float = 65.0):
    t_low = text.lower()
    if "mariana" in t_low or "challenger deep" in t_low:
        default_lat = 11.35
        default_lon = 142.20
    elif "oman" in t_low or "upwelling" in t_low:
        default_lat = 21.40
        default_lon = 64.20
    elif "arabian sea" in t_low or "arabian" in t_low or "अरब सागर" in t_low or "अरबी समुद्र" in t_low:
        default_lat = 18.0
        default_lon = 65.0
    elif "bay of bengal" in t_low or "बंगाल" in t_low:
        default_lat = 15.2
        default_lon = 88.5
    elif "equatorial indian" in t_low or "हिंदी महासागर" in t_low:
        default_lat = 1.5
        default_lon = 65.4

    all_c = extract_all_coordinates(text)
    if all_c:
        target_lat, target_lon = all_c[0]
    else:
        target_lat, target_lon = default_lat, default_lon

    target_lat = max(-90.0, min(90.0, target_lat))
    target_lon = max(-180.0, min(180.0, target_lon))

    pres_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:dbar|dbr|db|m|meters|bar|मीटर|डीबार|खोली)\b', text, re.IGNORECASE)
    target_pres = float(pres_m.group(1)) if pres_m else 100.0

    return target_lat, target_lon, target_pres

def detect_query_lang(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["आहे", "काय", "मधील", "सरासरी", "दाखवा", "किती", "झोन", "पाण्याचे", "सांगा", "माहिती", "खोलीवर", "सागरी", "समुद्रातील", "फरक", "तुलना", "जवळ", "करा"]):
        return "mr"
    if any(k in t for k in ["है", "क्या", "का", "औसत", "दिखाइए", "कितना", "बताइए", "गहराई", "तापमान", "लवणता", "सागर", "पर"]):
        return "hi"
    return "en"

# ==============================================================================
# 7. 3D VISUALIZERS & CONTEXT-AWARE FOLLOW-UP PARSER
# ==============================================================================
def render_full_dedicated_3d_globe():
    df_floats = pd.DataFrame(GLOBAL_FLOAT_DATASET)
    df_landmarks = pd.DataFrame(OCEAN_LANDMARKS)
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=df_floats["lat"].tolist(), lon=df_floats["lon"].tolist(), mode="markers",
        marker=dict(size=11, color=df_floats["sst"].tolist(), colorscale="Plasma", showscale=True)
    ))
    fig.update_geos(projection_type="orthographic", showland=True, landcolor="#84cc16", showocean=True, oceancolor="#0284c7")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10))
    return fig.to_json()

def build_obs_card(station):
    return f"""
    <div class="p-2.5 bg-slate-900/90 border border-cyan-500/40 rounded-xl mb-2 shadow-md">
        <span class="text-xs font-extrabold text-cyan-300 block mb-0.5">📍 Station: {station['name']}</span>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-1 text-[11px] text-slate-300 font-mono">
            <div>Lat/Lon: <strong class="text-white">{station['lat']}°, {station['lon']}°</strong></div>
            <div>SST: <strong class="text-amber-400">{station['sst']} °C</strong></div>
            <div>Salinity: <strong class="text-sky-300">{station['sal']} PSU</strong></div>
            <div>Cast Date: <strong class="text-emerald-300">{station.get('last_obs', '2025-11-14')}</strong></div>
        </div>
    </div>
    """

def process_ocean_query(combined_text: str, default_lat: float = 18.0, default_lon: float = 65.0, lang: str = "en", last_coords=None):
    t_low = combined_text.lower()

    # Context-aware follow-up handling for things like "Compare Salinity" after a comparison
    is_followup_salinity = any(k in t_low for k in ["compare salinity", "salinity compare", "क्षारता तुलना"])
    if is_followup_salinity and last_coords and len(last_coords) >= 2:
        c1, c2 = last_coords[0], last_coords[1]
    else:
        coords = extract_all_coordinates(combined_text)
        if len(coords) >= 2:
            c1, c2 = coords[0], coords[1]
        elif last_coords and len(last_coords) >= 2 and any(k in t_low for k in ["compare", "vs", "versus", "salinity", "temperature"]):
            c1, c2 = last_coords[0], last_coords[1]
        else:
            c1, c2 = (15.0, 65.0), (10.0, 70.0)

    is_trench = any(k in t_low for k in ["mariana", "challenger", "trench", "abyssal", "गर्त", "खंदक"])
    is_thermocline = any(k in t_low for k in ["thermocline", "dt/dz", "gradient", "थर्मोक्लाइन", "तापमान ग्रेडियंट"])
    is_comparison = any(k in t_low for k in ["compare", "vs", "versus", "तुलना", "फरक", "salinity"]) or len(coords) >= 2 or is_followup_salinity
    is_equator = any(k in t_low for k in ["within 5", "equator", "observations within", "find argo observations", "विषुववृत्ताच्या"])

    if is_trench:
        target_lat, target_lon, target_pres = extract_coordinates_and_depth(combined_text, 11.35, 142.20)
        confidence = compute_hydrographic_confidence(target_lat, target_lon, 10000.0, "abyssal")
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5],
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("Abyssal Trench CTD Profile", "Hydrostatic Pressure (MPa)")
        )
        fig.add_trace(go.Scatter3d(x=[0]*20, y=[2]*20, z=list(range(-20, 0)), mode="lines", line=dict(color="#38bdf8", width=7), name="Temperature"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=[0]*20, y=[112]*20, z=list(range(-20, 0)), mode="lines", line=dict(color="#a855f7", width=7), name="Pressure"), row=1, col=2)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        
        primary_highlight = """
        <div class="p-3 mb-3 bg-gradient-to-r from-amber-950 via-slate-900 to-slate-900 border-l-4 border-amber-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-amber-400 font-bold block mb-0.5">⭐ Featured Primary Answer</span>
            <p class="text-sm font-extrabold text-white">Region: <span class="text-cyan-300 font-mono">Mariana Trench (Challenger Deep)</span> | Max Depth: <span class="text-amber-300 font-mono">10,994 m</span></p>
        </div>
        """
        confidence_badge = f"""
        <div class="flex items-center gap-2 mb-2.5">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
                <i class="fa-solid fa-shield-halved"></i> Confidence Score: {confidence}%
            </span>
        </div>
        """
        obs_card = """
        <div class="p-2.5 bg-slate-900/90 border border-amber-500/40 rounded-xl mb-2 shadow-md">
            <span class="text-xs font-extrabold text-amber-300 block mb-0.5">📍 Abyssal Record: Challenger Deep</span>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-1 text-[11px] text-slate-300 font-mono">
                <div>Coords: <strong class="text-white">11.35°N, 142.20°E</strong></div>
                <div>Temp: <strong class="text-amber-400">2.1 °C</strong></div>
                <div>Depth: <strong class="text-sky-300">10,994 m</strong></div>
                <div>Pressure: <strong class="text-purple-300">112.1 MPa</strong></div>
            </div>
        </div>
        """
        resp = primary_highlight + confidence_badge + f'<div><p class="text-xs font-semibold text-slate-200 mb-2">📋 Observation Record:</p><div class="mb-3">{obs_card}</div></div>'
        return resp, fig.to_json(), [c1, c2]

    elif is_thermocline:
        target_lat, target_lon, target_pres = extract_coordinates_and_depth(combined_text, default_lat, default_lon)
        basin_name = get_ocean_basin_name(target_lat, target_lon, lang=lang)
        confidence = compute_hydrographic_confidence(target_lat, target_lon, target_pres, param_type="thermocline")
        pressures = np.linspace(0, 500, 50)
        temperatures = [calc_temperature_at_depth(p, target_lat, target_lon) for p in pressures]
        gradients = -np.gradient(temperatures, pressures)
        thermocline_depth = float(pressures[int(np.argmax(gradients))])

        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5],
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("3D Temperature Profile", "Thermal Gradient (dT/dz)")
        )
        fig.add_trace(go.Scatter3d(x=[0]*50, y=temperatures, z=(-pressures).tolist(), mode="lines", line=dict(color="#f43f5e", width=7), name="Temperature"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=[0]*50, y=gradients.tolist(), z=(-pressures).tolist(), mode="lines", line=dict(color="#38bdf8", width=7), name="Gradient"), row=1, col=2)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)

        nearest_obs = resolve_hydrography_at_coords(target_lat, target_lon)
        obs_card = build_obs_card(nearest_obs)
        primary_highlight = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-cyan-950 via-slate-900 to-slate-900 border-l-4 border-cyan-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-bold block mb-0.5">⭐ Featured Primary Answer</span>
            <p class="text-sm font-extrabold text-white">Region: <span class="text-cyan-300 font-mono">{basin_name} ({target_lat:.1f}°N, {target_lon:.1f}°E)</span> | Thermocline Core: <span class="text-amber-300 font-mono">~{thermocline_depth:.0f} dbar</span></p>
        </div>
        """
        confidence_badge = f"""
        <div class="flex items-center gap-2 mb-2.5">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
                <i class="fa-solid fa-shield-halved"></i> Confidence Score: {confidence}%
            </span>
        </div>
        """
        resp = primary_highlight + confidence_badge + f'<div><p class="text-xs font-semibold text-slate-200 mb-2">📋 Hydrographic Observation Record:</p><div class="mb-3">{obs_card}</div></div>'
        return resp, fig.to_json(), [c1, c2]

    elif is_comparison:
        basin1, basin2 = get_ocean_basin_name(c1[0], c1[1]), get_ocean_basin_name(c2[0], c2[1])
        is_salinity_comp = "salinity" in t_low
        
        if is_salinity_comp:
            vals1 = [calc_salinity_at_depth(p, c1[0], c1[1]) for p in np.linspace(0, 500, 30)]
            vals2 = [calc_salinity_at_depth(p, c2[0], c2[1]) for p in np.linspace(0, 500, 30)]
            param_label = "ΔSalinity"
            unit_label = "PSU"
        else:
            vals1 = [calc_temperature_at_depth(p, c1[0], c1[1]) for p in np.linspace(0, 500, 30)]
            vals2 = [calc_temperature_at_depth(p, c2[0], c2[1]) for p in np.linspace(0, 500, 30)]
            param_label = "ΔSST"
            unit_label = "°C"

        confidence = 93.5

        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5],
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=(f"Station 1 ({c1[0]}°N, {c1[1]}°E)", f"Station 2 ({c2[0]}°N, {c2[1]}°E)")
        )
        fig.add_trace(go.Scatter3d(x=[0]*30, y=vals1, z=list(range(-30, 0)), mode="lines", line=dict(color="#f43f5e", width=7), name="Station 1"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=[0]*30, y=vals2, z=list(range(-30, 0)), mode="lines", line=dict(color="#38bdf8", width=7), name="Station 2"), row=1, col=2)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)

        obs_card = f"""
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-3">
            <div class="p-2.5 bg-slate-900/90 border border-rose-500/40 rounded-xl shadow-inner">
                <span class="text-xs font-bold text-rose-400 block mb-1">📍 Station 1 ({c1[0]}°N, {c1[1]}°E)</span>
                <p class="text-xs text-slate-200 font-mono">Value: <strong>{vals1[0]:.2f} {unit_label}</strong> | Basin: {basin1}</p>
            </div>
            <div class="p-2.5 bg-slate-900/90 border border-sky-500/40 rounded-xl shadow-inner">
                <span class="text-xs font-bold text-sky-400 block mb-1">📍 Station 2 ({c2[0]}°N, {c2[1]}°E)</span>
                <p class="text-xs text-slate-200 font-mono">Value: <strong>{vals2[0]:.2f} {unit_label}</strong> | Basin: {basin2}</p>
            </div>
        </div>
        """
        primary_highlight = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-sky-950 via-slate-900 to-slate-900 border-l-4 border-sky-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-sky-400 font-bold block mb-0.5">⭐ Featured Primary Answer</span>
            <p class="text-sm font-extrabold text-white">Regions: <span class="text-cyan-300 font-mono">{basin1} vs {basin2}</span> | {param_label}: <span class="text-amber-300 font-mono">{abs(vals2[0]-vals1[0]):.2f} {unit_label}</span></p>
        </div>
        """
        confidence_badge = f"""
        <div class="flex items-center gap-2 mb-2.5">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
                <i class="fa-solid fa-shield-halved"></i> Confidence Score: {confidence}%
            </span>
        </div>
        """
        resp = primary_highlight + confidence_badge + f'<div><p class="text-xs font-semibold text-slate-200 mb-2">📋 Comparison Records:</p>{obs_card}</div>'
        return resp, fig.to_json(), [c1, c2]

    elif is_equator:
        equatorial_floats = [f for f in GLOBAL_FLOAT_DATASET if abs(f["lat"]) <= 5.0]
        df_eq = pd.DataFrame(equatorial_floats)
        confidence = compute_hydrographic_confidence(0.0, 70.0, 10.0, param_type="equator")
        fig = go.Figure(go.Scattergeo(lat=df_eq["lat"].tolist(), lon=df_eq["lon"].tolist(), mode="markers+text", text=df_eq["id"].tolist(), marker=dict(size=14, color=df_eq["sst"].tolist(), colorscale="Plasma")))
        fig.update_geos(projection_type="orthographic", showland=True, landcolor="#84cc16", showocean=True, oceancolor="#0284c7")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", margin=dict(l=10, r=10, t=30, b=10), title=dict(text="Equatorial ARGO Station Cluster", font=dict(size=13, color="#ffffff")))

        float_cards_html = "".join([build_obs_card(f) for f in equatorial_floats])
        primary_highlight = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-cyan-950 via-slate-900 to-slate-900 border-l-4 border-cyan-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-bold block mb-0.5">⭐ Featured Primary Answer</span>
            <p class="text-sm font-extrabold text-white">Region: <span class="text-cyan-300 font-mono">Global Equatorial Zone (±5°)</span> | Active Stations: <span class="text-amber-300 font-mono">{len(equatorial_floats)} Floats</span></p>
        </div>
        """
        confidence_badge = f"""
        <div class="flex items-center gap-2 mb-2.5">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
                <i class="fa-solid fa-shield-halved"></i> Confidence Score: {confidence}%
            </span>
        </div>
        """
        resp = primary_highlight + confidence_badge + f'<div><p class="text-xs font-semibold text-slate-200 mb-2">📋 Station Breakdown:</p><div class="max-h-60 overflow-y-auto">{float_cards_html}</div></div>'
        return resp, fig.to_json(), [c1, c2]

    else:
        target_lat, target_lon, target_pres = extract_coordinates_and_depth(combined_text, default_lat, default_lon)
        basin_name = get_ocean_basin_name(target_lat, target_lon, lang=lang)
        avg_temp = calc_temperature_at_depth(target_pres, target_lat, target_lon)
        avg_sal = calc_salinity_at_depth(target_pres, target_lat, target_lon)
        confidence = compute_hydrographic_confidence(target_lat, target_lon, target_pres, "temperature")

        pressures = np.linspace(0, 1000, 30)
        t_vals = [calc_temperature_at_depth(p, target_lat, target_lon) for p in pressures]

        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5],
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("3D CTD Profile", "Station Array Grid")
        )
        fig.add_trace(go.Scatter3d(x=[0]*30, y=t_vals, z=list(range(-30, 0)), mode="lines", line=dict(color="#f43f5e", width=7), name="Temperature"), row=1, col=1)
        df_locs = pd.DataFrame(GLOBAL_FLOAT_DATASET[:6])
        fig.add_trace(go.Scatter3d(x=df_locs["lon"].tolist(), y=df_locs["lat"].tolist(), z=[0]*len(df_locs), mode="markers+text", text=df_locs["id"].tolist(), marker=dict(size=9, color=df_locs["sst"].tolist(), colorscale="Thermal"), name="Stations"), row=1, col=2)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)

        nearest_obs = resolve_hydrography_at_coords(target_lat, target_lon)
        obs_card = build_obs_card(nearest_obs)

        primary_highlight = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-rose-950 via-slate-900 to-slate-900 border-l-4 border-rose-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-rose-400 font-bold block mb-0.5">⭐ Featured Primary Answer</span>
            <p class="text-sm font-extrabold text-white">Region: <span class="text-cyan-300 font-mono">{basin_name} ({target_lat:.1f}°N, {target_lon:.1f}°E)</span> | Average Temperature at {target_pres:.0f} dbar: <span class="text-amber-300 font-mono">{avg_temp:.2f} °C</span></p>
        </div>
        """
        confidence_badge = f"""
        <div class="flex items-center gap-2 mb-2.5">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
                <i class="fa-solid fa-shield-halved"></i> Confidence Score: {confidence}%
            </span>
            <span class="text-[10px] text-slate-400 font-mono">CTD Spatial Interpolation Match</span>
        </div>
        """
        resp = primary_highlight + confidence_badge + f'<div><p class="text-xs font-semibold text-slate-200 mb-2">📋 Observation Record:</p><div class="mb-3">{obs_card}</div></div>'
        return resp, fig.to_json(), [c1, c2]

# ==============================================================================
# 8. APPLICATION ENDPOINTS
# ==============================================================================
@app.route("/login")
def login_page():
    if "user" in session: return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    if not username or not password: return jsonify({"success": False, "error": "Required fields missing."})
    try:
        conn = get_db_connection()
        conn.cursor().execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, generate_password_hash(password)))
        conn.commit()
        session.permanent = True
        session["user"] = username
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "Username taken."})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    conn = get_db_connection()
    row = conn.cursor().execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if row and check_password_hash(row[0], password):
        session.permanent = True
        session["user"] = username
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid credentials."})

@app.route("/")
def index():
    if "user" not in session: return redirect(url_for("login_page"))
    return render_template("chat.html", current_user=session["user"])

@app.route("/share/<share_id>", methods=["GET"])
def view_shared_convo(share_id):
    try:
        res = history_collection.get(ids=[share_id])
        if res and "documents" in res and len(res["documents"]) > 0:
            return render_template("shared.html", prompt=res["documents"][0], reply=res["metadatas"][0].get("reply", ""), chart=res["metadatas"][0].get("chart_json", ""))
    except Exception: pass
    return "Conversation not found.", 404

@app.route("/api/share", methods=["POST"])
def api_share_convo():
    try:
        data = request.get_json(silent=True) or {}
        share_id = str(uuid.uuid4())[:8]
        save_to_chromadb(data.get("prompt", ""), data.get("reply", ""), data.get("chart", ""), time.strftime("%I:%M %p"), user=session.get("user", "guest"), share_id=share_id)
        return jsonify({"success": True, "share_url": request.host_url.rstrip("/") + f"/share/{share_id}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"history": get_history_from_chromadb(session.get("user", "guest"))})

@app.route("/history/<entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    history_collection.delete(ids=[entry_id])
    return jsonify({"success": True})

@app.route("/history/clear", methods=["POST"])
def clear_all_history():
    res = history_collection.get()
    if res and "ids" in res:
        ids = [res["ids"][i] for i in range(len(res["ids"])) if res["metadatas"][i].get("user") == session.get("user", "guest")]
        if ids: history_collection.delete(ids=ids)
    return jsonify({"success": True})

@app.route("/api/globe/full", methods=["GET"])
def api_globe_full():
    return jsonify({"chart": json.loads(render_full_dedicated_3d_globe())})

@app.route("/chat", methods=["POST"])
@app.route("/get", methods=["POST"])
def chat():
    try:
        current_user = session.get("user", "guest")
        req_json = request.get_json(silent=True) or {}
        msg = (req_json.get("message") or req_json.get("msg") or request.form.get("msg") or "").strip()
        f_lat = float(req_json.get("lat", 18.0))
        f_lon = float(req_json.get("lon", 65.0))

        # Retrieve last session coordinates from ChromaDB history if available
        hist = get_history_from_chromadb(current_user)
        last_coords = None
        if hist:
            for h in reversed(hist):
                c = extract_all_coordinates(h["prompt"])
                if len(c) >= 2:
                    last_coords = c
                    break

        lang = detect_query_lang(msg)
        resp, chart, _ = process_ocean_query(msg, default_lat=f_lat, default_lon=f_lon, lang=lang, last_coords=last_coords)
        share_uuid = save_to_chromadb(msg, resp, chart, time.strftime("%I:%M %p"), user=current_user)
        return jsonify({"reply": resp, "chart": chart, "share_id": share_uuid})
    except Exception as err:
        return jsonify({"reply": f"⚠️ Error: {str(err)}", "chart": None})

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        current_user = session.get("user", "guest")
        file = request.files["file"]
        msg = request.form.get("message", "").strip()
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        
        doc_text = extract_text_from_any_file(path)
        clean_extracted_text = extract_exact_clean_questions(doc_text, msg)

        hist = get_history_from_chromadb(current_user)
        last_coords = None
        if hist:
            for h in reversed(hist):
                c = extract_all_coordinates(h["prompt"])
                if len(c) >= 2:
                    last_coords = c
                    break

        resp, chart, _ = process_ocean_query(clean_extracted_text, last_coords=last_coords)
        
        extracted_box_html = f"""
        <div class="p-3.5 mb-3.5 bg-cyan-950/80 border border-cyan-500/50 rounded-xl shadow-lg">
            <span class="text-[11px] font-bold uppercase tracking-wider text-cyan-400 block mb-1.5">
                <i class="fa-solid fa-wand-magic-sparkles mr-1 text-amber-400"></i> Extracted File Question:
            </span>
            <p class="text-xs text-cyan-200 font-mono font-bold">"{clean_extracted_text}"</p>
        </div>
        """
        final_reply = extracted_box_html + resp
        share_uuid = save_to_chromadb(clean_extracted_text, final_reply, chart, time.strftime("%I:%M %p"), user=current_user)
        return jsonify({"reply": final_reply, "chart": chart, "share_id": share_uuid})
    except Exception as e:
        return jsonify({"reply": f"⚠️ File Error: {str(e)}", "chart": None})

@app.route('/stream', methods=['POST'])
def stream():
    def generate():
        if not genai or not GEMINI_API_KEY:
            yield "Streaming unavailable."
            return
        model = genai.GenerativeModel('gemini-2.0-flash')
        for chunk in model.generate_content("Stream analysis", stream=True):
            if chunk.text: yield chunk.text
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)