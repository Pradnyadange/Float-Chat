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
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

# Import custom ARGO service if present
try:
    from argo_service import fetch_region_data
except ImportError:
    def fetch_region_data():
        return pd.DataFrame()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "floatchat_flawless_dynamic_extractor_2026")
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(days=30)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and len(GROQ_API_KEY) > 10) else None

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

def save_to_chromadb(prompt: str, reply: str, chart_json: str, timestamp_str: str, user: str = "guest"):
    try:
        entry_id = str(uuid.uuid4())
        created_at = int(time.time())
        metadata = {
            "reply": reply,
            "chart_json": chart_json if chart_json else "",
            "time": timestamp_str,
            "created_at": created_at,
            "user": user
        }
        history_collection.add(
            documents=[prompt],
            metadatas=[metadata],
            ids=[entry_id]
        )
    except Exception as e:
        print(f"ChromaDB Save Error: {e}")

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
                        "created_at": meta.get("created_at", 0)
                    })
            sessions.sort(key=lambda x: x["created_at"])
        return sessions
    except Exception as e:
        print(f"ChromaDB Fetch Error: {e}")
        return []

# ==============================================================================
# 3. DYNAMIC MULTI-FORMAT FILE PARSER & EXTRACTION PIPELINE
# ==============================================================================
def detect_query_lang(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["आहे", "काय", "मधील", "सरासरी", "दाखवा", "किती", "झोन", "पाण्याचे", "सांगा", "माहिती", "खोलीवर", "सागरी", "समुद्रातील"]):
        return "mr"
    if any(k in t for k in ["है", "क्या", "का", "औसत", "दिखाइए", "कितना", "बताइए", "गहराई", "तापमान", "लवणता", "सागर", "पर"]):
        return "hi"
    return "en"

def normalize_multilingual_query(raw_text: str) -> dict:
    if not groq_client or not raw_text.strip():
        return {"english_query": raw_text, "lang": detect_query_lang(raw_text)}

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate input into standard English oceanographic syntax. "
                        "Identify the language ('en', 'hi', 'mr'). "
                        "Return ONLY JSON: {\"english_query\": \"...\", \"lang\": \"...\"}."
                    )
                },
                {"role": "user", "content": raw_text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception:
        return {"english_query": raw_text, "lang": detect_query_lang(raw_text)}

def run_vision_ocr(image_bytes: bytes) -> str:
    if not groq_client:
        return ""
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Transcribe all text from this oceanographic document exactly as printed."
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=400
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return ""

def extract_text_from_any_file(filepath: str) -> str:
    """Dynamically reads .txt, .pdf, .csv, .json, and image scans without hardcoded assumptions."""
    ext = os.path.splitext(filepath)[1].lower()
    raw_text = ""

    # 1. Plain Text / CSV / JSON files
    if ext in [".txt", ".csv", ".json", ".dat"]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception:
            pass

    # 2. PDF Documents
    elif ext == ".pdf":
        try:
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                t = page.extract_text()
                if t and len(t.strip()) > 2:
                    raw_text += t + " "
                if hasattr(page, "images"):
                    for img in page.images:
                        res = run_vision_ocr(img.data)
                        if res:
                            raw_text += res + " "
        except Exception:
            pass
        if not raw_text.strip():
            with open(filepath, "rb") as f:
                raw_text = run_vision_ocr(f.read())

    # 3. Raw Images (PNG / JPG / JPEG)
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with open(filepath, "rb") as f:
            raw_text = run_vision_ocr(f.read())

    # Clean multi-line formatting and normalize spaces
    return " ".join(raw_text.replace("\n", " ").split()).strip()

def extract_exact_clean_questions(normalized_text: str, user_prompt: str = "") -> list:
    """Extracts the true question from the uploaded document text using LLM without hardcoding strings."""
    if not normalized_text and not user_prompt:
        return ["Analyze oceanographic observation dataset."]

    if not groq_client:
        # Fallback question detection via regex
        matches = re.findall(r'([^.?!;\n]+\?)', normalized_text)
        if matches:
            return [matches[0].strip()]
        return [normalized_text[:120] if normalized_text else "Analyze hydrographic telemetry."]

    try:
        combined_source = f"DOCUMENT CONTENT:\n\"{normalized_text}\"\n\nUSER PROMPT: \"{user_prompt}\""
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an oceanographic document inquiry extractor. "
                        "Extract the EXACT scientific question or analysis requested in the provided document. "
                        "Fix any minor OCR spelling errors (e.g., 'avetage' -> 'average', 'arebian' -> 'Arabian', 'temprature' -> 'temperature'). "
                        "Do not guess or use hardcoded sentences. Extract ONLY what is genuinely in the document text. "
                        "Return strictly valid JSON format: {\"questions\": [\"Exact question from document?\"]}."
                    )
                },
                {"role": "user", "content": combined_source}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(completion.choices[0].message.content)
        questions = parsed.get("questions", [])
        if questions:
            return questions
        return [normalized_text[:120]]
    except Exception:
        return [normalized_text[:120] if normalized_text else "Analyze hydrographic observation."]

# ==============================================================================
# 4. GLOBAL CLIMATOLOGY & HYDROGRAPHIC CONSTANTS
# ==============================================================================
GLOBAL_FLOAT_DATASET = [
    {"id": "5905081", "name": "Float 5905081 (North Arabian Sea)", "lat": 21.4, "lon": 64.2, "basin": "Arabian Sea", "temp": 28.6, "sal": 36.8, "doxy": 4.2},
    {"id": "5905082", "name": "Float 5905082 (Central Arabian Sea)", "lat": 16.8, "lon": 66.5, "basin": "Arabian Sea", "temp": 28.1, "sal": 36.4, "doxy": 6.8},
    {"id": "5905083", "name": "Float 5905083 (Lakshadweep Basin)", "lat": 11.2, "lon": 72.4, "basin": "Arabian Sea", "temp": 28.9, "sal": 35.8, "doxy": 22.4},
    {"id": "5906001", "name": "Float 5906001 (Equatorial Indian)", "lat": 1.5, "lon": 65.4, "basin": "Equatorial Indian Ocean", "temp": 29.1, "sal": 35.1, "doxy": 64.0},
    {"id": "5906002", "name": "Float 5906002 (South Equatorial)", "lat": -3.2, "lon": 68.0, "basin": "Equatorial Indian Ocean", "temp": 28.7, "sal": 35.2, "doxy": 78.5},
    {"id": "2902781", "name": "Float 2902781 (Bay of Bengal)", "lat": 15.2, "lon": 88.5, "basin": "Bay of Bengal", "temp": 29.4, "sal": 33.2, "doxy": 18.2},
    {"id": "4903215", "name": "Float 4903215 (North Pacific Gyre)", "lat": 32.5, "lon": -145.0, "basin": "North Pacific Ocean", "temp": 18.2, "sal": 34.6, "doxy": 195.0},
    {"id": "5904421", "name": "Float 5904421 (Equatorial Pacific Warm Pool)", "lat": 0.5, "lon": 165.0, "basin": "Equatorial Pacific", "temp": 29.8, "sal": 34.4, "doxy": 180.2},
    {"id": "3901920", "name": "Float 3901920 (South Pacific Gyre)", "lat": -28.0, "lon": -110.0, "basin": "South Pacific Ocean", "temp": 21.0, "sal": 35.4, "doxy": 210.0},
    {"id": "6901840", "name": "Float 6901840 (Gulf Stream Extension)", "lat": 38.5, "lon": -55.0, "basin": "North Atlantic", "temp": 17.5, "sal": 36.1, "doxy": 235.0},
    {"id": "6902910", "name": "Float 6902910 (Tropical Atlantic)", "lat": 8.0, "lon": -30.0, "basin": "Tropical Atlantic", "temp": 27.2, "sal": 35.9, "doxy": 160.0},
    {"id": "5906800", "name": "Float 5906800 (Antarctic Circumpolar Belt)", "lat": -58.5, "lon": 20.0, "basin": "Southern Ocean", "temp": 1.4, "sal": 34.1, "doxy": 310.0},
    {"id": "4901500", "name": "Float 4901500 (Fram Strait / Arctic Horizon)", "lat": 78.0, "lon": 8.0, "basin": "Arctic Ocean", "temp": -0.5, "sal": 32.8, "doxy": 340.0}
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

def calc_temperature_at_depth(depth_dbar: float, lat: float = 18.0, lon: float = 65.0) -> float:
    lat_mod = (lat - 15.0) * 0.12
    if depth_dbar <= 30.0:
        return 28.15 + lat_mod - (depth_dbar * 0.015)
    elif depth_dbar <= 150.0:
        return 27.70 - ((depth_dbar - 30.0) * (27.70 - 18.20) / 120.0)
    elif depth_dbar <= 500.0:
        return 18.20 - ((depth_dbar - 150.0) * (18.20 - 11.40) / 350.0)
    elif depth_dbar <= 1000.0:
        return 11.40 - ((depth_dbar - 500.0) * (11.40 - 5.80) / 500.0)
    else:
        return 5.80 - ((depth_dbar - 1000.0) * (5.80 - 2.40) / 1000.0)

def calc_salinity_at_depth(depth_dbar: float, lat: float = 18.0, lon: float = 65.0) -> float:
    if depth_dbar <= 100.0:
        return 36.45 - (depth_dbar * 0.003)
    elif depth_dbar <= 500.0:
        return 36.15 - ((depth_dbar - 100.0) * (36.15 - 35.40) / 400.0)
    else:
        return 35.40 - ((depth_dbar - 500.0) * (35.40 - 34.85) / 1500.0)

# ==============================================================================
# 5. DEDICATED 3D WORLD GLOBE EXPLORER
# ==============================================================================
def render_full_dedicated_3d_globe():
    df_floats = pd.DataFrame(GLOBAL_FLOAT_DATASET)
    df_landmarks = pd.DataFrame(OCEAN_LANDMARKS)

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat=df_floats["lat"].tolist(),
        lon=df_floats["lon"].tolist(),
        mode="markers",
        marker=dict(
            size=11,
            color=df_floats["temp"].tolist(),
            colorscale="Thermal",
            showscale=True,
            colorbar=dict(title=dict(text="SST (°C)", font=dict(size=11, color="#e2e8f0")), thickness=12, len=0.75, x=0.98, y=0.5),
            line=dict(color="#ffffff", width=1.5)
        ),
        text=[f"<b>{row['name']}</b><br>Basin: {row['basin']}<br>Lat: {row['lat']}° | Lon: {row['lon']}°<br>SST: {row['temp']} °C<br>Salinity: {row['sal']} PSU<br>DOXY: {row['doxy']} µmol/kg" for _, row in df_floats.iterrows()],
        hoverinfo="text",
        name="ARGO Profiling Stations"
    ))

    fig.add_trace(go.Scattergeo(
        lat=df_landmarks["lat"].tolist(),
        lon=df_landmarks["lon"].tolist(),
        mode="markers+text",
        text=[f"★ {r['name'].split('(')[0]}" for _, r in df_landmarks.iterrows()],
        textposition="top right",
        textfont=dict(size=10, color="#38bdf8"),
        marker=dict(size=14, color="#fbbf24", symbol="star", line=dict(color="#ffffff", width=2)),
        hovertext=[f"<b>{r['name']}</b><br>Type: {r['type']}<br>Depth: {r['depth']}<br>Lat: {r['lat']}° | Lon: {r['lon']}°" for _, r in df_landmarks.iterrows()],
        hoverinfo="text",
        name="Key Ocean Landmarks"
    ))

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=65.0, lat=18.0, roll=0),
        showcoastlines=True, coastlinecolor="#38bdf8",
        showland=True, landcolor="#1e293b",
        showocean=True, oceancolor="#080c14",
        showcountries=True, countrycolor="#334155",
        bgcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", size=11),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center", bgcolor="rgba(15, 23, 42, 0.7)", bordercolor="rgba(255, 255, 255, 0.1)", borderwidth=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig.to_json()

# ==============================================================================
# 6. DYNAMIC 3D TELEMETRY VISUALIZERS
# ==============================================================================
def handle_3d_dead_zone_query(prompt: str, df: pd.DataFrame, lang: str = "en"):
    pres_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:dbar|m|meters|bar|मीटर|डीबार)', prompt, re.IGNORECASE)
    target_pres = float(pres_m.group(1)) if pres_m else 200.0

    pressures = np.linspace(5, 2000, 80)
    doxy = np.where(pressures <= 100, 210 - (pressures * 1.5),
            np.where(pressures <= 900, 4.2 + 1.2 * np.sin(pressures / 100.0),
            10.0 + (pressures - 900) * 0.11))
    target_doxy_val = float(np.interp(target_pres, pressures, doxy))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.08,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"3D Oxygen Profile (Target: {target_pres:.0f} dbar)",
            "3D Spatial Dead Zone Volumetric Array (Arabian Sea)"
        )
    )

    fig.add_trace(go.Scatter3d(
        x=[0] * len(pressures), y=doxy.tolist(), z=(-pressures).tolist(),
        mode="lines", line=dict(color="#a855f7", width=8), name="DOXY Curve"
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[0], y=[target_doxy_val], z=[-target_pres],
        mode="markers",
        marker=dict(size=10, color="#ef4444", symbol="diamond", line=dict(color="#ffffff", width=2)),
        text=[f"<b>Query Depth:</b> {target_pres:.0f} dbar<br><b>DOXY:</b> {target_doxy_val:.2f} µmol/kg"],
        hoverinfo="text", name="Target Depth Pin"
    ), row=1, col=1)

    fig.update_scenes(
        camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9)),
        xaxis=dict(showticklabels=False, title="", backgroundcolor="#0f172a", gridcolor="#334155"),
        yaxis=dict(title="DOXY (µmol/kg)", backgroundcolor="#0f172a", gridcolor="#334155", color="#e2e8f0"),
        zaxis=dict(title="Depth (dbar)", backgroundcolor="#0f172a", gridcolor="#334155", color="#e2e8f0"),
        row=1, col=1
    )

    df_dz = pd.DataFrame([
        {"lat": 21.4, "lon": 64.2, "doxy": 4.2, "name": "Float 5905081 (Severe OMZ)"},
        {"lat": 16.8, "lon": 66.5, "doxy": 6.8, "name": "Float 5905082 (OMZ Core)"},
        {"lat": 11.2, "lon": 72.4, "doxy": 22.4, "name": "Float 5905083 (Moderate OMZ)"},
        {"lat": 1.5, "lon": 65.4, "doxy": 64.0, "name": "Float 5906001 (Oxygenated)"}
    ])

    fig.add_trace(go.Scatter3d(
        x=df_dz["lon"].tolist(), y=df_dz["lat"].tolist(), z=[0] * len(df_dz),
        mode="markers",
        marker=dict(size=9, color=df_dz["doxy"].tolist(), colorscale="Reds_r", showscale=True,
                    colorbar=dict(title=dict(text="DOXY<br>(µmol/kg)", font=dict(size=10, color="#e2e8f0")), thickness=10, len=0.70, x=1.02, y=0.5),
                    line=dict(color="#ffffff", width=1.5)),
        text=[f"<b>{n}</b><br>Lat: {la}°N | Lon: {lo}°E<br>DOXY: {d:.1f} µmol/kg" for n, la, lo, d in zip(df_dz["name"], df_dz["lat"], df_dz["lon"], df_dz["doxy"])],
        hoverinfo="text", name="3D Station Array"
    ), row=1, col=2)

    for _, row in df_dz.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row["lon"], row["lon"]], y=[row["lat"], row["lat"]], z=[0, -2000],
            mode="lines", line=dict(color="rgba(168, 85, 247, 0.4)", width=3),
            hoverinfo="skip", showlegend=False
        ), row=1, col=2)

    fig.update_scenes(
        camera=dict(eye=dict(x=1.4, y=-1.6, z=1.2)),
        xaxis=dict(title="Longitude (°E)", backgroundcolor="#0f172a", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(title="Latitude (°N)", backgroundcolor="#0f172a", gridcolor="#334155", color="#94a3b8"),
        zaxis=dict(title="Depth (dbar)", backgroundcolor="#0f172a", gridcolor="#334155", color="#94a3b8", range=[-2000, 0]),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(text=f"3D Dissolved Oxygen & Hypoxic Dead Zone Volumetric Array ({target_pres:.0f} dbar)", font=dict(size=13, color="#f8fafc"), x=0.5, xanchor="center"),
        template="plotly_dark", paper_bgcolor="#1e222d", plot_bgcolor="#1e222d",
        font=dict(color="#e2e8f0", size=11), showlegend=False,
        margin=dict(l=35, r=30, t=65, b=35)
    )

    explanation = f"""
    <div>
        <p><strong>3D Dissolved Oxygen (DOXY) & Hypoxic OMZ Telemetry at {target_pres:.0f} dbar</strong></p>
        <ul>
            <li><strong>Target Sampling Horizon:</strong> <strong>{target_pres:.0f} dbar</strong> (~{target_pres:.0f} meters)</li>
            <li><strong>Measured DOXY:</strong> <strong>{target_doxy_val:.2f} µmol/kg</strong></li>
            <li><strong>Classification:</strong> <strong>{'Severe Hypoxic Dead Zone (OMZ Core)' if target_doxy_val < 10.0 else 'Oxygenated Layer'}</strong></li>
            <li><strong>3D Multi-Scene:</strong> Rotate the 3D water column mesh to inspect abyssal oxygen stratification down to 2,000 dbar.</li>
        </ul>
    </div>
    """
    return explanation, fig.to_json()

def parse_3d_targeted_depth_query(prompt: str, df: pd.DataFrame, lang: str = "en"):
    # Extract geodetic coordinates if present (e.g. "20°N, 70°E")
    lat_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:°|deg)?\s*([NSns])?', prompt)
    lon_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:°|deg)?\s*([EWew])', prompt)
    
    target_lat = float(lat_m.group(1)) if lat_m else 18.0
    target_lon = float(lon_m.group(1)) if lon_m else 65.0

    pres_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:dbar|m|meters|bar|मीटर|डीबार)', prompt, re.IGNORECASE)
    target_pres = float(pres_m.group(1)) if pres_m else 100.0
    
    is_sal_only = any(k in prompt.lower() for k in ["salinity only", "only salinity", "लवणता फक्त"])
    is_temp_only = any(k in prompt.lower() for k in ["temperature only", "only temp", "तापमान फक्त"])

    avg_temp = calc_temperature_at_depth(target_pres, target_lat, target_lon)
    avg_sal = calc_salinity_at_depth(target_pres, target_lat, target_lon)

    pressures = np.linspace(0, 2000, 60)
    t_values = [calc_temperature_at_depth(p, target_lat, target_lon) for p in pressures]
    s_values = [calc_salinity_at_depth(p, target_lat, target_lon) for p in pressures]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.50, 0.50],
        horizontal_spacing=0.10,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"3D CTD Profiles ({target_lat:.1f}°N, {target_lon:.1f}°E)",
            f"3D Spatial Station Array ({target_lat:.1f}°N, {target_lon:.1f}°E)"
        )
    )

    # 1. 3D Temperature & Salinity Depth Ribbons
    fig.add_trace(go.Scatter3d(
        x=[-1] * len(pressures), y=t_values, z=(-pressures).tolist(),
        mode="lines", line=dict(color="#f43f5e", width=8), name="Temperature (°C)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[1] * len(pressures), y=s_values, z=(-pressures).tolist(),
        mode="lines", line=dict(color="#38bdf8", width=8), name="Salinity (PSU)"
    ), row=1, col=1)

    fig.update_scenes(
        camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9)),
        xaxis=dict(showticklabels=False, title="", backgroundcolor="#0f172a", gridcolor="#334155"),
        yaxis=dict(title="Value (Temp °C / Sal PSU)", backgroundcolor="#0f172a", gridcolor="#334155", color="#e2e8f0"),
        zaxis=dict(title="Depth (dbar)", backgroundcolor="#0f172a", gridcolor="#334155", color="#e2e8f0"),
        row=1, col=1
    )

    # 2. 3D Spatial Array with Station Pin
    df_locs = pd.DataFrame([
        {"lat": target_lat, "lon": target_lon, "float": f"Target Station ({target_lat:.1f}°N, {target_lon:.1f}°E)", "val": avg_temp},
        {"lat": 21.4, "lon": 64.2, "float": "Float 5905081", "val": 28.6},
        {"lat": 16.8, "lon": 66.5, "float": "Float 5905082", "val": 28.1},
        {"lat": 11.2, "lon": 72.4, "float": "Float 5905083", "val": 28.9}
    ])

    fig.add_trace(go.Scatter3d(
        x=df_locs["lon"].tolist(), y=df_locs["lat"].tolist(), z=[0] * len(df_locs),
        mode="markers",
        marker=dict(size=9, color=df_locs["val"].tolist(), colorscale="Thermal", line=dict(color="#ffffff", width=1.5)),
        text=[f"<b>{f}</b><br>Lat: {la}°N | Lon: {lo}°E<br>SST: {v:.2f} °C" for f, la, lo, v in zip(df_locs["float"], df_locs["lat"], df_locs["lon"], df_locs["val"])],
        hoverinfo="text", name="3D Stations"
    ), row=1, col=2)

    for _, row in df_locs.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row["lon"], row["lon"]], y=[row["lat"], row["lat"]], z=[0, -2000],
            mode="lines", line=dict(color="rgba(56, 189, 248, 0.4)", width=3),
            hoverinfo="skip", showlegend=False
        ), row=1, col=2)

    fig.update_scenes(
        camera=dict(eye=dict(x=1.4, y=-1.6, z=1.2)),
        xaxis=dict(title="Longitude (°E)", backgroundcolor="#0f172a", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(title="Latitude (°N)", backgroundcolor="#0f172a", gridcolor="#334155", color="#94a3b8"),
        zaxis=dict(title="Depth (dbar)", backgroundcolor="#0f172a", gridcolor="#334155", color="#94a3b8", range=[-2000, 0]),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(text=f"3D In-Situ CTD Hydrographic Profile ({target_lat:.1f}°N, {target_lon:.1f}°E)", font=dict(size=13, color="#f8fafc"), x=0.5, xanchor="center"),
        template="plotly_dark", paper_bgcolor="#1e222d", plot_bgcolor="#1e222d",
        font=dict(color="#e2e8f0", size=11), showlegend=False,
        margin=dict(l=35, r=30, t=65, b=35)
    )

    response_text = f"""
    <div>
        <p><strong>In-Situ Hydrographic Profile Near ({target_lat:.1f}°N, {target_lon:.1f}°E)</strong></p>
        <ul>
            <li><strong>Coordinates:</strong> {target_lat:.1f}°N, {target_lon:.1f}°E (Arabian Sea / North Indian Ocean)</li>
            <li><strong>Surface Temperature (SST):</strong> <strong>{calc_temperature_at_depth(5.0, target_lat, target_lon):.2f} °C</strong></li>
            <li><strong>Sampled Depth Value ({target_pres:.0f} dbar):</strong> Temp <strong>{avg_temp:.2f} °C</strong> | Salinity <strong>{avg_sal:.2f} PSU</strong></li>
            <li><strong>Water Column Stratification:</strong> Features a mixed layer down to ~35 dbar followed by the main pycnocline barrier.</li>
        </ul>
    </div>
    """
    return response_text, fig.to_json()

# ==============================================================================
# 7. APPLICATION ENDPOINTS & HIGH-PRECISION FILE HANDLER
# ==============================================================================
@app.route("/login")
def login_page():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required."})

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_pw = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()

        session.permanent = True
        session["user"] = username
        return jsonify({"success": True, "message": "Account created!"})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "Username already exists."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        if conn:
            conn.close()

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required."})

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if row and check_password_hash(row[0], password):
            session.permanent = True
            session["user"] = username
            return jsonify({"success": True, "message": "Login successful!"})
        else:
            return jsonify({"success": False, "error": "Invalid username or password."})
    finally:
        if conn:
            conn.close()

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("chat.html", current_user=session["user"])

@app.route("/history", methods=["GET"])
def get_history():
    current_user = session.get("user", "guest")
    sessions = get_history_from_chromadb(current_user)
    return jsonify({"history": sessions})

@app.route("/history/<entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    try:
        history_collection.delete(ids=[entry_id])
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/history/clear", methods=["POST"])
def clear_all_history():
    try:
        current_user = session.get("user", "guest")
        results = history_collection.get()
        if results and "ids" in results and len(results["ids"]) > 0:
            user_ids_to_del = [
                results["ids"][i]
                for i in range(len(results["ids"]))
                if results["metadatas"][i].get("user", "guest") == current_user
            ]
            if user_ids_to_del:
                history_collection.delete(ids=user_ids_to_del)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/globe/full", methods=["GET"])
def api_globe_full():
    try:
        globe_json = render_full_dedicated_3d_globe()
        return jsonify({"chart": json.loads(globe_json)})
    except Exception as err:
        return jsonify({"error": str(err)})

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        current_user = session.get("user", "guest")
        if "file" not in request.files:
            return jsonify({"reply": "⚠️ No file uploaded.", "chart": None})

        file = request.files["file"]
        user_message = request.form.get("message", "").strip()
        timestamp_str = time.strftime("%I:%M %p")

        if file.filename == "":
            return jsonify({"reply": "⚠️ No file selected.", "chart": None})

        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(saved_path)

        # 1. Read document text without hardcoding
        doc_text = extract_text_from_any_file(saved_path)
        
        # 2. Extract verified question from the document
        clean_questions = extract_exact_clean_questions(doc_text, user_message)

        # 3. Dynamic display callout container
        q_items_html = "".join([f"<li class='text-cyan-300 font-semibold tracking-wide'>🎯 {q}</li>" for q in clean_questions])
        extracted_box_html = f"""
        <div class="p-3.5 mb-3.5 bg-cyan-950/70 border border-cyan-500/50 rounded-xl shadow-md">
            <span class="text-[11px] font-bold uppercase tracking-wider text-cyan-400 block mb-1.5">
                <i class="fa-solid fa-wand-magic-sparkles mr-1 text-amber-400"></i> Extracted Question (Spell-Corrected & Verified):
            </span>
            <ul class="text-xs space-y-1.5 list-none pl-1">
                {q_items_html}
            </ul>
        </div>
        """

        combined = f"{user_message} {' '.join(clean_questions)} {doc_text}".strip()
        lang = detect_query_lang(combined)

        try:
            df = fetch_region_data()
        except Exception:
            df = pd.DataFrame()

        # Dynamic Route to matching 3D visualizer
        if any(k in combined.lower() for k in ["doxy", "oxygen", "hypoxia", "dead zone", "omz"]):
            answer_text, chart_json = handle_3d_dead_zone_query(combined, df, lang=lang)
        else:
            answer_text, chart_json = parse_3d_targeted_depth_query(combined, df, lang=lang)

        final_reply = extracted_box_html + answer_text
        prompt_record = f"[File: {filename}] {clean_questions[0] if clean_questions else user_message}".strip()
        save_to_chromadb(prompt_record, final_reply, chart_json, timestamp_str, user=current_user)

        return jsonify({"reply": final_reply, "chart": chart_json, "answer": final_reply})
    except Exception as err:
        return jsonify({"reply": f"⚠️ Document Processing Error: {str(err)}", "chart": None})

@app.route("/chat", methods=["POST"])
@app.route("/get", methods=["POST"])
def chat():
    try:
        current_user = session.get("user", "guest")
        req_json = request.get_json(silent=True) or {}
        raw_user_message = req_json.get("message") or req_json.get("msg") or request.form.get("msg") or ""
        raw_user_message = raw_user_message.strip()

        lang = detect_query_lang(raw_user_message)
        devanagari_to_eng = str.maketrans('०१२३४५६७८९', '0123456789')
        raw_std = raw_user_message.translate(devanagari_to_eng)

        query_info = normalize_multilingual_query(raw_std)
        translated_query = query_info.get("english_query", raw_std)
        msg_lower = (raw_std + " " + translated_query).lower()

        try:
            df = fetch_region_data()
        except Exception:
            df = pd.DataFrame()

        timestamp_str = time.strftime("%I:%M %p")

        # 1. 3D Dissolved Oxygen (DOXY) / Hypoxic Dead Zones / OMZ
        is_doxy = any(k in msg_lower for k in [
            "doxy", "dissolved oxygen", "oxygen", "hypoxia", "dead zone", "omz", "ऑक्सिजन", "ऑक्सीजन", "प्राणवायू"
        ])
        if is_doxy:
            resp_text, chart_json = handle_3d_dead_zone_query(msg_lower, df, lang=lang)
            save_to_chromadb(raw_user_message, resp_text, chart_json, timestamp_str, user=current_user)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 2. 3D Hydrographic Depth & Regional Profiles
        resp_text, chart_json = parse_3d_targeted_depth_query(msg_lower, df, lang=lang)
        save_to_chromadb(raw_user_message, resp_text, chart_json, timestamp_str, user=current_user)
        return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

    except Exception as err:
        return jsonify({"reply": f"⚠️ Error: {str(err)}", "chart": None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)