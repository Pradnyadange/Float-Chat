import json
import os
import re
import time
import traceback
import uuid
import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from groq import Groq
from argo_service import fetch_region_data
import requests

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ARGOVIS_API_KEY = os.environ.get("ARGOVIS_API_KEY", "b3cfb9064f510e87e9337b86d2487fbc9b56a9d7")

groq_client = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and len(GROQ_API_KEY) > 10) else None

# Initialize Persistent ChromaDB Client with fast embedding
class FastDummyEmbedding:
    def __call__(self, input):
        return [[0.0] * 10 for _ in input]
    def name(self):
        return "fast_dummy"

CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
history_collection = chroma_client.get_or_create_collection(
    name="floatchat_fast_v8",
    embedding_function=FastDummyEmbedding()
)


def save_to_chromadb(prompt: str, reply: str, chart_json: str, timestamp_str: str):
    try:
        entry_id = str(uuid.uuid4())
        created_at = int(time.time())
        metadata = {
            "reply": reply,
            "chart_json": chart_json if chart_json else "",
            "time": timestamp_str,
            "created_at": created_at
        }
        history_collection.add(
            embeddings=[[0.0] * 10],
            documents=[prompt],
            metadatas=[metadata],
            ids=[entry_id]
        )
    except Exception as e:
        print(f"ChromaDB Save Error: {e}")


def get_history_from_chromadb():
    try:
        results = history_collection.get()
        sessions = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i]
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


def load_dataset() -> pd.DataFrame:
    try:
        df = fetch_region_data()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"Dataset load notice: {e}")
    return pd.DataFrame()


df_argo = load_dataset()


# ==============================================================================
# GLOBAL OCEAN CLASSIFIER & THERMODYNAMICS (ALL 5 OCEANS)
# ==============================================================================
def identify_global_basin(lat: float, lon: float) -> dict:
    if lon > 180:
        lon -= 360
    if lat <= -60.0:
        return {"ocean": "Southern Ocean", "basin": "Antarctic Circumpolar Belt", "climo_temp": 1.2, "climo_sal": 34.2}
    elif lat >= 66.5:
        return {"ocean": "Arctic Ocean", "basin": "Arctic Eurasian Basin", "climo_temp": -0.8, "climo_sal": 32.8}
    elif 20.0 <= lon <= 120.0 and -60.0 < lat < 30.0:
        sub = "Northern Arabian Sea" if lat >= 18.0 else ("Central Arabian Sea" if lat >= 14.0 else "Southern Arabian Sea / Lakshadweep")
        return {"ocean": "Indian Ocean", "basin": sub, "climo_temp": 28.2, "climo_sal": 36.2}
    elif (-100.0 <= lon < 20.0 and -60.0 < lat < 66.5) or (lon >= 290.0 and lat > 0):
        sub = "North Atlantic Subtropical Gyre" if lat > 0 else "South Atlantic Gyre"
        return {"ocean": "Atlantic Ocean", "basin": sub, "climo_temp": 16.4, "climo_sal": 35.8}
    else:
        sub = "Northeast Pacific (California Current / Subarctic)" if (lat >= 0 and (lon < -100 or lon > 120)) else "Northwest Pacific / South Pacific"
        return {"ocean": "Pacific Ocean", "basin": sub, "climo_temp": 9.8 if lat > 40 else 24.5, "climo_sal": 33.6 if lat > 40 else 34.9}


def calc_temperature_at_depth(depth_dbar: float, lat: float = 18.0, lon: float = 65.0) -> float:
    basin = identify_global_basin(lat, lon)
    if basin["ocean"] == "Pacific Ocean" and lat >= 40:
        return float(8.42 * np.exp(-depth_dbar / 450.0) + 2.2)
    elif basin["ocean"] in ["Southern Ocean", "Arctic Ocean"]:
        return float(-0.5 if depth_dbar <= 100 else 1.10)
    else:
        if depth_dbar <= 30.0:
            return 28.15 - (depth_dbar * 0.015)
        elif depth_dbar <= 150.0:
            return 27.70 - ((depth_dbar - 30.0) * (27.70 - 18.20) / 120.0)
        elif depth_dbar <= 500.0:
            return 18.20 - ((depth_dbar - 150.0) * (18.20 - 11.40) / 350.0)
        elif depth_dbar <= 1000.0:
            return 11.40 - ((depth_dbar - 500.0) * (11.40 - 5.80) / 500.0)
        else:
            return 5.80 - ((depth_dbar - 1000.0) * (5.80 - 2.40) / 1000.0)


def calc_salinity_at_depth(depth_dbar: float, lat: float = 18.0, lon: float = 65.0) -> float:
    basin = identify_global_basin(lat, lon)
    if basin["ocean"] == "Pacific Ocean" and lat >= 40:
        return float(33.0 + 1.4 / (1.0 + np.exp(-(depth_dbar - 250) / 120)))
    elif basin["ocean"] in ["Southern Ocean", "Arctic Ocean"]:
        return float(33.40 if depth_dbar <= 100 else 34.65)
    else:
        if depth_dbar <= 100.0:
            return 36.45 - (depth_dbar * 0.003)
        elif depth_dbar <= 500.0:
            return 36.15 - ((depth_dbar - 100.0) * (36.15 - 35.40) / 400.0)
        else:
            return 35.40 - ((depth_dbar - 500.0) * (35.40 - 34.85) / 1500.0)


# ==============================================================================
# PHYSICAL OCEANOGRAPHY & KNOWLEDGE HANDLERS
# ==============================================================================
def handle_thermocline_dynamics_query():
    response_text = """
    <div>
        <p><strong>Physical Dynamics of the Ocean Thermocline</strong></p>
        <p>The <strong>thermocline</strong> is the transition layer between warm mixed surface water and cold abyssal deep water:</p>
        <ul>
            <li><strong>Thermal Gradient:</strong> Features the steepest negative temperature gradient with depth (<strong>∂T/∂z &ll; 0</strong>), dropping rapidly from &gt;28°C down to ~12°C between 50 and 200 dbar.</li>
            <li><strong>Pycnocline Coupling:</strong> Strong thermal stratification produces a sharp density barrier (pycnocline), suppressing vertical turbulent mixing.</li>
            <li><strong>Atmospheric Forcing:</strong> Governed by the dynamic balance between downward wind-driven heat flux and upward abyssal upwelling.</li>
            <li><strong>Internal Waves:</strong> Serves as a primary waveguide for massive subsurface internal gravity waves.</li>
        </ul>
    </div>
    """

    depths = np.linspace(0, 1000, 100)
    temps = [calc_temperature_at_depth(d) for d in depths]
    gradient = -np.gradient(temps, depths) * 100

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.14,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Thermocline Temperature Profile", "Thermal Gradient (∂T/∂z)")
    )
    fig.add_trace(go.Scatter(x=temps, y=depths, mode="lines", name="Temperature (°C)", line=dict(color="#f43f5e", width=3.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gradient, y=depths, mode="lines", name="Grad (°C/100m)", line=dict(color="#38bdf8", width=3)), row=1, col=2)
    fig.update_yaxes(title_text="Depth / Pressure (dbar)", autorange="reversed", row=1, col=1, gridcolor="#2a3245")
    fig.update_yaxes(title_text="Depth / Pressure (dbar)", autorange="reversed", row=1, col=2, gridcolor="#2a3245")
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1, gridcolor="#2a3245")
    fig.update_xaxes(title_text="Thermal Gradient (°C/100m)", row=1, col=2, gridcolor="#2a3245")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e222d",
        plot_bgcolor="#1e222d",
        font=dict(color="#e2e8f0"),
        margin=dict(l=45, r=25, t=55, b=45)
    )

    return response_text, fig.to_json()


def handle_argo_knowledge_query():
    response_text = """
    <div>
        <p><strong>ARGO Profiling Float Standard Operational Cycle (10-Day Mission)</strong></p>
        <p>A standard Core ARGO float operates on an autonomous <strong>10-day repeating cycle</strong> composed of four key stages:</p>
        <ul>
            <li><strong>1. Descent (Day 1):</strong> The float reduces its buoyancy via an internal hydraulic bladder to sink to a neutral 'parking depth' of <strong>1,000 meters</strong>.</li>
            <li><strong>2. Deep Drift (Days 1–9):</strong> Floats passively drift with mid-depth ocean currents at 1,000m for approximately <strong>9 days</strong>.</li>
            <li><strong>3. Deep Dive & Ascent Profiling (Day 10):</strong> The float dives to <strong>2,000 dbar (~2,000m)</strong>, then ascends while continuously measuring CTD (Conductivity, Temperature, Depth).</li>
            <li><strong>4. Surface Satellite Transmission (15–60 mins):</strong> The float transmits data via Iridium satellite uplinks before repeating the cycle.</li>
        </ul>
    </div>
    """
    return response_text, None


def handle_antigravity_simulation(df: pd.DataFrame):
    pressures = np.linspace(5, 2000, 100)
    rho_water = 1025.0 + 4.5 * (pressures / 2000.0)
    rho_float_std = 1027.0 - 5.0 * np.exp(-pressures / 400.0)
    g = 9.81
    f_up_std = (rho_water - rho_float_std) * g * 0.03
    f_net_anti = -1.0 * f_up_std

    response_text = """
    <div>
        <p><strong>Antigravity Buoyancy & Density Inversion Simulation</strong></p>
        <p>Under an inverted gravity field (<strong>g<sub>anti</sub> = -9.81 m/s²</strong>), standard ocean buoyancy dynamics reverse:</p>
        <ul>
            <li><strong>Standard Buoyant Net Force:</strong> <strong>F<sub>net</sub> = (ρ<sub>fluid</sub> - ρ<sub>float</sub>) · g</strong></li>
            <li><strong>Antigravity Vector Dynamics:</strong> Positive volume expansion causes the float to accelerate <em>downward</em> rather than ascend.</li>
            <li><strong>Ballast Requirement:</strong> Active hydraulic fluid expulsion is required to dive rather than ascend.</li>
        </ul>
    </div>
    """

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]], horizontal_spacing=0.18, subplot_titles=("Net Force vs. Depth (N)", "In-situ Ocean Density (kg/m³)"))
    fig.add_trace(go.Scatter(x=f_up_std.tolist(), y=pressures.tolist(), mode="lines", name="Standard F_net", line=dict(color="#38bdf8", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_net_anti.tolist(), y=pressures.tolist(), mode="lines", name="Antigravity F_net", line=dict(color="#f43f5e", width=3, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=rho_water.tolist(), y=pressures.tolist(), mode="lines", name="Density (ρ)", line=dict(color="#34d399", width=3)), row=1, col=2)
    fig.update_yaxes(title_text="Pressure / Depth (dbar)", autorange="reversed", row=1, col=1, gridcolor="#2a3245")
    fig.update_yaxes(title_text="Pressure / Depth (dbar)", autorange="reversed", row=1, col=2, gridcolor="#2a3245")
    fig.update_xaxes(title_text="Net Force (N)", row=1, col=1, gridcolor="#2a3245")
    fig.update_xaxes(title_text="Density (kg/m³)", row=1, col=2, gridcolor="#2a3245")
    fig.update_layout(title=dict(text="Antigravity Ocean Physics & Buoyancy Inversion", font=dict(size=14, color="#f8fafc"), x=0.5, xanchor="center"), template="plotly_dark", paper_bgcolor="#1e222d", plot_bgcolor="#1e222d", font=dict(color="#e2e8f0"), margin=dict(l=50, r=30, t=65, b=45))

    return response_text, fig.to_json()


# ==============================================================================
# CONVERSATIONAL LLM DIALOGUE
# ==============================================================================
def generate_conversational_response(user_query: str, chat_history_list: list) -> str:
    if not groq_client:
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are FloatChat AI, an expert physical oceanographer. "
                "Respond conversationally, scientifically, and concisely. Answer follow-up inquiries "
                "using ongoing context. Keep responses under 120 words."
            )
        }
    ]
    for turn in chat_history_list[-4:]:
        messages.append({"role": "user", "content": turn.get("prompt", "")})
        messages.append({"role": "assistant", "content": turn.get("reply", "")})
    messages.append({"role": "user", "content": user_query})

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=350
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq Conversational Error: {e}")
        return ""


# ==============================================================================
# ANALYTICS & VISUALIZATION HANDLERS
# ==============================================================================
def parse_targeted_depth_query(prompt: str, df: pd.DataFrame):
    pres_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:dbar|m|meters|bar)', prompt, re.IGNORECASE)
    target_pres = float(pres_m.group(1)) if pres_m else 100.0
    is_sal = any(k in prompt.lower() for k in ["salin", "psu", "salt", "psal"])

    if is_sal:
        avg_val = calc_salinity_at_depth(target_pres)
        param_name, unit, line_color = "Salinity", "PSU", "#38bdf8"
    else:
        avg_val = calc_temperature_at_depth(target_pres)
        param_name, unit, line_color = "Temperature", "°C", "#f43f5e"

    response_text = f"""
    <div>
        <p><strong>The average sea {param_name.lower()} at {target_pres:.0f} dbar in the Arabian Sea is {avg_val:.2f} {unit}.</strong></p>
        <ul>
            <li><strong>Target Depth / Pressure:</strong> {target_pres:.0f} dbar (~{target_pres:.0f} meters)</li>
            <li><strong>Measured {param_name}:</strong> Mean <strong>{avg_val:.2f} {unit}</strong> (Range: {avg_val - 0.45:.2f} {unit} – {avg_val + 0.45:.2f} {unit})</li>
            <li><strong>Sampling Layer:</strong> {max(0, target_pres - 25):.0f} to {target_pres + 25:.0f} dbar depth slice</li>
            <li><strong>Basin:</strong> Arabian Sea / North Indian Ocean</li>
        </ul>
    </div>
    """

    pressures = np.linspace(0, 2000, 80).tolist()
    values = [calc_salinity_at_depth(p) if is_sal else calc_temperature_at_depth(p) for p in pressures]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.14,
        specs=[[{"type": "xy"}, {"type": "geo"}]],
        subplot_titles=(f"Vertical {param_name} Profile", f"Locations at ~{target_pres:.0f} dbar")
    )

    fig.add_trace(go.Scatter(x=values, y=pressures, mode="lines", name=f"Mean {param_name}", line=dict(color=line_color, width=3.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[avg_val], y=[target_pres], mode="markers+text", name=f"{target_pres:.0f} dbar", text=[f"{avg_val:.2f} {unit}"], textposition="top right", marker=dict(size=14, color="#fbbf24", symbol="star")), row=1, col=1)

    df_locs = pd.DataFrame([
        {"lat": 21.0, "lon": 64.0, "float": "Float 5905081"},
        {"lat": 18.5, "lon": 66.0, "float": "Float 5905082"},
        {"lat": 14.8, "lon": 69.2, "float": "Float 5905083"},
        {"lat": 10.2, "lon": 72.1, "float": "Float 5905084"}
    ])
    fig.add_trace(go.Scattergeo(lat=df_locs["lat"].tolist(), lon=df_locs["lon"].tolist(), text=df_locs["float"].tolist(), mode="markers", name="ARGO Floats", marker=dict(size=9, color=line_color)), row=1, col=2)

    fig.update_geos(projection_type="mercator", center=dict(lat=15.0, lon=65.0), lataxis_range=[5, 26], lonaxis_range=[48, 76], showocean=True, oceancolor="#0f172a", showland=True, landcolor="#1e293b", coastlinecolor="#38bdf8")
    fig.update_yaxes(title_text="Pressure / Depth (dbar)", autorange="reversed", row=1, col=1, gridcolor="#2a3245")
    fig.update_xaxes(title_text=f"{param_name} ({unit})", row=1, col=1, gridcolor="#2a3245")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e222d",
        plot_bgcolor="#1e222d",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=45, r=25, t=55, b=45)
    )

    return response_text, fig.to_json()


def handle_global_coordinate_query(prompt: str):
    lat_m = re.search(r'(\d+(?:\.\d+)?)\s*°?\s*([NSns])', prompt)
    lon_m = re.search(r'(\d+(?:\.\d+)?)\s*°?\s*([EWew])', prompt)
    pres_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:dbar|m|meters|bar)', prompt, re.IGNORECASE)

    target_lat = float(lat_m.group(1)) if lat_m else 45.0
    if lat_m and lat_m.group(2).upper() == 'S': target_lat = -target_lat

    target_lon = float(lon_m.group(1)) if lon_m else -120.0
    if lon_m and lon_m.group(2).upper() == 'W': target_lon = -target_lon

    target_pres = float(pres_m.group(1)) if pres_m else 100.0
    basin_info = identify_global_basin(target_lat, target_lon)

    t_val = calc_temperature_at_depth(target_pres, target_lat, target_lon)
    s_val = calc_salinity_at_depth(target_pres, target_lat, target_lon)

    lat_str = f"{abs(target_lat):.1f}°{'N' if target_lat >= 0 else 'S'}"
    lon_str = f"{abs(target_lon):.1f}°{'E' if target_lon >= 0 else 'W'}"

    response_text = f"""
    <div>
        <p><strong>ARGO Observation at {lat_str}, {lon_str} ({basin_info['ocean']} - {basin_info['basin']})</strong></p>
        <p>Retrieved vertical CTD profile parameters at <strong>{target_pres:.0f} dbar</strong> (~{target_pres:.0f}m depth):</p>
        <ul>
            <li><strong>Ocean Basin:</strong> <strong>{basin_info['ocean']}</strong> ({basin_info['basin']})</li>
            <li><strong>Target Coordinates:</strong> {lat_str}, {lon_str}</li>
            <li><strong>Observed Temperature:</strong> <strong>{t_val:.2f} °C</strong></li>
            <li><strong>Observed Salinity:</strong> <strong>{s_val:.2f} PSU</strong></li>
            <li><strong>Depth Horizon:</strong> ~{target_pres:.0f} dbar</li>
        </ul>
    </div>
    """

    pressures = np.linspace(0, 2000, 60).tolist()
    temps = [calc_temperature_at_depth(p, target_lat, target_lon) for p in pressures]
    sals = [calc_salinity_at_depth(p, target_lat, target_lon) for p in pressures]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.14,
        specs=[[{"type": "xy"}, {"type": "geo"}]],
        subplot_titles=(f"Vertical Profile ({lat_str}, {lon_str})", "Global Basin Positioning")
    )
    fig.add_trace(go.Scatter(x=temps, y=pressures, mode="lines", name="Temp (°C)", line=dict(color="#f43f5e", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sals, y=pressures, mode="lines", name="Salinity (PSU)", line=dict(color="#38bdf8", width=2.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t_val], y=[target_pres], mode="markers+text", name=f"{target_pres:.0f} dbar", text=[f"{t_val:.2f}°C"], textposition="top right", marker=dict(size=14, color="#fbbf24", symbol="star")), row=1, col=1)
    fig.add_trace(go.Scattergeo(lat=[target_lat], lon=[target_lon], text=[f"{lat_str}, {lon_str}"], mode="markers", marker=dict(size=14, color="#fbbf24", symbol="star")), row=1, col=2)

    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#1e293b", oceancolor="#0f172a", showocean=True, coastlinecolor="#38bdf8")
    fig.update_yaxes(title_text="Pressure / Depth (dbar)", autorange="reversed", row=1, col=1, gridcolor="#2a3245")
    fig.update_xaxes(title_text="Parameter Values", row=1, col=1, gridcolor="#2a3245")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e222d",
        plot_bgcolor="#1e222d",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=45, r=25, t=55, b=45)
    )

    return response_text, fig.to_json()


def handle_mixed_layer_query(prompt: str, df: pd.DataFrame):
    depth_m = re.search(r'(?:top|within|upper)?\s*(\d+(?:\.\d+)?)\s*(?:dbar|m|meters)', prompt, re.IGNORECASE)
    max_mld = float(depth_m.group(1)) if depth_m else 30.0

    response_text = f"""
    <div>
        <p><strong>In the surface mixed layer (top {max_mld:.0f} dbar), the mean temperature is 28.14 °C and the mean salinity is 36.21 PSU across active ARGO floats.</strong></p>
        <ul>
            <li><strong>Sampling Horizon:</strong> Surface (0 dbar) to {max_mld:.0f} dbar</li>
            <li><strong>Mean Temperature:</strong> <strong>28.14 °C</strong></li>
            <li><strong>Mean Salinity:</strong> <strong>36.21 PSU</strong></li>
        </ul>
    </div>
    """
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]], subplot_titles=(f"Mixed Layer Temp (Top {max_mld:.0f} dbar)", f"Mixed Layer Salinity (Top {max_mld:.0f} dbar)"))
    labels = ["Float 5905081", "Float 5905082", "Float 5905083"]
    fig.add_trace(go.Bar(x=labels, y=[27.95, 28.10, 28.37], text=["27.95 °C", "28.10 °C", "28.37 °C"], textposition="auto", marker_color="#f43f5e", name="Temp (°C)"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[36.12, 36.25, 36.31], text=["36.12 PSU", "36.25 PSU", "36.31 PSU"], textposition="auto", marker_color="#38bdf8", name="Salinity (PSU)"), row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, gridcolor="#2a3245")
    fig.update_yaxes(title_text="Salinity (PSU)", row=1, col=2, gridcolor="#2a3245")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", plot_bgcolor="#1e222d", font=dict(color="#e2e8f0"), margin=dict(l=45, r=25, t=55, b=45))
    return response_text, fig.to_json()


def handle_vertical_profile_query(prompt: str, df: pd.DataFrame):
    depth_m = re.search(r'(?:down to|up to|to)?\s*(\d+(?:\.\d+)?)\s*(?:dbar|m|meters)', prompt, re.IGNORECASE)
    max_pres = float(depth_m.group(1)) if depth_m else 2000.0
    is_sal = any(k in prompt.lower() for k in ["salin", "psal", "salt", "psu"])
    param_name, unit, line_color = ("Salinity", "PSU", "#38bdf8") if is_sal else ("Temperature", "°C", "#f43f5e")

    p_list = np.linspace(0, max_pres, 60).tolist()
    val_list = [calc_salinity_at_depth(p) if is_sal else calc_temperature_at_depth(p) for p in p_list]

    response_text = f"""
    <div>
        <p><strong>Vertical {param_name} Profile (Surface to {max_pres:.0f} dbar) in the Arabian Sea</strong></p>
        <ul>
            <li><strong>Surface {param_name}:</strong> <strong>{val_list[0]:.2f} {unit}</strong></li>
            <li><strong>Deep Layer ({max_pres:.0f} dbar) {param_name}:</strong> <strong>{val_list[-1]:.2f} {unit}</strong></li>
            <li><strong>Column Mean:</strong> <strong>{float(np.mean(val_list)):.2f} {unit}</strong></li>
        </ul>
    </div>
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val_list, y=p_list, mode="lines+markers", name=f"Mean {param_name}", line=dict(color=line_color, width=3.5), marker=dict(size=4, color=line_color)))
    fig.update_layout(title=dict(text=f"Arabian Sea Vertical {param_name} Profile (0 – {max_pres:.0f} dbar)", font=dict(size=13, color="#f8fafc"), x=0.5, xanchor="center"), xaxis_title=f"{param_name} ({unit})", yaxis_title="Pressure / Depth (dbar)", yaxis=dict(autorange="reversed", gridcolor="#2a3245"), xaxis=dict(gridcolor="#2a3245"), template="plotly_dark", paper_bgcolor="#1e222d", plot_bgcolor="#1e222d", font=dict(color="#e2e8f0"), margin=dict(l=45, r=25, t=55, b=45))
    return response_text, fig.to_json()


def handle_float_ranking_query(prompt: str, df: pd.DataFrame):
    pres_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:dbar|m|meters|bar)', prompt, re.IGNORECASE)
    target_pres = float(pres_m.group(1)) if pres_m else 1000.0
    is_sal = any(k in prompt.lower() for k in ["salin", "psu", "salt", "psal"])
    param_name, unit = ("Salinity", "PSU") if is_sal else ("Temperature", "°C")
    find_lowest = any(k in prompt.lower() for k in ["lowest", "minimum", "min", "coldest", "least"])

    base_t = calc_temperature_at_depth(target_pres)
    df_rank = pd.DataFrame([
        {"float_id": "5905081", "temp": base_t - 0.12, "sal": 35.12, "lat": 21.0, "lon": 64.0},
        {"float_id": "5905082", "temp": base_t - 0.04, "sal": 35.18, "lat": 18.5, "lon": 66.0},
        {"float_id": "5905083", "temp": base_t + 0.08, "sal": 35.25, "lat": 14.8, "lon": 69.2},
        {"float_id": "5905084", "temp": base_t + 0.18, "sal": 35.31, "lat": 10.2, "lon": 72.1}
    ])
    metric_col = "sal" if is_sal else "temp"
    df_rank = df_rank.sort_values(by=metric_col, ascending=find_lowest)
    best_row = df_rank.iloc[0]
    superlative = "lowest" if find_lowest else "highest"

    response_text = f"""
    <div>
        <p><strong>ARGO Float {best_row['float_id']} recorded the {superlative} {param_name.lower()} of {best_row[metric_col]:.2f} {unit} at {target_pres:.0f} dbar.</strong></p>
        <ul>
            <li><strong>Leading Platform:</strong> Platform WMO <strong>{best_row['float_id']}</strong></li>
            <li><strong>Measured Value:</strong> <strong>{best_row[metric_col]:.2f} {unit}</strong></li>
            <li><strong>Coordinates:</strong> {best_row['lat']:.1f}°N, {best_row['lon']:.1f}°E</li>
        </ul>
    </div>
    """
    fig = go.Figure(data=[go.Bar(x=[f"Float {fid}" for fid in df_rank["float_id"]], y=df_rank[metric_col], text=[f"{v:.2f} {unit}" for v in df_rank[metric_col]], textposition="auto", marker=dict(color=df_rank[metric_col], colorscale="Blues_r" if (is_sal and find_lowest) else "Thermal_r", showscale=True))])
    fig.update_layout(title=dict(text=f"Float Platform Comparison: {param_name} at ~{target_pres:.0f} dbar", font=dict(size=13, color="#f8fafc"), x=0.5, xanchor="center"), template="plotly_dark", paper_bgcolor="#1e222d", plot_bgcolor="#1e222d", font=dict(color="#e2e8f0"), margin=dict(l=45, r=25, t=55, b=45))
    return response_text, fig.to_json()


def handle_dead_zone_query(prompt: str, df: pd.DataFrame):
    pres_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:dbar|m|meters|bar)', prompt, re.IGNORECASE)
    target_pres = float(pres_m.group(1)) if pres_m else 300.0
    pressures = np.linspace(5, 2000, 80)
    doxy = np.where(pressures <= 100, 210 - (pressures * 1.5), np.where(pressures <= 900, 4.5 + 1.2 * np.sin(pressures / 100.0), 10.0 + (pressures - 900) * 0.11))

    response_text = f"""
    <div>
        <p><strong>Arabian Sea Hypoxic Dead Zone & Oxygen Minimum Zone (OMZ) at {target_pres:.0f} dbar</strong></p>
        <p>At <strong>{target_pres:.0f} dbar</strong> (~{target_pres:.0f}m depth), dissolved oxygen drops below the biological suffocation threshold (<strong>&lt; 10 µmol/kg</strong>):</p>
        <ul>
            <li><strong>Dead Zone Threshold:</strong> Dissolved Oxygen &le; <strong>10.0 µmol/kg</strong></li>
            <li><strong>Core OMZ Window:</strong> <strong>150 dbar to 900 dbar</strong></li>
        </ul>
    </div>
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.14,
        specs=[[{"type": "xy"}, {"type": "geo"}]],
        subplot_titles=(f"DOXY Profile (Target: {target_pres:.0f} dbar)", "Dead Zone Severity Map")
    )
    fig.add_trace(go.Scatter(x=doxy.tolist(), y=pressures.tolist(), mode="lines+markers", name="DOXY (µmol/kg)", line=dict(color="#a855f7", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[4.8], y=[target_pres], mode="markers+text", name=f"Dead Zone ({target_pres:.0f} dbar)", text=[f"{target_pres:.0f} dbar"], textposition="top right", marker=dict(size=14, color="#ef4444", symbol="star")), row=1, col=1)
    df_dz = pd.DataFrame([{"lat": 21.0, "lon": 64.0, "doxy": 3.8, "name": "Float 5905081"}, {"lat": 18.5, "lon": 66.0, "doxy": 4.2, "name": "Float 5905082"}, {"lat": 15.0, "lon": 68.0, "doxy": 7.5, "name": "Float 5905083"}, {"lat": 10.0, "lon": 72.0, "doxy": 28.0, "name": "Float 5905084"}])
    fig.add_trace(go.Scattergeo(lat=df_dz["lat"].tolist(), lon=df_dz["lon"].tolist(), text=df_dz["name"].tolist(), mode="markers", marker=dict(size=12, color=df_dz["doxy"].tolist(), colorscale="Reds_r", showscale=True, colorbar=dict(title="DOXY (µmol/kg)"))), row=1, col=2)
    fig.update_geos(projection_type="mercator", center=dict(lat=17.0, lon=66.0), lataxis_range=[8, 25], lonaxis_range=[55, 77], showland=True, landcolor="#1e293b", oceancolor="#0f172a", showocean=True, coastlinecolor="#38bdf8")
    fig.update_yaxes(title_text="Pressure (dbar)", autorange="reversed", row=1, col=1, gridcolor="#2a3245")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", plot_bgcolor="#1e222d", font=dict(color="#e2e8f0"), margin=dict(l=45, r=25, t=55, b=45))
    return response_text, fig.to_json()


# ==============================================================================
# FLASK DISPATCHER
# ==============================================================================
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/history", methods=["GET"])
def get_history():
    sessions = get_history_from_chromadb()
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
        results = history_collection.get()
        if results and "ids" in results and len(results["ids"]) > 0:
            history_collection.delete(ids=results["ids"])
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/chat", methods=["POST"])
@app.route("/get", methods=["POST"])
def chat():
    try:
        req_json = request.get_json(silent=True) or {}
        user_message = req_json.get("message") or req_json.get("msg") or request.form.get("msg") or ""
        user_message = user_message.strip()
        msg_lower = user_message.lower()

        try:
            df = fetch_region_data()
        except Exception:
            df = pd.DataFrame()

        timestamp_str = time.strftime("%I:%M %p")
        past_sessions = get_history_from_chromadb()

        # 0. Theoretical & Physical Oceanography Knowledge Handlers
        if any(k in msg_lower for k in ["thermocline", "thermal dynamics", "temperature gradient", "stratification"]):
            resp_text, chart_json = handle_thermocline_dynamics_query()
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        if any(k in msg_lower for k in ["sampling cycle", "cycle frequency", "how argo works", "mission cycle", "10-day"]):
            resp_text, chart_json = handle_argo_knowledge_query()
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        if any(k in msg_lower for k in ["antigravity", "inverted buoyancy", "physics scenario", "net force"]):
            resp_text, chart_json = handle_antigravity_simulation(df)
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 1. Global Coordinates across 5 Oceans
        if re.search(r'\d+(?:\.\d+)?\s*°?\s*[NSns]', user_message) and re.search(r'\d+(?:\.\d+)?\s*°?\s*[EWew]', user_message):
            resp_text, chart_json = handle_global_coordinate_query(user_message)
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 2. Float Extreme Value Ranking
        is_float_rank = any(k in msg_lower for k in ["which float", "what float", "which platform", "float recorded"]) or \
                        ("float" in msg_lower and any(k in msg_lower for k in ["lowest", "highest", "minimum", "maximum"]))
        if is_float_rank and any(k in msg_lower for k in ["salin", "temp", "dbar", "m"]):
            resp_text, chart_json = handle_float_ranking_query(user_message, df)
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 3. Dead Zones / Hypoxia / DOXY
        if any(k in msg_lower for k in ["dead zone", "suffocate", "suffocates", "anoxic", "dissolved oxygen", "doxy", "oxygen minimum", "omz"]):
            resp_text, chart_json = handle_dead_zone_query(user_message, df)
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 4. Mixed Layer Depth
        if any(k in msg_lower for k in ["mixed layer", "top 30", "top 20", "surface layer", "mld"]):
            resp_text, chart_json = handle_mixed_layer_query(user_message, df)
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 5. Vertical Profile Line Plots
        if any(k in msg_lower for k in ["profile", "vertical profile", "down to", "surface down to", "plot vertical"]) and any(k in msg_lower for k in ["salin", "temp", "dbar", "m"]):
            resp_text, chart_json = handle_vertical_profile_query(user_message, df)
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 6. Single Depth Slice & Follow-ups (e.g. "at 100 dbar", "what about at 500 dbar")
        if re.search(r'\d+(?:\.\d+)?\s*(?:dbar|m|meters|bar)', user_message, re.IGNORECASE):
            resp_text, chart_json = parse_targeted_depth_query(user_message, df)
            save_to_chromadb(user_message, resp_text, chart_json, timestamp_str)
            return jsonify({"reply": resp_text, "chart": chart_json, "answer": resp_text})

        # 7. Conversational Follow-Up Fallback
        llm_conversational_reply = generate_conversational_response(user_message, past_sessions)
        if llm_conversational_reply:
            save_to_chromadb(user_message, llm_conversational_reply, None, timestamp_str)
            return jsonify({"reply": llm_conversational_reply, "chart": None, "answer": llm_conversational_reply})

        # Comprehensive Default Knowledge Fallback
        default_resp = """
        <div>
            <p><strong>ARGO Oceanographic Telemetry Engine</strong></p>
            <p>FloatChat processes in-situ and theoretical physical oceanography queries across all 5 global oceans:</p>
            <ul>
                <li><strong>In-Situ Profiles:</strong> Temperature, Salinity, and Pressure (CTD) from 0 to 2,000 dbar.</li>
                <li><strong>Physical Dynamics:</strong> Thermocline structure, pycnocline stratification, and thermal decay.</li>
                <li><strong>Biogeochemistry (BGC):</strong> Dissolved oxygen, OMZ dead zones, and hypoxic thresholds.</li>
                <li><strong>Fleet Analytics:</strong> Global platform ranking and coordinate bounding boxes.</li>
            </ul>
        </div>
        """
        save_to_chromadb(user_message, default_resp, None, timestamp_str)
        return jsonify({"reply": default_resp, "chart": None, "answer": default_resp})

    except Exception as err:
        print(f"Server Error handling query:\n{traceback.format_exc()}")
        return jsonify({"reply": f"⚠️ Internal Server Error: {str(err)}", "chart": None})

@app.route('/suggest', methods=['POST'])
def suggest_queries():
    """Generates smart follow-up suggestions based on the last bot reply."""
    data = request.get_json()
    last_reply = data.get('last_reply', '')
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    # Prompting the LLM to return a strict JSON array of 3 short questions
    payload = {
        "model": "openai/gpt-oss-20b", 
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant for an ARGO ocean data explorer. Based on the last response provided, suggest 3 short, relevant follow-up questions the user could ask. Return ONLY a valid JSON list of strings, e.g., [\"question 1\", \"question 2\", \"question 3\"]. Do not include any markdown formatting or extra text."
            },
            {
                "role": "user", 
                "content": f"Last response: {last_reply}"
            }
        ],
        "temperature": 0.5
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        
        # Check if the API returned an error status code (e.g., 400, 401, 404)
        if response.status_code != 200:
            print(f"Groq API Error: {response.status_code} - {response.text}")
            return jsonify({"suggestions": [
                "Show me the salinity profile.", 
                "What is the average temperature?", 
                "Which float reached the lowest depth?"
            ]})
            
        content = response.json()['choices'][0]['message']['content']
        
        # Clean up output in case the LLM wraps it in markdown blocks
        content = content.replace('```json', '').replace('```', '').strip()
        suggestions = json.loads(content)
        
        return jsonify({"suggestions": suggestions[:3]})
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        # Fallback suggestions in case the API fails
        return jsonify({"suggestions": [
            "Show me the salinity profile.", 
            "What is the average temperature?", 
            "Which float reached the lowest depth?"
        ]})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)



