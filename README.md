# 🌊 FloatChat AI
### Autonomous Ocean Intelligence & Telemetry Portal

**FloatChat AI** is an advanced, multilingual hydrographic discovery platform that bridges the gap between massive real-time ARGO profiling float databases and human natural language. Designed for researchers, students, and oceanographers, it enables seamless multi-parameter querying via text, speech, or multimodal document uploads.

---

## ✨ Key Features

* **💬 Natural Language Telemetry Querying:** Ask complex oceanographic questions in plain English, Hindi, or Marathi and receive instant, precise insights.
* **🌐 Interactive 3D Globe & Spatial Arrays:** Explore real-time global ARGO float networks, orthographic projections, and deep-sea landmarks (e.g., Mariana Trench, OMZ dead zones).
* **📄 Multimodal OCR & Document Parsing:** Upload research scans, PDFs, or data tables; the system automatically extracts clean text and routes it to the correct intent handler.
* **📊 Dual-Scene 3D Visualizations:** Render multi-axis subsurface profiles (Temperature, Salinity, Dissolved Oxygen, Thermocline Gradients) down to 11,000 meters.
* **🛡️ QC-Aware Confidence Scoring:** Dynamic reliability and confidence metrics ($65\% - 99.4\%$) calculated from spatial proximity and depth-tier resolution.
* **📤 Export & Collaboration Suite:** One-click downloads of 3D charts in **PNG, JPEG, or PDF** formats, plus instant generation of shareable conversation links.

---

## 🏗️ System Architecture & Workflow

1. **Input Layer:** Accepts text prompts, multilingual voice input (Web Speech API), or uploaded files/images.
2. **Intent Parsing & OCR Engine:** Powered by Groq/Llama-3 models and local Tesseract/Vision fallback pipelines to clean and translate user syntax.
3. **Hydrographic Climatology Engine:** Computes depth-dependent and latitude-dependent ocean profiles mimicking real-world CTD casts.
4. **Visualization Layer:** Powered by Plotly.js and Flask to generate interactive 3D spatial grids and subsurface water columns.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Flask, Plotly, Pandas, NumPy, ChromaDB, Groq

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Float-Chat.git](https://github.com/your-username/Float-Chat.git)
   cd Float-Chat