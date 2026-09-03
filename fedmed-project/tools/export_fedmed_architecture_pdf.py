from pathlib import Path

from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SVG = ROOT / "public" / "fedmed-smart-healthcare-architecture.svg"
OUTPUT_DIR = Path(r"C:\Users\Pratyush Dash\OneDrive\Documents\FEDMED_ARCH")
OUTPUT_SVG = OUTPUT_DIR / "fedmed_smart_healthcare_architecture_ieee.svg"
OUTPUT_PDF = OUTPUT_DIR / "fedmed_smart_healthcare_architecture_ieee.pdf"


HOSPITALS = [
    {
        "y": 196,
        "name": "Hospital A",
        "subtitle": "IoT wearables, EHR systems, and diabetes-oriented local cohort",
        "iot_sub": "Wearables, smart sensors, ECG",
        "ehr_sub": "Clinical records and histories",
        "data_sub": "Diabetes, cardiovascular, cholesterol",
    },
    {
        "y": 435,
        "name": "Hospital B",
        "subtitle": "Distributed hospital node for cardiovascular monitoring and EHR fusion",
        "iot_sub": "Cardiac sensors and smart beds",
        "ehr_sub": "Medication, vitals, longitudinal data",
        "data_sub": "Cardiovascular and lipid cohorts",
    },
    {
        "y": 674,
        "name": "Hospital C",
        "subtitle": "Smart cholesterol surveillance with multimodal sensing and local analytics",
        "iot_sub": "Wearables and lab-linked sensors",
        "ehr_sub": "Lab values, prescriptions, outcomes",
        "data_sub": "Cholesterol, diabetes, CVD markers",
    },
]


EDGE_NODES = [
    (338, "EA1", "Edge cloud collects hospital updates"),
    (546, "EA2", "Secure edge pre-aggregation"),
    (754, "EA3", "Regional edge fusion node"),
]


OUTPUTS = [
    (456, "Diabetes Risk Prediction", "Early detection and intervention support"),
    (560, "Cardiovascular Risk Analysis", "Monitoring longitudinal heart-health trends"),
    (664, "Cholesterol Risk Assessment", "Personalized lipid-risk interpretation"),
]


SOLID_FILLS = {
    "url(#bgGrad)": "#f4f8fb",
    "url(#blueGrad)": "#2f6f93",
    "url(#tealGrad)": "#178a8f",
    "url(#shieldGrad)": "#245d86",
    "url(#roseGrad)": "#c96878",
}


SVG_HEAD = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 980" role="img" aria-labelledby="title desc">
  <title id="title">Federated Learning with Differential Privacy in Smart Healthcare</title>
  <desc id="desc">IEEE-style system architecture diagram showing hospitals, edge aggregators, global aggregation, security protections, and healthcare analytics outputs.</desc>
  <defs>
    <linearGradient id="bgGrad" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" stop-color="#f2f8fb"/>
      <stop offset="55%%" stop-color="#f8fbfd"/>
      <stop offset="100%%" stop-color="#e8f1f6"/>
    </linearGradient>
    <linearGradient id="blueGrad" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" stop-color="#4e8fc0"/>
      <stop offset="100%%" stop-color="#2f6f93"/>
    </linearGradient>
    <linearGradient id="tealGrad" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" stop-color="#34a3a7"/>
      <stop offset="100%%" stop-color="#178a8f"/>
    </linearGradient>
    <linearGradient id="shieldGrad" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" stop-color="#3aa7af"/>
      <stop offset="100%%" stop-color="#245d86"/>
    </linearGradient>
    <linearGradient id="roseGrad" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" stop-color="#e5939d"/>
      <stop offset="100%%" stop-color="#c96878"/>
    </linearGradient>
    <marker id="arrowBlue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M0,0 L12,6 L0,12 z" fill="#2f6f93"/>
    </marker>
    <marker id="arrowTeal" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M0,0 L12,6 L0,12 z" fill="#178a8f"/>
    </marker>
    <marker id="arrowGray" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M0,0 L12,6 L0,12 z" fill="#7c95a6"/>
    </marker>
    <style>
      .title { font: 700 30px 'Segoe UI', Arial, sans-serif; fill: #12344e; }
      .subtitle { font: 500 15px 'Segoe UI', Arial, sans-serif; fill: #5a7586; }
      .layerTitle { font: 700 15px 'Segoe UI', Arial, sans-serif; fill: #133b55; letter-spacing: 1.1px; }
      .smallCaps { font: 700 11px 'Segoe UI', Arial, sans-serif; fill: #416071; letter-spacing: 1.7px; }
      .cardTitle { font: 700 18px 'Segoe UI', Arial, sans-serif; fill: #12344e; }
      .cardText { font: 500 13px 'Segoe UI', Arial, sans-serif; fill: #4f6d80; }
      .miniText { font: 600 11px 'Segoe UI', Arial, sans-serif; fill: #466373; }
      .labelText { font: 600 12px 'Segoe UI', Arial, sans-serif; fill: #23485d; }
      .valueText { font: 700 13px 'Segoe UI', Arial, sans-serif; fill: #10324a; }
      .outline { fill: none; stroke: #8fa7b6; stroke-width: 2.4; stroke-dasharray: 8 7; }
      .softPanel { fill: rgba(255,255,255,0.82); stroke: rgba(72,109,129,0.18); stroke-width: 1.6; }
      .chip { fill: #eef5f8; stroke: rgba(55,92,111,0.14); stroke-width: 1; }
    </style>
  </defs>
  <rect x="0" y="0" width="1600" height="980" fill="url(#bgGrad)"/>
  <circle cx="120" cy="120" r="90" fill="#dbeef2" opacity="0.55"/>
  <circle cx="1480" cy="145" r="110" fill="#d6e7ef" opacity="0.45"/>
  <circle cx="1460" cy="890" r="140" fill="#dfeff0" opacity="0.5"/>
  <circle cx="210" cy="910" r="120" fill="#e2eef5" opacity="0.45"/>
  <text x="800" y="52" text-anchor="middle" class="title">Federated Learning with Differential Privacy in Smart Healthcare</text>
  <text x="800" y="78" text-anchor="middle" class="subtitle">Privacy-Preserving Smart Healthcare using Federated Learning and Differential Privacy</text>
  <rect x="40" y="145" width="560" height="790" rx="30" class="outline"/>
  <rect x="625" y="250" width="370" height="685" rx="30" class="outline"/>
  <rect x="540" y="96" width="720" height="124" rx="28" class="outline"/>
  <rect x="1020" y="250" width="230" height="685" rx="30" class="outline"/>
  <rect x="1280" y="250" width="280" height="685" rx="30" class="outline"/>
  <g transform="translate(68 124)">
    <rect width="190" height="34" rx="17" fill="#ffffff" stroke="#d3e0e8"/>
    <text x="95" y="22" text-anchor="middle" class="layerTitle">CLIENT / IoT HEALTHCARE LAYER</text>
  </g>
  <g transform="translate(664 230)">
    <rect width="205" height="34" rx="17" fill="#ffffff" stroke="#d3e0e8"/>
    <text x="102.5" y="22" text-anchor="middle" class="layerTitle">EDGE AGGREGATION LAYER</text>
  </g>
  <g transform="translate(792 76)">
    <rect width="215" height="34" rx="17" fill="#ffffff" stroke="#d3e0e8"/>
    <text x="107.5" y="22" text-anchor="middle" class="layerTitle">GLOBAL SERVER</text>
  </g>
  <g transform="translate(1037 230)">
    <rect width="197" height="34" rx="17" fill="#ffffff" stroke="#d3e0e8"/>
    <text x="98.5" y="22" text-anchor="middle" class="layerTitle">SECURITY / PRIVACY</text>
  </g>
  <g transform="translate(1324 230)">
    <rect width="190" height="34" rx="17" fill="#ffffff" stroke="#d3e0e8"/>
    <text x="95" y="22" text-anchor="middle" class="layerTitle">APPLICATION OUTPUTS</text>
  </g>
"""


SVG_MID = """
  <g transform="translate(722 132)">
    <rect width="355" height="62" rx="22" fill="#ffffff" stroke="#cddce4"/>
    <circle cx="43" cy="31" r="21" fill="url(#blueGrad)"/>
    <rect x="31" y="19" width="24" height="10" rx="3" fill="#ffffff"/>
    <rect x="31" y="33" width="24" height="10" rx="3" fill="#ffffff"/>
    <circle cx="61" cy="31" r="7" fill="url(#tealGrad)"/>
    <text x="82" y="28" class="cardTitle">Global Aggregator (GA)</text>
    <text x="82" y="47" class="cardText">Federated Averaging (FedAvg Algorithm) combines all edge updates into a global model</text>
  </g>
  <g transform="translate(1100 117)">
    <rect width="120" height="86" rx="24" fill="#f2f8fb" stroke="#c7d8e1"/>
    <circle cx="60" cy="34" r="20" fill="url(#tealGrad)"/>
    <circle cx="60" cy="34" r="6" fill="#ffffff"/>
    <circle cx="43" cy="48" r="4" fill="#2e6d93"/>
    <circle cx="77" cy="48" r="4" fill="#2e6d93"/>
    <path d="M60 40 L43 48 M60 40 L77 48" stroke="#2e6d93" stroke-width="2"/>
    <text x="60" y="68" text-anchor="middle" class="smallCaps">GLOBAL MODEL</text>
  </g>
  <path d="M571 299 C654 299, 671 299, 755 341" fill="none" stroke="#2f6f93" stroke-width="3" marker-end="url(#arrowBlue)"/>
  <path d="M571 539 C653 539, 666 539, 755 549" fill="none" stroke="#2f6f93" stroke-width="3" marker-end="url(#arrowBlue)"/>
  <path d="M571 779 C651 779, 671 779, 755 757" fill="none" stroke="#2f6f93" stroke-width="3" marker-end="url(#arrowBlue)"/>
  <g transform="translate(640 286)">
    <rect width="275" height="48" rx="16" fill="#ffffff" stroke="#cfdde6"/>
    <circle cx="26" cy="24" r="12" fill="url(#blueGrad)"/>
    <rect x="21" y="18" width="10" height="12" rx="2" fill="#ffffff"/>
    <path d="M26 17 v-3 M34 24 h3 M15 24 h3" stroke="#ffffff" stroke-width="1.7" stroke-linecap="round"/>
    <text x="50" y="20" class="valueText">Encrypted Model Updates</text>
    <text x="50" y="35" class="miniText">(No Raw Data Sharing)</text>
  </g>
  <path d="M800 338 C800 283, 860 252, 900 194" fill="none" stroke="#2f6f93" stroke-width="3" marker-end="url(#arrowBlue)"/>
  <path d="M800 546 C800 395, 900 310, 910 194" fill="none" stroke="#2f6f93" stroke-width="3" marker-end="url(#arrowBlue)"/>
  <path d="M800 754 C800 515, 955 350, 928 194" fill="none" stroke="#2f6f93" stroke-width="3" marker-end="url(#arrowBlue)"/>
  <path d="M827 193 C700 170, 600 165, 566 210 C529 260, 540 262, 540 283" fill="none" stroke="#178a8f" stroke-width="3" stroke-dasharray="10 7" marker-end="url(#arrowTeal)"/>
  <path d="M838 193 C725 205, 595 355, 548 500" fill="none" stroke="#178a8f" stroke-width="3" stroke-dasharray="10 7" marker-end="url(#arrowTeal)"/>
  <path d="M848 193 C728 240, 596 575, 548 740" fill="none" stroke="#178a8f" stroke-width="3" stroke-dasharray="10 7" marker-end="url(#arrowTeal)"/>
  <g transform="translate(518 118)">
    <rect width="240" height="52" rx="16" fill="#ffffff" stroke="#cfdde6"/>
    <circle cx="24" cy="26" r="11" fill="url(#tealGrad)"/>
    <path d="M18 26 h12" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
    <path d="M24 20 v12" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
    <text x="48" y="22" class="valueText">Updated Global Model Broadcast</text>
    <text x="48" y="38" class="miniText">Bidirectional FL communication loop</text>
  </g>
  <g transform="translate(1038 286)">
    <rect width="194" height="120" rx="26" fill="#fff4f5" stroke="#e7c3c8"/>
    <circle cx="45" cy="42" r="22" fill="url(#roseGrad)"/>
    <path d="M35 53 q10-30 20 0" fill="none" stroke="#ffffff" stroke-width="3" stroke-linecap="round"/>
    <circle cx="45" cy="33" r="8" fill="#ffffff"/>
    <text x="80" y="36" class="cardTitle">Adversary</text>
    <text x="80" y="56" class="cardText">Threat model includes leakage and gradient attacks</text>
    <path d="M40 83 C70 77, 116 77, 147 83" fill="none" stroke="#d26b7a" stroke-width="2.8" stroke-dasharray="7 6" marker-end="url(#arrowGray)"/>
    <text x="42" y="101" class="miniText">Blocked by privacy-preserving defenses</text>
  </g>
  <g transform="translate(1063 438)">
    <path d="M66 0 L116 18 L108 70 C102 103 84 126 66 136 C48 126 30 103 24 70 L16 18 Z" fill="url(#shieldGrad)"/>
    <path d="M50 68 l12 12 l24-26" fill="none" stroke="#ffffff" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
  <g transform="translate(1043 604)">
    <rect width="184" height="50" rx="16" fill="#eef7f7" stroke="#c0dcdb"/>
    <circle cx="24" cy="25" r="10" fill="url(#tealGrad)"/>
    <path d="M19 25 h10" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
    <path d="M24 20 v10" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
    <text x="44" y="21" class="valueText">Differential Privacy</text>
    <text x="44" y="37" class="miniText">Gaussian Noise Injection</text>
  </g>
  <g transform="translate(1043 666)">
    <rect width="184" height="50" rx="16" fill="#eef4f9" stroke="#c8d8e5"/>
    <circle cx="24" cy="25" r="10" fill="url(#blueGrad)"/>
    <path d="M19 25 h10" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
    <path d="M24 20 v10" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
    <text x="44" y="21" class="valueText">Gradient Clipping</text>
    <text x="44" y="37" class="miniText">Bounds sensitivity before sharing</text>
  </g>
  <g transform="translate(1043 728)">
    <rect width="184" height="50" rx="16" fill="#edf7f5" stroke="#bed8cf"/>
    <circle cx="24" cy="25" r="10" fill="#39a889"/>
    <rect x="19" y="20" width="10" height="10" rx="2.5" fill="#ffffff"/>
    <text x="44" y="21" class="valueText">Secure Aggregation</text>
    <text x="44" y="37" class="miniText">Masks updates from raw inspection</text>
  </g>
  <g transform="translate(1038 810)">
    <rect width="194" height="92" rx="22" fill="#ffffff" stroke="#d3e0e8"/>
    <text x="97" y="34" text-anchor="middle" class="valueText">Protection Against Data Leakage</text>
    <text x="97" y="55" text-anchor="middle" class="valueText">and Gradient Attacks</text>
    <text x="97" y="77" text-anchor="middle" class="miniText">Privacy guarantees applied before any model update leaves a hospital</text>
  </g>
  <g transform="translate(1305 292)">
    <rect width="230" height="128" rx="28" fill="#ffffff" stroke="#d1dee7"/>
    <circle cx="46" cy="44" r="23" fill="url(#blueGrad)"/>
    <circle cx="46" cy="34" r="8" fill="#ffffff"/>
    <path d="M31 60 q15-18 30 0" fill="none" stroke="#ffffff" stroke-width="3.2" stroke-linecap="round"/>
    <rect x="76" y="23" width="126" height="32" rx="12" fill="#eef5f8"/>
    <text x="139" y="44" text-anchor="middle" class="cardTitle">Doctor / Analyst</text>
    <text x="76" y="72" class="cardText">Receives privacy-safe decision support</text>
    <text x="76" y="92" class="cardText">from the final global healthcare model</text>
  </g>
  <g transform="translate(1312 782)">
    <rect width="216" height="94" rx="24" fill="#ffffff" stroke="#d1dee7"/>
    <text x="108" y="32" text-anchor="middle" class="valueText">Clinical Decision Support</text>
    <text x="108" y="54" text-anchor="middle" class="miniText">No raw patient records leave any hospital</text>
    <text x="108" y="74" text-anchor="middle" class="miniText">Only privacy-preserved model knowledge is shared</text>
  </g>
  <text x="69" y="944" class="smallCaps">SMART HEALTHCARE  •  FEDERATED LEARNING  •  DIFFERENTIAL PRIVACY  •  SECURE AGGREGATION</text>
"""


SVG_TAIL = "</svg>\n"


def hospital_iot_icon(index: int) -> str:
    if index == 0:
        return (
            '<rect x="18" y="15" width="16" height="25" rx="8" fill="url(#tealGrad)"/>'
            '<rect x="22" y="19" width="8" height="12" rx="4" fill="#ffffff"/>'
        )
    if index == 1:
        return (
            '<circle cx="26" cy="28" r="10" fill="url(#tealGrad)"/>'
            '<path d="M19 28 h4 l3-7 5 14 4-7 h4" fill="none" stroke="#ffffff" '
            'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return (
        '<rect x="17" y="20" width="18" height="18" rx="4" fill="url(#tealGrad)"/>'
        '<path d="M26 17 v-4 M38 27 h4 M14 27 h4 M33 20 l3-3 M19 34 l-3 3 M33 34 l3 3 M19 20 l-3-3" '
        'stroke="#178a8f" stroke-width="2" stroke-linecap="round"/>'
    )


def hospital_card(index: int, y: int, name: str, subtitle: str, iot_sub: str, ehr_sub: str, data_sub: str) -> str:
    return f"""
  <g transform="translate(70 {y})">
    <rect width="500" height="205" rx="28" class="softPanel"/>
    <rect x="0" y="0" width="500" height="50" rx="28" fill="#eaf4f8"/>
    <circle cx="40" cy="26" r="19" fill="url(#blueGrad)"/>
    <rect x="29" y="18" width="22" height="16" rx="2.5" fill="#ffffff"/>
    <rect x="36" y="10" width="8" height="28" rx="1.5" fill="#ffffff"/>
    <rect x="32" y="23" width="16" height="4" fill="url(#blueGrad)"/>
    <text x="70" y="25" class="cardTitle">{name}</text>
    <text x="70" y="43" class="cardText">{subtitle}</text>
    <g transform="translate(25 72)">
      <rect width="120" height="56" rx="18" class="chip"/>
      <circle cx="26" cy="28" r="14" fill="#dbeef2"/>
      {hospital_iot_icon(index)}
      <text x="51" y="25" class="valueText">IoT Devices</text>
      <text x="51" y="42" class="miniText">{iot_sub}</text>
    </g>
    <g transform="translate(157 72)">
      <rect width="120" height="56" rx="18" class="chip"/>
      <circle cx="26" cy="28" r="14" fill="#dbeef2"/>
      <path d="M20 15 h13 l6 6 v20 h-19 z" fill="url(#blueGrad)"/>
      <polyline points="27,24 32,24 32,19" fill="none" stroke="#ffffff" stroke-width="2"/>
      <path d="M23 31 h12" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
      <text x="51" y="25" class="valueText">EHR Systems</text>
      <text x="51" y="42" class="miniText">{ehr_sub}</text>
    </g>
    <g transform="translate(289 72)">
      <rect width="120" height="56" rx="18" class="chip"/>
      <circle cx="26" cy="28" r="14" fill="#dbeef2"/>
      <ellipse cx="26" cy="21" rx="12" ry="5" fill="url(#tealGrad)"/>
      <path d="M14 21 v14 c0 3 5 6 12 6 s12-3 12-6 V21" fill="url(#tealGrad)"/>
      <ellipse cx="26" cy="35" rx="12" ry="5" fill="#43bab3"/>
      <text x="51" y="25" class="valueText">Local Datasets</text>
      <text x="51" y="42" class="miniText">{data_sub}</text>
    </g>
    <g transform="translate(391 68)">
      <rect width="84" height="74" rx="22" fill="#f0f7f8" stroke="#bed3de"/>
      <circle cx="42" cy="28" r="16" fill="url(#tealGrad)"/>
      <circle cx="42" cy="28" r="4.5" fill="#ffffff"/>
      <circle cx="26" cy="42" r="3.6" fill="#2e6d93"/>
      <circle cx="58" cy="42" r="3.6" fill="#2e6d93"/>
      <path d="M42 32 L26 42 M42 32 L58 42" stroke="#2e6d93" stroke-width="2"/>
      <text x="42" y="61" text-anchor="middle" class="miniText">LOCAL MODEL</text>
    </g>
    <path d="M109 156 H392" fill="none" stroke="#7c95a6" stroke-width="2.4" stroke-dasharray="8 6" marker-end="url(#arrowGray)"/>
    <rect x="24" y="164" width="448" height="28" rx="14" fill="#e9f6f6" stroke="#bddedd"/>
    <text x="248" y="183" text-anchor="middle" class="labelText">Local Training + Gradient Clipping + Gaussian Noise (Differential Privacy)</text>
  </g>
"""


def edge_node(y: int, label: str, subtitle: str) -> str:
    return f"""
  <g transform="translate(734 {y})">
    <ellipse cx="66" cy="35" rx="42" ry="23" fill="#dfeff3"/>
    <circle cx="43" cy="39" r="18" fill="#dfeff3"/>
    <circle cx="68" cy="26" r="20" fill="url(#blueGrad)"/>
    <circle cx="91" cy="41" r="16" fill="#dfeff3"/>
    <rect x="44" y="51" width="44" height="32" rx="10" fill="#ffffff" stroke="#c8d8e0"/>
    <rect x="52" y="58" width="28" height="6" rx="3" fill="url(#tealGrad)"/>
    <rect x="52" y="68" width="20" height="6" rx="3" fill="#8fb8ca"/>
    <text x="66" y="103" text-anchor="middle" class="cardTitle">{label}</text>
    <text x="66" y="122" text-anchor="middle" class="cardText">{subtitle}</text>
  </g>
"""


def output_card(y: int, title: str, subtitle: str, variant: int) -> str:
    colors = ["#eef6fa", "#eef8f8", "#f2f8fb"]
    strokes = ["#ccdde8", "#c6dfdf", "#cfdee8"]
    icons = [
        '<path d="M25 29 h6 l4-8 6 14 5-8 h7" fill="none" stroke="#ffffff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>',
        '<path d="M24 29 h20" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/><path d="M34 19 v20" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>',
        '<ellipse cx="34" cy="27" rx="8" ry="4" fill="#ffffff"/><path d="M26 27 v10 c0 2 3 4 8 4 s8-2 8-4 v-10" fill="#ffffff"/>',
    ]
    fills = ["url(#tealGrad)", "url(#blueGrad)", "#39a889"]
    return f"""
  <g transform="translate(1312 {y})">
    <rect width="216" height="82" rx="22" fill="{colors[variant]}" stroke="{strokes[variant]}"/>
    <circle cx="34" cy="29" r="15" fill="{fills[variant]}"/>
    {icons[variant]}
    <text x="62" y="29" class="cardTitle">{title}</text>
    <text x="62" y="50" class="cardText">{subtitle}</text>
  </g>
"""


def build_svg() -> str:
    parts = [SVG_HEAD]
    for idx, item in enumerate(HOSPITALS):
        parts.append(
            hospital_card(
                idx,
                item["y"],
                item["name"],
                item["subtitle"],
                item["iot_sub"],
                item["ehr_sub"],
                item["data_sub"],
            )
        )
    parts.append(SVG_MID)
    for y, label, subtitle in EDGE_NODES:
        parts.append(edge_node(y, label, subtitle))
    for variant, (y, title, subtitle) in enumerate(OUTPUTS):
        parts.append(output_card(y, title, subtitle, variant))
    parts.append(SVG_TAIL)
    return "".join(parts)


def main() -> None:
    svg_text = build_svg()
    for source, replacement in SOLID_FILLS.items():
        svg_text = svg_text.replace(source, replacement)
    PUBLIC_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_SVG.write_text(svg_text, encoding="utf-8")
    OUTPUT_SVG.write_text(svg_text, encoding="utf-8")
    drawing = svg2rlg(str(OUTPUT_SVG))
    if drawing is None:
        raise RuntimeError("Unable to parse generated SVG into a PDF drawing.")
    renderPDF.drawToFile(drawing, str(OUTPUT_PDF))
    print(PUBLIC_SVG)
    print(OUTPUT_SVG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
