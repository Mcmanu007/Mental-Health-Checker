import os
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScan · Mental State Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── API URL (never shown to user) ─────────────────────────────────────────────
def get_api_url() -> str:
    try:
        return st.secrets["API_URL"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_URL", "http://127.0.0.1:8000")

API_URL = get_api_url()

# ── Minimal page-level CSS (only layout, textarea, button) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Merriweather:wght@700&display=swap');

html, body { margin:0; padding:0; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(150deg,#D6E8F7 0%,#E8F5EF 50%,#D6E8F7 100%) !important;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 780px !important; padding: 1.5rem 1.5rem 4rem !important; }

[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    border: 1.5px solid #B8D4E8 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    color: #1C2B3A !important;
    line-height: 1.65 !important;
    padding: 1rem 1.1rem !important;
    box-shadow: 0 2px 8px rgba(30,90,140,0.07) !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #2980B9 !important;
    box-shadow: 0 0 0 3px rgba(41,128,185,0.13) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }

[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg,#1A5276,#2980B9) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    margin-top: 0.4rem !important;
    box-shadow: 0 4px 14px rgba(41,128,185,0.35) !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── State config ──────────────────────────────────────────────────────────────
STATE_CONFIG = {
    "Anxiety": {
        "icon": "⚠️",
        "accent": "#C9A227",
        "light": "#FEF9E7",
        "border": "#F7CA40",
        "tag_bg": "#FEF5CC",
        "tag_color": "#7D6608",
        "guidance": (
            "Possible indicators of anxiety have been detected in this text. "
            "Consider speaking with a licensed therapist or counsellor. "
            "Grounding techniques, mindfulness, and regular physical activity "
            "are evidence-based strategies. If symptoms persist, a psychiatrist "
            "can evaluate medication-assisted options."
        ),
    },
    "Depression": {
        "icon": "📋",
        "accent": "#2471A3",
        "light": "#EAF4FB",
        "border": "#AED6F1",
        "tag_bg": "#D6EAF8",
        "tag_color": "#1A5276",
        "guidance": (
            "Language patterns associated with depression have been identified. "
            "Reaching out to a mental health professional is strongly encouraged. "
            "Cognitive-behavioural therapy (CBT) and interpersonal therapy have "
            "strong clinical evidence. Maintaining routine, social connection, and "
            "sleep hygiene supports recovery alongside professional care."
        ),
    },
    "Normal": {
        "icon": "✅",
        "accent": "#1E8449",
        "light": "#EAFAF1",
        "border": "#A9DFBF",
        "tag_bg": "#D5F5E3",
        "tag_color": "#145A32",
        "guidance": (
            "No significant indicators of psychological distress were detected. "
            "Continue prioritising mental wellness through regular social connection, "
            "physical activity, adequate sleep, and stress management. "
            "Routine mental health check-ins remain beneficial for everyone."
        ),
    },
    "Suicidal": {
        "icon": "🚨",
        "accent": "#C0392B",
        "light": "#FDEDEC",
        "border": "#F1948A",
        "tag_bg": "#FADBD8",
        "tag_color": "#78281F",
        "guidance": (
            "CRITICAL — Urgent professional support is strongly recommended. "
            "If you or someone you know is in immediate danger, contact emergency "
            "services or a crisis helpline right away.\n\n"
            "🇬🇭 Ghana: CHAG Mental Health Line — 0800 111 222\n"
            "🌍 International: findahelpline.com\n"
            "📞 Crisis Text Line: Text HOME to 741741"
        ),
    },
}

LABELS = ["Anxiety", "Depression", "Normal", "Suicidal"]
LABEL_COLORS = {
    "Anxiety":    "#C9A227",
    "Depression": "#2471A3",
    "Normal":     "#1E8449",
    "Suicidal":   "#C0392B",
}

# ── API call ──────────────────────────────────────────────────────────────────
def call_api(text: str) -> dict:
    resp = requests.post(
        f"{API_URL.rstrip('/')}/predict_mental_state",
        json={"text_prepocess": text},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

# ── Build result HTML (rendered via components.html — never via st.markdown) ──
def build_result_html(data: dict) -> str:
    prediction = data.get("prediction", "Unknown")
    confidence = data.get("confidence_score", 0.0)
    probs      = data.get("probabilities", [])
    cfg        = STATE_CONFIG.get(prediction, STATE_CONFIG["Normal"])
    conf_pct   = confidence * 100

    # probability cards
    prob_cards = ""
    for i, label in enumerate(LABELS):
        pct = probs[i] * 100 if i < len(probs) else 0.0
        color = LABEL_COLORS[label]
        prob_cards += f"""
        <div style="background:#F4F8FB;border-radius:10px;padding:12px 14px;
                    border:1px solid #DDE8EF;">
            <div style="font-size:12px;font-weight:600;color:#34495E;margin-bottom:4px;">
                {label}
            </div>
            <div style="font-size:22px;font-weight:700;color:{color};margin-bottom:6px;">
                {pct:.1f}%
            </div>
            <div style="background:#DDE8EF;border-radius:99px;height:5px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:100%;background:{color};
                            border-radius:99px;"></div>
            </div>
        </div>"""

    guidance_text = cfg['guidance'].replace('\n', '<br>')

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', sans-serif; background: transparent; padding: 4px; }}
</style>
</head>
<body>
<div style="background:#FFFFFF;border:1.5px solid {cfg['border']};border-left:5px solid {cfg['accent']};
            border-radius:14px;padding:24px 24px 20px;box-shadow:0 4px 20px rgba(30,90,140,0.09);">

  <!-- Header -->
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;
              padding-bottom:18px;border-bottom:1px solid #E8EEF2;">
    <div style="width:54px;height:54px;border-radius:12px;background:{cfg['light']};
                display:flex;align-items:center;justify-content:center;font-size:26px;flex-shrink:0;">
      {cfg['icon']}
    </div>
    <div>
      <div style="font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
                  color:#5D7A8A;margin-bottom:3px;">Detected Mental State</div>
      <div style="font-family:'Merriweather',serif;font-size:28px;font-weight:700;
                  color:{cfg['accent']};line-height:1.1;">{prediction}</div>
    </div>
    <div style="margin-left:auto;flex-shrink:0;">
      <span style="background:{cfg['tag_bg']};color:{cfg['tag_color']};border-radius:20px;
                   padding:5px 14px;font-size:11px;font-weight:600;">AI Assessment</span>
    </div>
  </div>

  <!-- Confidence -->
  <div style="margin-bottom:20px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
                   color:#5D7A8A;">Confidence Score</span>
      <span style="font-size:16px;font-weight:700;color:{cfg['accent']};">{conf_pct:.1f}%</span>
    </div>
    <div style="background:#E8EEF2;border-radius:99px;height:10px;overflow:hidden;">
      <div style="width:{conf_pct:.1f}%;height:100%;background:{cfg['accent']};
                  border-radius:99px;"></div>
    </div>
  </div>

  <!-- Probability breakdown -->
  <div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
              color:#5D7A8A;margin-bottom:12px;">Probability Breakdown</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px;">
    {prob_cards}
  </div>

  <!-- Clinical guidance -->
  <div style="background:{cfg['light']};border:1px solid {cfg['border']};
              border-radius:10px;padding:14px 16px;">
    <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                color:{cfg['accent']};margin-bottom:8px;">⚕ Clinical Guidance</div>
    <div style="font-size:14px;color:#1C2B3A;line-height:1.7;">
      {guidance_text}
    </div>
  </div>

</div>
</body>
</html>"""

# ── HEADER (via components.html) ──────────────────────────────────────────────
components.html("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@700&display=swap" rel="stylesheet">
<style>* { box-sizing:border-box; margin:0; padding:0; } body { font-family:'Inter',sans-serif; background:transparent; }</style>
</head>
<body>
<div style="background:linear-gradient(135deg,#1A5276 0%,#2471A3 55%,#1A8A6A 100%);
            border-radius:16px;padding:28px 28px 24px;margin-bottom:4px;
            box-shadow:0 8px 32px rgba(26,82,118,0.25);position:relative;overflow:hidden;">
  <div style="position:absolute;top:-30px;right:-30px;width:150px;height:150px;
              border-radius:50%;background:rgba(255,255,255,0.07);"></div>
  <div style="position:absolute;bottom:-40px;left:80px;width:110px;height:110px;
              border-radius:50%;background:rgba(255,255,255,0.05);"></div>

  <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
    <div style="background:rgba(255,255,255,0.15);border-radius:12px;width:54px;height:54px;
                display:flex;align-items:center;justify-content:center;font-size:26px;">🧠</div>
    <div>
      <div style="font-size:10px;font-weight:600;letter-spacing:3px;text-transform:uppercase;
                  color:rgba(255,255,255,0.6);margin-bottom:2px;">Clinical Decision Support</div>
      <div style="font-family:'Merriweather',serif;font-size:28px;font-weight:700;
                  color:#FFFFFF;line-height:1.1;">MindScan</div>
    </div>
    <div style="margin-left:auto;">
      <span style="background:rgba(255,255,255,0.18);color:#FFFFFF;border-radius:20px;
                   padding:5px 14px;font-size:11px;font-weight:600;letter-spacing:1px;">ML · v1.0</span>
    </div>
  </div>

  <div style="font-size:14px;color:rgba(255,255,255,0.82);line-height:1.65;max-width:520px;margin-bottom:20px;">
    Enter a patient statement or personal journal entry below. MindScan will screen
    for signs of anxiety, depression, or suicidal ideation and provide clinical guidance.
  </div>

  <!-- Condition pills -->
  <div style="display:flex;flex-wrap:wrap;gap:8px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.15);">
    <span style="background:rgba(255,255,255,0.15);color:#FFF;border-radius:20px;
                 padding:5px 14px;font-size:12px;font-weight:500;">&#128560; Anxiety</span>
    <span style="background:rgba(255,255,255,0.15);color:#FFF;border-radius:20px;
                 padding:5px 14px;font-size:12px;font-weight:500;">&#128203; Depression</span>
    <span style="background:rgba(255,255,255,0.15);color:#FFF;border-radius:20px;
                 padding:5px 14px;font-size:12px;font-weight:500;">&#9989; Normal</span>
    <span style="background:rgba(255,255,255,0.15);color:#FFF;border-radius:20px;
                 padding:5px 14px;font-size:12px;font-weight:500;">&#128680; Suicidal Ideation</span>
  </div>
</div>
</body>
</html>""", height=260)

# ── DISCLAIMER (via components.html) ─────────────────────────────────────────
components.html("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>* { box-sizing:border-box; margin:0; padding:0; } body { font-family:'Inter',sans-serif; background:transparent; }</style>
</head><body>
<div style="background:#FFFDE7;border:1px solid #F7CA40;border-left:4px solid #C9A227;
            border-radius:10px;padding:12px 16px;font-size:13px;color:#4A3700;line-height:1.65;">
  <strong style="color:#5D3A00;">⚕ Clinical Disclaimer:</strong>
  This tool is for informational and research purposes only. It does <strong>not</strong>
  constitute a clinical diagnosis and must not replace assessment by a qualified mental
  health professional. If you or someone is in crisis, contact emergency services immediately.
</div>
</body></html>""", height=80)

# ── INPUT LABEL ───────────────────────────────────────────────────────────────
st.markdown("""<div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
color:#2471A3;margin-bottom:4px;font-family:'Inter',sans-serif;">
Patient / User Statement</div>""", unsafe_allow_html=True)

user_text = st.text_area(
    label="input",
    placeholder="Enter the text to analyse — e.g. a journal entry, message, or clinical note…",
    height=160,
    key="user_input",
)

analyze = st.button("🔍  Analyse Mental State", use_container_width=True)

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
if analyze:
    if not user_text.strip():
        st.warning("Please enter some text before analysing.")
    else:
        with st.spinner("Analysing linguistic patterns…"):
            time.sleep(0.3)
            try:
                result = call_api(user_text.strip())
                prediction = result.get("prediction", "Unknown")
                html_output = build_result_html(result)
                # Height auto-sizes: Normal/Anxiety ~520, Suicidal ~580
                height = 580 if prediction == "Suicidal" else 520
                components.html(html_output, height=height, scrolling=False)
            except requests.exceptions.ConnectionError:
                st.error("⚠️ **Could not connect to the backend.** Ensure your FastAPI server is running (`python main.py`).")
            except requests.exceptions.Timeout:
                st.error("⏱️ **Request timed out.** Please try again.")
            except requests.exceptions.HTTPError as e:
                st.error(f"🔴 **Server error:** {e.response.status_code}")
            except Exception as e:
                st.error(f"❌ **Unexpected error:** {e}")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:11px;color:#7F98A8;padding:2rem 0 0.5rem;
            font-family:'Inter',sans-serif;letter-spacing:1px;">
MindScan &nbsp;·&nbsp; For research &amp; educational use only
&nbsp;·&nbsp; Not a substitute for professional medical advice
</div>""", unsafe_allow_html=True)