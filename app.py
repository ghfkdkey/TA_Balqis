"""
Klasifikasi Multi-Label Gejala Depresi, Kecemasan, dan Bipolar
Tugas Akhir - Balqis Eka Nurfadisyah (1202220223)
S1 Sistem Informasi, Fakultas Rekayasa Industri, Universitas Telkom

Aplikasi ini dirancang untuk berjalan langsung di Streamlit Community Cloud
tanpa memerlukan Google Colab atau tunnel (ngrok). Bobot model diambil satu
kali dari Google Drive lalu disimpan pada cache lokal proses (lihat fungsi
`load_model`). Lihat README_DEPLOY.md untuk langkah konfigurasi.
"""

import os
import re
import json
import unicodedata

import numpy as np
import pandas as pd
import torch
import altair as alt
import plotly.graph_objects as go
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────────────────
MODEL_NAME   = "indobenchmark/indobert-large-p2"
MODEL_DIR    = "model_cache"
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model_ASL_seed_456.pt")
CONFIG_PATH  = "config/model_config.json"
MAX_LEN      = 128

TARGET_NAMES        = ["Depresi", "Kecemasan", "Bipolar"]
LABEL_TAG            = {"Depresi": "D", "Kecemasan": "K", "Bipolar": "B"}
FALLBACK_THRESHOLDS  = [0.50, 0.50, 0.50]  # ganti dengan threshold hasil kalibrasi model

LABEL_DESC = {
    "Depresi":   "Kesedihan, kehampaan, atau hilangnya minat yang berlangsung lama.",
    "Kecemasan": "Kekhawatiran, ketegangan, atau rasa takut berlebihan yang sulit dikendalikan.",
    "Bipolar":   "Pergantian suasana hati ekstrem antara episode sangat bersemangat dan sangat terpuruk.",
}

LABEL_INFO = {
    "Depresi": (
        "Depresi ditandai dengan rasa sedih, hampa, atau putus asa yang menetap lebih dari "
        "dua minggu, disertai hilangnya minat pada aktivitas yang biasa disukai, perubahan "
        "pola tidur atau makan, sulit berkonsentrasi, dan kelelahan berkepanjangan."
    ),
    "Kecemasan": (
        "Gangguan kecemasan adalah kekhawatiran atau ketakutan berlebihan yang sulit "
        "dikendalikan, sering muncul tanpa ancaman nyata. Gejalanya meliputi jantung berdebar, "
        "sesak napas, gemetar, dan pikiran yang terus berputar pada kemungkinan buruk."
    ),
    "Bipolar": (
        "Gangguan bipolar ditandai dengan pergantian episode suasana hati ekstrem: episode "
        "manik atau hipomanik (energi tinggi, impulsif) dan episode depresif (sedih mendalam, "
        "tidak bertenaga), yang dapat berlangsung dari hitungan hari hingga minggu."
    ),
}

LABEL_ADVICE = {
    "Depresi": (
        "Pola bahasa yang terdeteksi konsisten dengan gejala depresi. Bicarakan perasaanmu "
        "pada orang terdekat, jaga rutinitas tidur, dan jangan memaksakan diri untuk terlihat "
        "baik-baik saja. Jika perasaan ini menetap lebih dari dua minggu atau mengganggu "
        "aktivitas harian, ini saatnya mencari bantuan profesional."
    ),
    "Kecemasan": (
        "Pola bahasa yang terdeteksi menunjukkan indikasi kecemasan berlebih. Teknik "
        "pernapasan dalam dan latihan grounding dapat membantu meredakan gejala akut. Bila "
        "kecemasan sering muncul dan mengganggu tidur, pekerjaan, atau hubungan sosial, "
        "diskusikan dengan konselor atau psikolog."
    ),
    "Bipolar": (
        "Teks menunjukkan pola perubahan suasana hati yang cukup ekstrem dalam rentang waktu "
        "berdekatan. Catat pola suasana hatimu selama beberapa hari terakhir dan konsultasikan "
        "ke psikiater, karena gangguan bipolar umumnya memerlukan penanganan medis terarah."
    ),
}

# Token desain — lihat README_DEPLOY.md untuk rasional palet dan tipografi
C = {
    "bg":          "#F6F7F5",
    "surface":     "#FFFFFF",
    "border":      "#E2E5E1",
    "ink":         "#1C2321",
    "muted":       "#5B655F",
    "primary":     "#1F4B47",
    "primary_soft":"#E8EFEC",
    "detected":    "#8C3B4A",
    "detected_soft":"#F5E9EA",
    "safe":        "#3F6B4F",
    "safe_soft":   "#EAF1EB",
    "note_bg":     "#F5F1E8",
    "note_ink":    "#7A5C2E",
}

st.set_page_config(page_title="Klasifikasi Gejala Kesehatan Mental", layout="wide")

# ─────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:{C['ink']}; }}
  .stApp {{ background:{C['bg']}; }}
  .block-container {{ max-width:1080px; padding:2.2rem 1.4rem 4rem; }}
  #MainMenu, footer, header {{ visibility:hidden; }}

  .mono {{ font-family:'IBM Plex Mono',monospace; }}

  /* — Landing — */
  .eyebrow {{
    font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.12em;
    text-transform:uppercase; color:{C['muted']}; margin-bottom:14px;
  }}
  .hero-title {{
    font-family:'Fraunces',serif; font-weight:500; font-size:2.6rem; line-height:1.15;
    color:{C['ink']}; margin:0 0 18px; max-width:820px;
  }}
  .byline {{
    font-size:.92rem; color:{C['muted']}; margin-bottom:28px; line-height:1.7;
  }}
  .byline b {{ color:{C['ink']}; font-weight:600; }}
  .rule {{ border:none; border-top:1px solid {C['border']}; margin:26px 0; }}
  .abstract {{
    font-size:1.02rem; line-height:1.85; color:{C['ink']}; max-width:760px; margin-bottom:8px;
  }}

  .spec-grid {{
    display:grid; grid-template-columns:repeat(3,1fr); gap:1px;
    background:{C['border']}; border:1px solid {C['border']}; border-radius:10px;
    overflow:hidden; margin:30px 0;
  }}
  .spec-cell {{ background:{C['surface']}; padding:18px 20px; }}
  .spec-label {{
    font-size:.7rem; letter-spacing:.06em; text-transform:uppercase; color:{C['muted']};
    margin-bottom:6px;
  }}
  .spec-value {{ font-family:'IBM Plex Mono',monospace; font-size:1.15rem; color:{C['primary']}; }}

  .section-title {{
    font-family:'Fraunces',serif; font-size:1.3rem; font-weight:500; margin:0 0 10px;
    color:{C['ink']};
  }}
  .card {{
    background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px;
    padding:22px 24px; margin-bottom:16px;
  }}
  .card p {{ font-size:.92rem; color:{C['muted']}; line-height:1.75; margin:0; }}

  .limit-item {{ display:flex; gap:14px; padding:14px 0; border-top:1px solid {C['border']}; }}
  .limit-item:first-child {{ border-top:none; }}
  .limit-num {{ font-family:'IBM Plex Mono',monospace; color:{C['muted']}; font-size:.85rem; padding-top:2px; }}
  .limit-body b {{ display:block; font-size:.94rem; margin-bottom:3px; }}
  .limit-body p {{ font-size:.87rem; color:{C['muted']}; line-height:1.7; margin:0; }}

  /* — App shell — */
  .app-title {{ font-family:'Fraunces',serif; font-size:1.4rem; font-weight:500; margin:0; }}
  .app-sub {{ font-size:.88rem; color:{C['muted']}; margin:4px 0 0; }}
  .divider {{ border-top:1px solid {C['border']}; margin:18px 0; }}

  .badge {{
    display:inline-flex; align-items:center; justify-content:center;
    width:26px; height:26px; border-radius:50%; font-family:'IBM Plex Mono',monospace;
    font-size:.78rem; font-weight:600; color:#fff; margin-right:8px;
  }}
  .badge-detected {{ background:{C['detected']}; }}
  .badge-safe {{ background:{C['safe']}; }}

  .callout {{
    border-left:3px solid {C['detected']}; background:{C['detected_soft']};
    padding:14px 18px; border-radius:0 8px 8px 0; margin-bottom:14px; font-size:.9rem;
  }}
  .callout.safe {{ border-left-color:{C['safe']}; background:{C['safe_soft']}; }}
  .callout-note {{ font-size:.82rem; color:{C['muted']}; margin-top:6px; }}

  .advice {{
    border-left:3px solid {C['primary']}; background:{C['surface']};
    border:1px solid {C['border']}; border-left-width:3px; border-left-color:{C['primary']};
    padding:14px 18px; border-radius:0 8px 8px 0; margin-top:8px; font-size:.87rem;
    color:{C['ink']}; line-height:1.75;
  }}
  .advice b {{ color:{C['primary']}; }}

  .note-strip {{
    text-align:center; font-size:.8rem; color:{C['note_ink']}; background:{C['note_bg']};
    border-radius:8px; padding:9px 16px; margin:14px 0;
  }}

  .stitle {{
    font-size:.72rem; font-weight:600; color:{C['muted']}; margin:2px 0 10px;
    letter-spacing:.08em; text-transform:uppercase;
  }}

  .metric-row {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:2px; }}
  .metric-name {{ font-size:.86rem; font-weight:500; }}
  .metric-pct  {{ font-family:'IBM Plex Mono',monospace; font-size:.86rem; font-weight:600; }}

  .stButton > button {{
    border-radius:8px !important; font-weight:500 !important; border:1px solid {C['border']} !important;
    transition:all .15s ease !important;
  }}
  .stButton > button:hover {{ transform:translateY(-1px); border-color:{C['primary']} !important; }}

  section[data-testid="stSidebar"] {{ background:{C['surface']}; border-right:1px solid {C['border']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────
for key, default in [("route", "landing"), ("teks_input", ""), ("hasil", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.route == "landing":
    st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)


def goto(route: str):
    st.session_state.route = route


# ─────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# Bobot .pt diambil dari Google Drive sekali per proses lalu di-cache.
# Set GDRIVE_FILE_ID pada Streamlit Secrets. Lihat README_DEPLOY.md.
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thr = FALLBACK_THRESHOLDS
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                thr = json.load(f).get("thresholds", FALLBACK_THRESHOLDS)
        except Exception:
            pass

    if not os.path.exists(MODEL_PATH):
        file_id = st.secrets.get("GDRIVE_FILE_ID", "")
        if not file_id:
            return None, None, None, thr, (
                "GDRIVE_FILE_ID belum diatur di Streamlit Secrets. "
                "Tambahkan file_id checkpoint Google Drive pada menu Settings > Secrets."
            )
        try:
            import gdown
            os.makedirs(MODEL_DIR, exist_ok=True)
            gdown.download(id=file_id, output=MODEL_PATH, quiet=False)
        except Exception as e:
            return None, None, None, thr, f"Gagal mengunduh checkpoint dari Google Drive: {e}"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, problem_type="multi_label_classification")

    try:
        mdl.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        return tok, None, device, thr, f"Gagal memuat bobot model: {e}"

    mdl.to(device).eval()
    return tok, mdl, device, [float(t) for t in thr], None


# ─────────────────────────────────────────────────────────────────────────
# PREPROCESSING & INFERENCE
# ─────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    import emoji as emoji_lib
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^RT[\s]+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = emoji_lib.replace_emoji(text, replace="")
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict(text, tok, mdl, device, thr):
    clean = preprocess(text)
    enc = tok(clean, return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        logits = mdl(**enc).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    preds = [int(p >= t) for p, t in zip(probs, thr)]
    return probs, preds, clean


@st.cache_resource(show_spinner=False)
def build_explainer(_tok, _mdl, _dev):
    import shap
    def f(texts):
        enc = _tok(list(texts), return_tensors="pt", padding=True,
                   truncation=True, max_length=MAX_LEN).to(_dev)
        with torch.no_grad():
            logits = _mdl(**enc).logits
        return torch.sigmoid(logits).cpu().numpy()
    masker = shap.maskers.Text(r"\s+")
    return shap.Explainer(f, masker)


def shap_per_kata(explainer, clean_text, label_idx, max_evals=150):
    sv = explainer([clean_text], max_evals=max_evals, silent=True)
    toks = [str(t).strip() for t in sv.data[0]]
    vals = sv.values[0][:, label_idx]
    df = pd.DataFrame({"kata": toks, "kontribusi": [float(v) for v in vals]})
    df = df[df["kata"].str.len() > 1].copy()
    df["abs"] = df["kontribusi"].abs()
    return df.nlargest(15, "abs").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# VISUAL
# ─────────────────────────────────────────────────────────────────────────
def build_radar(probs, thr, names):
    cats = names + [names[0]]
    r_prob = list(probs) + [float(probs[0])]
    r_thr  = list(thr) + [float(thr[0])]
    marker_colors = [C["detected"] if p >= t else C["safe"] for p, t in zip(probs, thr)]
    marker_colors.append(marker_colors[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_thr, theta=cats, mode="lines",
        line=dict(color="#A9B3AC", dash="dot", width=1.6),
        name="Ambang batas",
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_prob, theta=cats, mode="lines+markers",
        line=dict(color=C["primary"], width=2.2),
        marker=dict(size=8, color=marker_colors, line=dict(width=1, color="#fff")),
        fill="toself", fillcolor="rgba(31,75,71,0.12)",
        name="Skor model",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", gridcolor=C["border"]),
            angularaxis=dict(gridcolor=C["border"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.5, xanchor="center", font=dict(size=11)),
        margin=dict(l=40, r=40, t=20, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["ink"]),
    )
    return fig


def bar_with_threshold(pct, thr, color):
    left = min(max(thr * 100, 1), 98)
    return f"""
    <div style="position:relative;background:{C['border']};border-radius:999px;height:8px;margin:4px 0 12px;">
      <div style="position:absolute;left:0;top:0;bottom:0;width:{pct*100:.1f}%;
                  background:{color};border-radius:999px;"></div>
      <div title="Ambang batas: {thr*100:.0f}%"
           style="position:absolute;left:{left:.1f}%;top:-3px;bottom:-3px;width:2px;
                  background:{C['ink']};"></div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────
def landing_page():
    st.markdown('<div class="eyebrow">Tugas Akhir — S1 Sistem Informasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Klasifikasi Multi-Label Gejala Depresi, '
        'Kecemasan, dan Bipolar pada Media Sosial Indonesia</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div class="byline">
      <b>Balqis Eka Nurfadisyah</b> · NIM 1202220223 · Program Studi S1 Sistem Informasi ·
      Fakultas Rekayasa Industri · Universitas Telkom · 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Ringkasan Penelitian</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="abstract">
    Penelitian ini mengembangkan model kecerdasan buatan berbasis arsitektur transformer
    IndoBERT-Large-p2 untuk klasifikasi multi-label pada teks media sosial berbahasa
    Indonesia. Model mendeteksi tiga indikasi gejala kesehatan mental secara bersamaan —
    depresi, kecemasan, dan bipolar — dan dilatih menggunakan data unggahan dari X (Twitter)
    dan Facebook.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="spec-grid">', unsafe_allow_html=True)
    specs = [
        ("Model", "IndoBERT-Large-p2"),
        ("Jumlah Data", "13.000+"),
        ("Macro F1-Score", "0.7982"),
        ("Sumber Data", "X & Facebook"),
        ("Jenis Klasifikasi", "Multi-label"),
        ("Jumlah Label", "3"),
    ]
    cells = "".join(
        f'<div class="spec-cell"><div class="spec-label">{k}</div>'
        f'<div class="spec-value">{v}</div></div>' for k, v in specs
    )
    st.markdown(cells + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tujuan Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>Aplikasi ini merupakan alat bantu penapisan awal untuk mengenali sinyal kecenderungan
    tekanan psikologis dari teks narasi personal. Tujuannya adalah meningkatkan kesadaran
    masyarakat terhadap kesehatan mental, membantu pengenalan tanda-tanda awal, dan mendorong
    individu segera mencari bantuan dari tenaga profesional yang tepat.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Batasan Aplikasi</div>', unsafe_allow_html=True)
    limits = [
        ("Bukan alat diagnosis",
         "Prediksi murni berdasarkan pola linguistik dan tidak dapat menggantikan diagnosis "
         "klinis. Asesmen presisi hanya dapat dilakukan oleh psikolog atau psikiater berlisensi."),
        ("Konteks teks personal",
         "Model dirancang untuk teks curahan hati atau narasi pengalaman personal orang "
         "pertama. Berita, kutipan motivasi, atau tulisan informatif mungkin tidak relevan."),
        ("Spesifikasi bahasa",
         "Aplikasi memproses teks berbahasa Indonesia termasuk bahasa gaul dan singkatan "
         "internet. Teks campuran atau berbahasa asing tidak diproses secara optimal."),
    ]
    body = '<div class="card">'
    for i, (title, desc) in enumerate(limits, start=1):
        body += (
            f'<div class="limit-item"><div class="limit-num">{i:02d}</div>'
            f'<div class="limit-body"><b>{title}</b><p>{desc}</p></div></div>'
        )
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Buka Aplikasi", type="primary", use_container_width=True):
            goto("app")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# HALAMAN DALAM APLIKASI
# ─────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="app-title">Deteksi Gejala</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-sub">Klasifikasi multi-label berbasis IndoBERT</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        page = st.radio("Menu", ["Deteksi", "Informasi Penelitian"], label_visibility="collapsed")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.caption("Balqis Eka Nurfadisyah")
        st.caption("1202220223 · S1 Sistem Informasi")
        st.caption("Universitas Telkom")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("Kembali ke Beranda", use_container_width=True):
            goto("landing")
            st.rerun()
    return page


def info_page(thresholds):
    st.markdown('<div class="app-title">Informasi Penelitian</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Metode, cakupan, dan batasan sistem.</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">Kondisi yang Dianalisis</div>', unsafe_allow_html=True)
    for name in TARGET_NAMES:
        st.markdown(f"""
        <div class="card">
          <b>{name}</b>
          <p style="margin-top:8px">{LABEL_INFO[name]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="stitle">Ambang Batas Model</div>', unsafe_allow_html=True)
    df_thr = pd.DataFrame({
        "Kondisi": TARGET_NAMES,
        "Threshold": [f"{t*100:.0f}%" for t in thresholds],
    })
    st.dataframe(df_thr, hide_index=True, use_container_width=True)
    st.caption(
        "Model menghasilkan skor probabilitas 0–100% untuk tiap kondisi. Suatu kondisi "
        "dianggap terdeteksi apabila skornya melewati ambang batas yang telah dikalibrasi "
        "pada tahap pelatihan."
    )

    st.markdown("""
    <div class="note-strip">
    Hasil sistem ini tidak menggantikan diagnosis klinis. Diagnosis akhir hanya dapat
    ditegakkan oleh psikolog atau psikiater berlisensi melalui asesmen langsung.
    </div>
    """, unsafe_allow_html=True)


def detection_page(tokenizer, model, device, thresholds):
    st.markdown('<div class="app-title">Deteksi Gejala Kesehatan Mental</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Tulis ceritamu — sistem akan membantu mengenali indikasi awal.</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    EXAMPLES = {
        "Sedih dan putus asa": "udah capek hidup capek kenyataan capek semuanya rasanya pengen nyerah",
        "Cemas berlebihan": "aku takut banget tidak tahu kenapa tiba tiba sesak nafas gemeter susah tenang",
        "Mood naik turun": "kadang aku euforia semangat banget tapi besoknya drop sedih berkepanjangan mood swing parah",
    }

    st.markdown('<div class="stitle">Contoh Teks</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    for col, (label, txt) in zip(ex_cols, EXAMPLES.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"ex_{label}"):
                st.session_state.teks_input = txt
                st.session_state.hasil = None
                st.rerun()

    user_input = st.text_area(
        label="teks", value=st.session_state.teks_input, height=120,
        placeholder="Contoh: akhir-akhir ini aku susah tidur dan merasa tidak berharga",
        label_visibility="collapsed", key="ta_main",
    )
    st.session_state.teks_input = user_input
    n_char = len(user_input.strip())

    c1, c2 = st.columns([3, 2])
    with c1:
        if n_char > 0:
            st.caption(f"{n_char} karakter")
    with c2:
        use_shap = st.checkbox("Tampilkan kata paling berpengaruh", value=False)

    btn = st.button("Analisis Sekarang", type="primary",
                     use_container_width=True, disabled=(n_char < 5))

    st.markdown("""
    <div class="note-strip">
    Alat bantu penapisan berbasis riset, bukan diagnosis medis. Konsultasikan kondisimu
    ke psikolog atau psikiater berlisensi.
    </div>
    """, unsafe_allow_html=True)

    if btn and n_char >= 5:
        with st.spinner("Menganalisis..."):
            probs, preds, clean_text = predict(user_input, tokenizer, model, device, thresholds)
        st.session_state.hasil = (probs, preds, clean_text)

    if st.session_state.hasil:
        probs, preds, clean_text = st.session_state.hasil
        detected = [TARGET_NAMES[i] for i, p in enumerate(preds) if p == 1]

        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="stitle">Profil Skor</div>', unsafe_allow_html=True)
            st.plotly_chart(build_radar(probs, thresholds, TARGET_NAMES), use_container_width=True)

        with right:
            st.markdown('<div class="stitle">Hasil</div>', unsafe_allow_html=True)
            if detected:
                badges = "".join(
                    f'<span class="badge badge-detected">{LABEL_TAG[d]}</span>{d} &nbsp; ' for d in detected
                )
                st.markdown(f"""
                <div class="callout">
                  <b>Terdeteksi indikasi:</b><br>{badges}
                  <div class="callout-note">Bukan berarti pasti terjadi — ini sinyal awal untuk diperhatikan.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="callout safe">
                  <b>Tidak terdeteksi indikasi</b> dari ketiga kondisi yang dianalisis.
                  <div class="callout-note">Tetap jaga kesehatan mentalmu.</div>
                </div>""", unsafe_allow_html=True)

            for i, name in enumerate(TARGET_NAMES):
                pct, thr = float(probs[i]), float(thresholds[i])
                is_on = preds[i] == 1
                color = C["detected"] if is_on else C["safe"]
                st.markdown(f"""
                <div class="metric-row">
                  <span class="metric-name">{name}</span>
                  <span class="metric-pct" style="color:{color}">{pct*100:.1f}%</span>
                </div>
                {bar_with_threshold(pct, thr, color)}
                """, unsafe_allow_html=True)
            st.caption("Garis tegak menandai ambang batas deteksi tiap kondisi.")

        if detected:
            st.markdown('<div class="stitle" style="margin-top:6px">Yang Bisa Kamu Lakukan</div>', unsafe_allow_html=True)
            for d in detected:
                st.markdown(f'<div class="advice"><b>{d}</b><br>{LABEL_ADVICE[d]}</div>', unsafe_allow_html=True)
            if len(detected) > 1:
                st.markdown(f"""
                <div class="advice">
                <b>Catatan</b><br>Lebih dari satu indikasi muncul sekaligus ({', '.join(detected)}).
                Gejala kondisi mental sering tumpang tindih, sehingga evaluasi menyeluruh oleh
                psikolog atau psikiater sangat disarankan.
                </div>
                """, unsafe_allow_html=True)

        if use_shap:
            with st.expander("Kata Paling Berpengaruh", expanded=True):
                st.caption(
                    "Batang hijau mendorong prediksi ke arah terdeteksi. "
                    "Batang merah menahan prediksi ke arah tidak terdeteksi."
                )
                show_idxs = [i for i in range(3) if preds[i] == 1] or [int(np.argmax(probs))]
                shap_cols = st.columns(len(show_idxs))
                try:
                    explainer = build_explainer(tokenizer, model, device)
                    for col, idx in zip(shap_cols, show_idxs):
                        with col:
                            with st.spinner(f"Menghitung kontribusi {TARGET_NAMES[idx]}..."):
                                df_shap = shap_per_kata(explainer, clean_text, idx, max_evals=150)
                            if df_shap.empty:
                                st.warning("Tidak cukup kata untuk dianalisis.")
                                continue
                            st.markdown(f"**{TARGET_NAMES[idx]}**")
                            chart = (
                                alt.Chart(df_shap)
                                .mark_bar(cornerRadiusEnd=4)
                                .encode(
                                    x=alt.X("kontribusi:Q", title="Kontribusi",
                                            axis=alt.Axis(format=".3f", labelFontSize=10)),
                                    y=alt.Y("kata:N",
                                            sort=alt.EncodingSortField(field="abs", order="descending"),
                                            title=None, axis=alt.Axis(labelFontSize=12, labelLimit=120)),
                                    color=alt.condition(alt.datum.kontribusi > 0,
                                                         alt.value(C["safe"]), alt.value(C["detected"])),
                                    tooltip=[alt.Tooltip("kata:N", title="Kata"),
                                             alt.Tooltip("kontribusi:Q", title="Kontribusi", format=".4f")],
                                )
                                .properties(height=max(220, len(df_shap) * 28))
                                .configure_axis(grid=False)
                                .configure_view(strokeOpacity=0)
                            )
                            st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.warning(f"Analisis kontribusi kata tidak dapat dijalankan: {e}")

        st.markdown("&nbsp;")
        if st.button("Coba Teks Lain", use_container_width=True):
            st.session_state.teks_input = ""
            st.session_state.hasil = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.route == "landing":
    landing_page()
else:
    page = render_sidebar()
    tokenizer, model, device, thresholds, load_err = load_model()

    if load_err:
        st.error(load_err)
        st.stop()

    if page == "Informasi Penelitian":
        info_page(thresholds)
    else:
        detection_page(tokenizer, model, device, thresholds)