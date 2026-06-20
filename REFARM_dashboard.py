import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="لوحة إدارة المخلفات الزراعية - الشرقية",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Farmer-Friendly Color Palette ───────────────────────────────────────────
COLORS = {
    "soil_dark":    "#3D2B1F",
    "soil_mid":     "#6B4226",
    "soil_light":   "#A0522D",
    "wheat":        "#D4A843",
    "wheat_light":  "#E8C97A",
    "straw":        "#C8B560",
    "leaf_dark":    "#2D5016",
    "leaf_mid":     "#4A7C1F",
    "leaf_light":   "#7AAB3A",
    "sage":         "#8FAF6A",
    "mint":         "#B5CCB0",
    "sky":          "#6B9DB5",
    "sky_light":    "#A8C8D8",
    "rust":         "#A0522D",
    "clay_red":     "#8B3A2A",
    "parchment":    "#F5EDD8",
    "cream":        "#EDE3C8",
    "linen":        "#E5D9BE",
    "sidebar_bg":   "#2D4A1A",
    "sidebar_text": "#E8D8A0",
    "card_bg":      "#F0E8D0",
    "text_dark":    "#1C1008",
    "text_mid":     "#4A3520",
    "text_muted":   "#7A6040",
    "success":      "#2D7A1F",
    "warning":      "#C8860A",
    "danger":       "#A0341A",
    "info":         "#1A6B8A",
}

CROP_COLORS = {
    "Wheat":      "#D4A843",
    "Rice":       "#4A7C1F",
    "Maize":      "#C8860A",
    "Sugar Beet": "#8B4513",
    "Potato":     "#7A6A4A",
}

SOIL_COLORS = {
    "Loamy": "#A0522D",
    "Clay":  "#6B4226",
    "Sandy": "#D4A843",
}

HARVEST_COLORS = {
    "Manual":     "#2D5016",
    "Mechanical": "#C8860A",
}

CENTER_COLORS = [
    "#4A7C1F","#D4A843","#C8860A","#A0522D",
    "#6B4226","#2D5016","#8FAF6A","#8B3A2A"
]

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800;900&family=Tajawal:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cairo', 'Tajawal', sans-serif !important;
    }

    .stApp {
        background: 
            radial-gradient(ellipse at 10% 20%, rgba(45,80,22,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(212,168,67,0.08) 0%, transparent 50%),
            linear-gradient(160deg, #F5EDD8 0%, #EDE3C8 40%, #F0E8D0 100%);
        color: #1C1008;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3510 0%, #2D5016 40%, #1A3A0C 100%) !important;
        border-right: 3px solid #D4A843;
        box-shadow: 4px 0 20px rgba(45,80,22,0.3);
    }
  [data-testid="stSidebar"] * { 
        color: #E8D8A0 !important; 
        font-family: 'Cairo', sans-serif !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #D4A843 !important;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.5px;
    }
    [data-testid="stSidebar"] hr {
        border-color: #4A7C1F !important;
        opacity: 0.6;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #F0E8D0 0%, #E8D8B8 100%);
        border: 1px solid #C8B090;
        border-top: 4px solid #4A7C1F;
        border-radius: 14px;
        padding: 18px 20px !important;
        box-shadow: 0 3px 12px rgba(45,80,22,0.10), 0 1px 3px rgba(0,0,0,0.06);
        transition: all .25s ease;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 60px; height: 60px;
        background: radial-gradient(circle, rgba(212,168,67,0.12), transparent 70%);
        border-radius: 50%;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(74,124,31,0.18);
        border-top-color: #D4A843;
    }
    [data-testid="stMetric"] label { 
        color: #7A6040 !important; 
        font-size: 0.78rem; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1C1008 !important;
        font-size: 1.65rem !important;
        font-weight: 900 !important;
        font-family: 'Cairo', sans-serif !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #4A7C1F !important;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .section-header {
        background: linear-gradient(90deg, #2D5016 0%, #4A7C1F 50%, rgba(74,124,31,0) 100%);
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 0.98rem;
        font-weight: 700;
        color: #F5EDD8 !important;
        margin-bottom: 14px;
        letter-spacing: 0.3px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .page-title {
        background: linear-gradient(135deg, #1E3510 0%, #2D5016 30%, #4A7C1F 65%, #3D6A18 100%);
        border-radius: 16px;
        padding: 22px 36px;
        font-size: 1.55rem;
        font-weight: 900;
        color: #F5EDD8;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 6px 24px rgba(45,80,22,0.28), inset 0 1px 0 rgba(212,168,67,0.3);
        border: 1px solid #4A7C1F;
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
    }
    .page-title::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #D4A843, transparent);
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(74,124,31,0.08), rgba(212,168,67,0.06));
        border: 1px solid rgba(74,124,31,0.25);
        border-right: 5px solid #D4A843;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #2A1C08;
        font-size: 0.94rem;
        line-height: 1.7;
        text-align: right;
        box-shadow: 0 2px 8px rgba(45,80,22,0.06);
    }
    .insight-box.danger {
        background: linear-gradient(135deg, rgba(160,82,45,0.08), rgba(139,58,42,0.05));
        border-right-color: #A0341A;
        border-color: rgba(160,82,45,0.3);
    }
    .insight-box.success {
        background: linear-gradient(135deg, rgba(45,80,22,0.10), rgba(74,124,31,0.06));
        border-right-color: #2D5016;
        border-color: rgba(45,80,22,0.25);
    }
    .insight-box.info {
        background: linear-gradient(135deg, rgba(26,107,138,0.08), rgba(107,157,181,0.05));
        border-right-color: #1A6B8A;
        border-color: rgba(26,107,138,0.25);
    }

    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #1E3510, #2D5016);
        border-radius: 12px;
        padding: 5px 6px;
        gap: 4px;
        border: 1px solid #4A7C1F;
        box-shadow: 0 3px 12px rgba(45,80,22,0.20);
    }
    .stTabs [data-baseweb="tab"] {
        color: #B8D090 !important;
        background: transparent !important;
        border-radius: 8px !important;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 14px !important;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(74,124,31,0.3) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #D4A843, #C89830) !important;
        color: #1C1008 !important;
        font-weight: 800 !important;
        box-shadow: 0 2px 8px rgba(212,168,67,0.3);
    }

    .stRadio > div { display: flex; flex-direction: column; gap: 7px; }
    .stRadio input[type="radio"] { display: none; }
    .stRadio label {
        display: block !important;
        width: 100% !important;
        min-height: 40px !important;
        background: rgba(255,255,255,0.06);
        border: 1.5px solid rgba(212,168,67,0.35);
        border-radius: 10px;
        padding: 9px 12px;
        margin: 0 !important;
        color: #E8D8A0 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-align: center;
        transition: all 0.22s ease;
        cursor: pointer;
        box-sizing: border-box;
        letter-spacing: 0.3px;
    }
    .stRadio label:hover {
        background: rgba(74,124,31,0.3);
        border-color: #D4A843;
        transform: translateX(-2px);
        box-shadow: 2px 0 12px rgba(212,168,67,0.2);
    }
    .stRadio label:has(input[type="radio"]:checked) {
        background: linear-gradient(135deg, #D4A843, #C89830) !important;
        color: #1C1008 !important;
        border: 2px solid #F5EDD8;
        box-shadow: 0 4px 16px rgba(212,168,67,0.35);
        font-weight: 800 !important;
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #F0E8D0 !important;
        border-color: #C8B090 !important;
        color: #1C1008 !important;
        border-radius: 9px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2D5016, #1E3510) !important;
        color: #F5EDD8 !important;
        border: 2px solid #D4A843 !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-family: 'Cairo', sans-serif !important;
        width: 100%;
        padding: 12px !important;
        font-size: 1rem !important;
        transition: all 0.3s;
        letter-spacing: 0.5px;
        box-shadow: 0 3px 12px rgba(45,80,22,0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #D4A843, #C89830) !important;
        color: #1C1008 !important;
        box-shadow: 0 6px 20px rgba(212,168,67,0.35) !important;
        transform: translateY(-2px);
    }

    .sidebar-logo {
        text-align: center;
        padding: 16px 10px 18px;
        border-bottom: 2px solid rgba(212,168,67,0.5);
        margin-bottom: 20px;
    }
    .sidebar-logo .logo-icon { font-size: 2.8rem; line-height: 1; margin-bottom: 8px; }
    .sidebar-logo h2 { color: #F5EDD8 !important; font-size: 1.0rem; margin: 4px 0; font-weight: 800; }
    .sidebar-logo p { color: #B8A878 !important; font-size: 0.8rem; margin: 2px 0 0 0; }

    .stat-card {
        background: linear-gradient(145deg, #EDE3C8, #E5D9B8);
        border: 1px solid #C8B090;
        border-left: 5px solid #4A7C1F;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 6px 0;
        text-align: right;
    }
    .stat-card h4 { color: #7A6040; font-size: 0.8rem; margin: 0 0 4px 0; font-weight: 600; }
    .stat-card .value { color: #1C1008; font-size: 1.4rem; font-weight: 900; }
    .stat-card .delta { color: #4A7C1F; font-size: 0.78rem; }

    .stProgress > div > div { 
        background: linear-gradient(90deg, #2D5016, #4A7C1F, #D4A843) !important; 
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #E5D9BE; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(#4A7C1F, #2D5016); border-radius: 4px; }

    [data-testid="stDataFrame"] {
        border: 1px solid #C8B090;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(45,80,22,0.08);
    }

    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #F0E8D0, #EDE3C8) !important;
        color: #1C1008 !important;
        border-radius: 10px !important;
        border: 1px solid #C8B090 !important;
        font-weight: 700 !important;
    }

    .stNumberInput > div > div > input {
        background: #F0E8D0 !important;
        border-color: #C8B090 !important;
        color: #1C1008 !important;
        border-radius: 8px !important;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #2D5016, #4A7C1F) !important;
    }

    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    hr { border-color: rgba(74,124,31,0.2) !important; }

    .farm-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4A7C1F, #2D5016);
        color: #F5EDD8 !important;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 2px;
    }

    .alert-box {
        background: linear-gradient(135deg, rgba(200,134,10,0.12), rgba(212,168,67,0.08));
        border: 1px solid rgba(200,134,10,0.4);
        border-left: 5px solid #C8860A;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
        font-weight: 600;
        color: #3A2010;
    }

    .tip-card {
        background: linear-gradient(135deg, #F0E8D0, #EAE0C5);
        border: 1px solid #C8B090;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        transition: all 0.25s;
        height: 100%;
    }
    .tip-card:hover {
        border-color: #4A7C1F;
        box-shadow: 0 4px 16px rgba(74,124,31,0.14);
        transform: translateY(-2px);
    }
    .tip-card .tip-icon { font-size: 2.2rem; margin-bottom: 10px; }
    .tip-card h4 { color: #2D5016; font-size: 0.92rem; margin: 0 0 6px 0; font-weight: 800; }
    .tip-card p { color: #5A4030; font-size: 0.82rem; margin: 0; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('sharqia_agri_waste.csv')
    return df

df_raw = load_data()

# ─── Plotly layout helper ─────────────────────────────────────────────────────
def farm_layout(fig, title="", height=420):
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            font=dict(family="Cairo", size=14, color="#2D5016"),
            x=0.5
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,232,208,0.45)",
        font=dict(family="Cairo", color="#1C1008"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(240,232,208,0.92)",
            bordercolor="rgba(200,176,144,0.7)",
            borderwidth=1,
            font=dict(family="Cairo", size=11)
        ),
        xaxis=dict(
            gridcolor="rgba(74,124,31,0.12)",
            zerolinecolor="rgba(0,0,0,0.08)",
            color="#4A3520",
            linecolor="rgba(74,124,31,0.25)",
        ),
        yaxis=dict(
            gridcolor="rgba(74,124,31,0.12)",
            zerolinecolor="rgba(0,0,0,0.08)",
            color="#4A3520",
            linecolor="rgba(74,124,31,0.25)",
        ),
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🌿</div>
        <h2>إدارة المخلفات الزراعية</h2>
        <p>محافظة الشرقية – مصر 🇪🇬</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "القائمة",
        options=[
            "🏠 نظرة_عامة",
            "📊 تحليل_المخلفات",
            "🌾 المحاصيل_والانتاجية",
            "🌍 الأثر_البيئي",
            "🚜 الحصاد_والتربة",
            "🗺️ الخريطة_الجغرافية",
            "🤖 نموذج_التنبؤ",
            "💡 توصيات_المزارع",
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    total_farms = len(df_raw)
    total_waste = df_raw['waste_tons'].sum()
    avg_waste_ratio = df_raw['waste_ratio'].mean() * 100
    
    st.markdown(f"""
    <div style='background:rgba(212,168,67,0.15);border:1px solid rgba(212,168,67,0.4);border-radius:10px;padding:12px;text-align:center;'>
        <div style='color:#D4A843;font-size:0.75rem;font-weight:700;margin-bottom:8px;'>📊 إحصائيات سريعة</div>
        <div style='color:#E8D8A0;font-size:0.82rem;'>🏡 {total_farms:,} مزرعة</div>
        <div style='color:#E8D8A0;font-size:0.82rem;'>🗑️ {total_waste:,.0f} طن مخلفات</div>
        <div style='color:#E8D8A0;font-size:0.82rem;'>⚠️ {avg_waste_ratio:.1f}% متوسط النسبة</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Working copy ─────────────────────────────────────────────────────────────
df = df_raw.copy()

if df.empty:
    st.error("⚠️ لا توجد بيانات. يرجى التحقق من الملف.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – نظرة_عامة
# ═══════════════════════════════════════════════════════════════════════════════
if "نظرة" in page:
    st.markdown('<div class="page-title">🏠 نظرة عامة شاملة – محافظة الشرقية الزراعية</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("🏡 إجمالي المزارع",         f"{len(df):,}",                     delta="جميع المراكز")
    with col2: st.metric("📐 المساحة الكلية (فدان)",   f"{df['area_feddan'].sum():,.0f}",  delta=f"متوسط {df['area_feddan'].mean():.1f}/مزرعة")
    with col3: st.metric("🌾 الإنتاج الكلي (طن)",      f"{df['total_yield'].sum():,.0f}",  delta=f"كفاءة {df['yield_per_feddan'].mean():.1f} ط/فدان")
    with col4: st.metric("🗑️ المخلفات الكلية (طن)",    f"{df['waste_tons'].sum():,.0f}",   delta=f"متوسط {df['waste_tons'].mean():.1f}/مزرعة")
    with col5:
        avg_ratio = df['waste_ratio'].mean() * 100
        st.metric("⚠️ نسبة المخلفات",                 f"{avg_ratio:.1f}%",                delta="من إجمالي الإنتاج")

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="section-header">🌾 توزيع المحاصيل المزروعة</div>', unsafe_allow_html=True)
        crop_counts = df["crop_type"].value_counts().reset_index()
        crop_counts.columns = ["crop", "count"]
        fig = px.pie(
            crop_counts, names="crop", values="count",
            color="crop", color_discrete_map=CROP_COLORS, hole=0.50
        )
        fig.update_traces(
            textinfo="label+percent",
            textfont=dict(size=13, family="Cairo"),
            marker=dict(line=dict(color="#F5EDD8", width=2.5)),
            pull=[0.04] * len(crop_counts)
        )
        farm_layout(fig, "نسبة كل محصول")
        fig.update_layout(
            annotations=[dict(text=f"<b>{len(df):,}</b><br>مزرعة", x=0.5, y=0.5,
                                font=dict(size=16, family="Cairo", color="#2D5016"),
                                showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">🏙️ عدد المزارع حسب المركز</div>', unsafe_allow_html=True)
        center_counts = df["center"].value_counts().reset_index()
        center_counts.columns = ["center", "count"]
        fig2 = px.bar(
            center_counts, x="count", y="center", orientation="h",
            color="count",
            color_continuous_scale=["#D4A843", "#7AAB3A", "#4A7C1F", "#2D5016"]
        )
        fig2.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="عدد المزارع")
        fig2.update_traces(marker_line_color="#1E3510", marker_line_width=0.8)
        farm_layout(fig2, "المزارع بكل مركز")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">📊 الإنتاج مقابل المخلفات لكل محصول</div>', unsafe_allow_html=True)
        agg = df.groupby("crop_type").agg(
            total_waste=("waste_tons", "sum"),
            total_yield=("total_yield", "sum")
        ).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="الإنتاج (طن)",  x=agg["crop_type"], y=agg["total_yield"],
                                marker_color="#4A7C1F", marker_line_color="#2D5016", marker_line_width=0.8))
        fig3.add_trace(go.Bar(name="المخلفات (طن)", x=agg["crop_type"], y=agg["total_waste"],
                                marker_color="#A0522D", marker_line_color="#6B4226", marker_line_width=0.8))
        fig3.update_layout(barmode="group", legend_orientation="h")
        farm_layout(fig3, "الإنتاج مقابل المخلفات")
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">🔵 المساحة – المخلفات – الإنتاج (حجم)</div>', unsafe_allow_html=True)
        center_agg = df.groupby("center").agg(
            waste=("waste_tons", "sum"),
            yield_=("total_yield", "sum"),
            area=("area_feddan", "sum")
        ).reset_index()
        fig4 = px.scatter(
            center_agg, x="area", y="waste", size="yield_",
            color="center", text="center",
            color_discrete_sequence=CENTER_COLORS
        )
        fig4.update_traces(textposition="top center", textfont=dict(size=10, family="Cairo"))
        farm_layout(fig4, "المساحة vs المخلفات (الحجم = الإنتاج)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📈 نسبة المخلفات لكل مركز ومحصول</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📍 حسب المركز", "🌾 حسب المحصول", "🔄 مقارنة سنوية"])
    
    with tabs[0]:
        center_waste = df.groupby("center")["waste_ratio"].mean().reset_index()
        center_waste["waste_pct"] = center_waste["waste_ratio"] * 100
        center_waste = center_waste.sort_values("waste_pct", ascending=False)
        
        fig_cw = px.bar(center_waste, x="center", y="waste_pct",
                        color="waste_pct",
                        color_continuous_scale=["#7AAB3A", "#D4A843", "#C8860A", "#A0341A"],
                        text="waste_pct")
        fig_cw.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                                textfont=dict(size=11, family="Cairo"))
        fig_cw.update_layout(coloraxis_showscale=False, xaxis_title="المركز", yaxis_title="نسبة المخلفات %")
        farm_layout(fig_cw, "متوسط نسبة المخلفات لكل مركز", height=360)
        st.plotly_chart(fig_cw, use_container_width=True)
    
    with tabs[1]:
        crop_waste = df.groupby("crop_type")["waste_ratio"].mean().reset_index()
        crop_waste["waste_pct"] = crop_waste["waste_ratio"] * 100
        crop_waste = crop_waste.sort_values("waste_pct", ascending=False)
        
        fig_crop_w = px.bar(crop_waste, x="crop_type", y="waste_pct",
                            color="crop_type", color_discrete_map=CROP_COLORS, text="waste_pct")
        fig_crop_w.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                                    textfont=dict(size=12, family="Cairo"))
        fig_crop_w.update_layout(showlegend=False, xaxis_title="المحصول", yaxis_title="نسبة المخلفات %")
        farm_layout(fig_crop_w, "متوسط نسبة المخلفات لكل محصول", height=360)
        st.plotly_chart(fig_crop_w, use_container_width=True)
    
    with tabs[2]:
        st.markdown(
            '<div class="insight-box info">ℹ️ هذا التحليل يعرض توزيع المخلفات عبر مجموعات المزارع لمحاكاة المقارنة الزمنية.</div>',
            unsafe_allow_html=True
        )
        df_sorted = df.sort_values("waste_tons").reset_index(drop=True)
        df_sorted["farm_rank"] = range(1, len(df_sorted)+1)
        fig_trend = px.area(df_sorted.iloc[::5], x="farm_rank", y="waste_tons",
                            color="crop_type", color_discrete_map=CROP_COLORS,
                            line_group="crop_type")
        farm_layout(fig_trend, "توزيع المخلفات عبر المزارع (مرتبة تصاعدياً)", height=360)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    
    st.markdown('<div class="section-header">💡 أبرز المؤشرات الزراعية</div>', unsafe_allow_html=True)

    top_waste_crop  = df.groupby("crop_type")["waste_tons"].sum().idxmax()
    top_center      = df.groupby("center")["waste_tons"].sum().idxmax()
    most_common     = df["harvest_method"].mode()[0]
    best_yield_crop = df.groupby("crop_type")["yield_per_feddan"].mean().idxmax()
    best_center_eff = df.groupby("center")["yield_per_feddan"].mean().idxmax()
    low_waste_crop  = df.groupby("crop_type")["waste_ratio"].mean().idxmin()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="insight-box danger">🏆 <b>أعلى محصول في المخلفات</b><br>يتصدر <b>{top_waste_crop}</b> قائمة المحاصيل الأكثر إنتاجاً للمخلفات، مما يستدعي مراجعة أسلوب الحصاد والتخزين.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box danger">📍 <b>أعلى مركز في المخلفات</b><br>مركز <b>{top_center}</b> يسجل أعلى معدلات المخلفات الزراعية ويحتاج خطة عاجلة للحد من الهدر.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="insight-box">🚜 <b>أكثر طرق الحصاد استخداماً</b><br>يُعدّ الحصاد <b>{most_common}</b> الأكثر شيوعاً في المنطقة، ويؤثر مباشرةً على كمية المخلفات الناتجة.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box info">🏙️ <b>أكفأ مركز زراعي</b><br>مركز <b>{best_center_eff}</b> يحقق أعلى إنتاجية لكل فدان، ويُنصح بدراسة أساليبه وتعميمها.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="insight-box success">🌟 <b>أعلى محصول إنتاجية/فدان</b><br>يحقق <b>{best_yield_crop}</b> أفضل عائد زراعي لكل فدان، مما يجعله الخيار الأمثل لتحسين الربحية.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box success">✅ <b>أقل محصول في نسبة المخلفات</b><br><b>{low_waste_crop}</b> يُسجّل أدنى نسبة هدر، ويُمثّل نموذجاً ناجحاً لإدارة المحاصيل.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – تحليل_المخلفات
# ═══════════════════════════════════════════════════════════════════════════════
elif "تحليل" in page:
    st.markdown('<div class="page-title">📊 تحليل المخلفات الزراعية – تعمق في البيانات</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🗑️ إجمالي المخلفات",   f"{df['waste_tons'].sum():,.1f} طن",  delta=f"متوسط {df['waste_tons'].mean():.1f}/مزرعة")
    with col2: st.metric("📊 الوسيط (طن)",        f"{df['waste_tons'].median():.1f} طن")
    with col3: st.metric("📈 أعلى كمية",          f"{df['waste_tons'].max():.1f} طن")
    with col4: st.metric("⚠️ متوسط النسبة",       f"{df['waste_ratio'].mean()*100:.1f}%", delta=f"σ={df['waste_ratio'].std()*100:.1f}%")

    st.markdown("---")

    # ── Advanced filter ──────────────────────────────────────────────────────
    # Initialize df_f before the expander so it's always defined
    df_f = df.copy()

    with st.expander("🔍 تصفية متقدمة للبيانات", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_crops   = st.multiselect("🌾 المحصول", df["crop_type"].unique(), default=list(df["crop_type"].unique()))
        with fc2:
            sel_centers = st.multiselect("📍 المركز",  df["center"].unique(),    default=list(df["center"].unique()))
        with fc3:
            sel_methods = st.multiselect("🚜 طريقة الحصاد", df["harvest_method"].unique(), default=list(df["harvest_method"].unique()))

        waste_range = st.slider(
            "نطاق المخلفات (طن)",
            float(df["waste_tons"].min()), float(df["waste_tons"].max()),
            (float(df["waste_tons"].min()), float(df["waste_tons"].max()))
        )

        df_f = df[
            df["crop_type"].isin(sel_crops) &
            df["center"].isin(sel_centers) &
            df["harvest_method"].isin(sel_methods) &
            df["waste_tons"].between(*waste_range)
        ]
        st.info(f"📋 السجلات المصفّاة: **{len(df_f):,}** من أصل **{len(df):,}**")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">📉 توزيع كميات المخلفات</div>', unsafe_allow_html=True)
        fig = px.histogram(df_f, x="waste_tons", nbins=45,
                            color_discrete_sequence=["#4A7C1F"])
        fig.update_traces(marker_line_color="#2D5016", marker_line_width=1)
        fig.add_vline(x=df_f["waste_tons"].mean(), line_dash="dash",
                        line_color="#D4A843", line_width=2,
                        annotation_text=f"متوسط: {df_f['waste_tons'].mean():.1f}",
                        annotation_font=dict(color="#D4A843", family="Cairo"))
        farm_layout(fig, "توزيع المخلفات (طن)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">📦 مخلفات كل محصول – Box Plot</div>', unsafe_allow_html=True)
        fig2 = px.box(df_f, x="crop_type", y="waste_tons",
                        color="crop_type", color_discrete_map=CROP_COLORS,
                        notched=True)
        fig2.update_traces(boxmean="sd")
        farm_layout(fig2, "توزيع المخلفات لكل محصول")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🗺️ خريطة حرارية – نسبة المخلفات (محصول × مركز)</div>', unsafe_allow_html=True)

    pivot = df_f.pivot_table(index="center", columns="crop_type", values="waste_ratio", aggfunc="mean") * 100
    fig3  = px.imshow(
        pivot,
        color_continuous_scale=["#F5EDD8", "#D4C898", "#D4A843", "#C8860A", "#A0341A"],
        text_auto=".1f", aspect="auto",
        zmin=0, zmax=pivot.values.max()
    )
    fig3.update_traces(textfont=dict(size=13, family="Cairo", color="#1C1008"))
    farm_layout(fig3, "نسبة المخلفات % – محصول × مركز", height=360)
    fig3.update_layout(coloraxis_colorbar=dict(
        title="نسبة %",
        tickfont=dict(color="#4A3520", family="Cairo"),
        title_font=dict(color="#4A3520", family="Cairo")
    ))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">🔁 المخلفات حسب طريقة الحصاد</div>', unsafe_allow_html=True)
        h_agg = df_f.groupby(["crop_type", "harvest_method"])["waste_tons"].mean().reset_index()
        fig4  = px.bar(h_agg, x="crop_type", y="waste_tons",
                        color="harvest_method", barmode="group",
                        color_discrete_map=HARVEST_COLORS)
        fig4.update_layout(xaxis_title="المحصول", yaxis_title="متوسط المخلفات (طن)")
        farm_layout(fig4, "متوسط المخلفات: يدوي مقابل آلي")
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">🪨 المخلفات حسب نوع التربة</div>', unsafe_allow_html=True)
        s_agg = df_f.groupby(["crop_type", "soil_type"])["waste_tons"].mean().reset_index()
        fig5  = px.bar(s_agg, x="crop_type", y="waste_tons",
                        color="soil_type", barmode="group",
                        color_discrete_map=SOIL_COLORS)
        fig5.update_layout(xaxis_title="المحصول", yaxis_title="متوسط المخلفات (طن)")
        farm_layout(fig5, "متوسط المخلفات حسب نوع التربة")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    
    st.markdown('<div class="section-header">🎻 توزيع نسبة المخلفات لكل مركز</div>', unsafe_allow_html=True)
    fig6 = px.violin(df_f, x="center", y="waste_ratio",
                        color="center", box=True, points=False,
                        color_discrete_sequence=CENTER_COLORS)
    fig6.update_layout(showlegend=False, xaxis_title="المركز", yaxis_title="نسبة المخلفات")
    farm_layout(fig6, "توزيع نسبة المخلفات حسب المركز", height=360)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📋 جدول ملخص المخلفات التفصيلي</div>', unsafe_allow_html=True)

    summary = df_f.groupby(["center", "crop_type"]).agg(
        عدد_المزارع=("farm_id", "count"),
        متوسط_المخلفات=("waste_tons", "mean"),
        اجمالي_المخلفات=("waste_tons", "sum"),
        اعلى_كمية=("waste_tons", "max"),
        متوسط_النسبة=("waste_ratio", lambda x: x.mean() * 100),
        متوسط_الإنتاجية=("yield_per_feddan", "mean"),
    ).round(2).reset_index()
    summary.columns = ["المركز","المحصول","عدد المزارع","متوسط المخلفات (طن)","إجمالي المخلفات (طن)","أعلى كمية (طن)","متوسط النسبة %","متوسط الإنتاجية"]
    st.dataframe(summary, use_container_width=True, height=360)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – المحاصيل_والانتاجية
# ═══════════════════════════════════════════════════════════════════════════════
elif "المحاصيل" in page:
    st.markdown('<div class="page-title">🌾 المحاصيل والإنتاجية الزراعية – تحليل شامل</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🌾 إجمالي الإنتاج",         f"{df['total_yield'].sum():,.0f} طن")
    with col2: st.metric("📐 متوسط الإنتاج/فدان",     f"{df['yield_per_feddan'].mean():.2f} طن")
    with col3: st.metric("🏆 أعلى إنتاجية/فدان",      f"{df['yield_per_feddan'].max():.2f} طن")
    with col4: st.metric("📊 أنواع المحاصيل",         f"{df['crop_type'].nunique()} أنواع")

    st.markdown("---")

    tabs = st.tabs(["📊 الإنتاج الكلي", "🔗 العلاقات", "🌍 الجغرافيا", "📐 مصفوفة الارتباط"])

    with tabs[0]:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-header">🌾 إجمالي الإنتاج لكل محصول</div>', unsafe_allow_html=True)
            prod = df.groupby("crop_type")["total_yield"].sum().reset_index().sort_values("total_yield", ascending=True)
            fig  = px.bar(prod, x="total_yield", y="crop_type", orientation="h",
                            color="crop_type", color_discrete_map=CROP_COLORS, text="total_yield")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                                textfont=dict(family="Cairo", size=11))
            fig.update_layout(showlegend=False, xaxis_title="إجمالي الإنتاج (طن)", yaxis_title="")
            farm_layout(fig, "إجمالي الإنتاج لكل محصول")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">📐 مقارنة متوسط الإنتاجية/فدان</div>', unsafe_allow_html=True)
            ypf = df.groupby("crop_type")["yield_per_feddan"].agg(["mean","std","min","max"]).reset_index()
            fig2 = go.Figure()
            for i, row in ypf.iterrows():
                crop = row["crop_type"]
                color = CROP_COLORS.get(crop, "#4A7C1F")
                fig2.add_trace(go.Bar(
                    name=crop, x=[crop],
                    y=[row["mean"]],
                    error_y=dict(type="data", array=[row["std"]], visible=True, color=color),
                    marker_color=color,
                    marker_line_color="#1C1008", marker_line_width=0.8
                ))
            fig2.update_layout(showlegend=False, xaxis_title="المحصول", yaxis_title="طن/فدان")
            farm_layout(fig2, "متوسط الإنتاجية مع الانحراف المعياري")
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">🔗 المساحة والإنتاج (مع خط الاتجاه)</div>', unsafe_allow_html=True)
            fig3 = px.scatter(df, x="area_feddan", y="total_yield",
                                color="crop_type", color_discrete_map=CROP_COLORS,
                                trendline="ols", trendline_color_override="#A0341A",
                                opacity=0.6, size="waste_tons", size_max=15)
            fig3.update_traces(marker=dict(line=dict(width=0.5, color="#1C1008")))
            farm_layout(fig3, "المساحة vs الإنتاج (الحجم = المخلفات)", height=400)
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">🎻 توزيع الإنتاجية/فدان</div>', unsafe_allow_html=True)
            fig4 = px.violin(df, x="crop_type", y="yield_per_feddan",
                            color="crop_type", color_discrete_map=CROP_COLORS,
                            box=True, points="outliers")
            fig4.update_layout(showlegend=False, xaxis_title="المحصول", yaxis_title="طن/فدان")
            farm_layout(fig4, "توزيع الإنتاجية/فدان لكل محصول", height=400)
            st.plotly_chart(fig4, use_container_width=True)

    with tabs[2]:
        st.markdown('<div class="section-header">🌍 الإنتاجية حسب المركز والمحصول</div>', unsafe_allow_html=True)
        center_crop = df.groupby(["center", "crop_type"])["total_yield"].sum().reset_index()
        fig5 = px.sunburst(
            center_crop, path=["center", "crop_type"], values="total_yield",
            color="total_yield",
            color_continuous_scale=["#D4C898","#D4A843","#7AAB3A","#4A7C1F","#2D5016"]
        )
        fig5.update_traces(textinfo="label+percent entry",
                            textfont=dict(family="Cairo", size=12))
        farm_layout(fig5, "إجمالي الإنتاج: مركز → محصول", height=500)
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown('<div class="section-header">🏙️ Treemap – الإنتاج حسب المركز</div>', unsafe_allow_html=True)
        fig_tree = px.treemap(
            center_crop, path=["center","crop_type"], values="total_yield",
            color="total_yield",
            color_continuous_scale=["#D4C898","#D4A843","#4A7C1F","#2D5016"]
        )
        fig_tree.update_traces(textfont=dict(family="Cairo"))
        farm_layout(fig_tree, "", height=400)
        st.plotly_chart(fig_tree, use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="section-header">📐 مصفوفة الارتباط بين المتغيرات الرقمية</div>', unsafe_allow_html=True)
        num_cols = ["area_feddan","yield_per_feddan","total_yield","waste_ratio","waste_tons","temperature","humidity","rainfall"]
        corr     = df[num_cols].corr()
        fig_sns, ax = plt.subplots(figsize=(11, 5))
        fig_sns.patch.set_facecolor("#F0E8D0")
        ax.set_facecolor("#F5EDD8")
        cmap = sns.diverging_palette(30, 145, s=70, l=50, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                    linewidths=0.6, linecolor="#C8B090",
                    annot_kws={"size": 9, "color": "#1C1008", "family": "Cairo"},
                    ax=ax, vmin=-1, vmax=1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color="#4A3520", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#4A3520", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_sns, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – الأثر_البيئي
# ═══════════════════════════════════════════════════════════════════════════════
elif "الأثر" in page or "البيئي" in page:
    st.markdown('<div class="page-title">🌍 الأثر البيئي للمخلفات الزراعية</div>', unsafe_allow_html=True)

    df["co2_estimate"]  = df["waste_tons"] * 0.9
    df["energy_kwh"]    = df["waste_tons"] * 350
    df["compost_tons"]  = df["waste_tons"] * 0.4
    df["biogas_m3"]     = df["waste_tons"] * 120

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("💨 تقدير CO₂ (طن)",         f"{df['co2_estimate'].sum():,.0f}")
    with col2: st.metric("⚡ طاقة حيوية (م kWh)",      f"{df['energy_kwh'].sum()/1e6:.1f}")
    with col3: st.metric("♻️ إمكانية الكمبوست (طن)",   f"{df['compost_tons'].sum():,.0f}")
    with col4: st.metric("🔥 الغاز الحيوي (م م³)",     f"{df['biogas_m3'].sum()/1e6:.2f}")
    with col5: st.metric("🌡️ متوسط الحرارة",           f"{df['temperature'].mean():.1f} °C")

    st.markdown("---")

    co2_total  = df['co2_estimate'].sum()
    energy_gwh = df['energy_kwh'].sum() / 1e6
    compost_total = df['compost_tons'].sum()
    
    st.markdown(
        f'<div class="insight-box danger">🌱 <b>رؤية بيئية مهمة:</b> لو تم حرق كل المخلفات ستنتج <b>{co2_total:,.0f} طن CO₂</b>، '
        f'يعادل انبعاثات آلاف السيارات سنوياً. بالمقابل، تحويلها لطاقة حيوية يولّد <b>{energy_gwh:.1f} مليون kWh</b> '
        f'وتدويرها كمبوست يُنتج <b>{compost_total:,.0f} طن سماد عضوي</b> يُغني التربة ويوفر مدخلات الأسمدة.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    tabs_env = st.tabs(["💨 الانبعاثات", "♻️ إمكانيات الاسترداد", "🌧️ الطقس والبيئة"])
    
    with tabs_env[0]:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-header">💨 انبعاثات CO₂ المقدرة لكل محصول</div>', unsafe_allow_html=True)
            co2_crop = df.groupby("crop_type")["co2_estimate"].sum().reset_index().sort_values("co2_estimate", ascending=False)
            fig = px.bar(co2_crop, x="crop_type", y="co2_estimate",
                            color="co2_estimate",
                            color_continuous_scale=["#D4A843","#C8860A","#A0522D","#A0341A","#6B2A1A"])
            fig.update_layout(coloraxis_showscale=False, xaxis_title="المحصول", yaxis_title="CO₂ (طن)")
            farm_layout(fig, "انبعاثات CO₂ المقدرة لكل محصول")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">🌡️ درجة الحرارة vs المخلفات</div>', unsafe_allow_html=True)
            fig2 = px.scatter(df, x="temperature", y="waste_tons",
                                color="crop_type", color_discrete_map=CROP_COLORS,
                                opacity=0.55, trendline="lowess", size_max=8)
            farm_layout(fig2, "درجة الحرارة مقابل المخلفات")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">🏙️ المؤشرات البيئية حسب المركز</div>', unsafe_allow_html=True)
        env_center = df.groupby("center").agg(
            co2=("co2_estimate","sum"), energy=("energy_kwh","sum"), compost=("compost_tons","sum"),
        ).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="CO₂ (طن)",        x=env_center["center"], y=env_center["co2"],         marker_color="#A0341A"))
        fig3.add_trace(go.Bar(name="طاقة (MWh)",       x=env_center["center"], y=env_center["energy"]/1000, marker_color="#D4A843"))
        fig3.add_trace(go.Bar(name="كمبوست (طن)",      x=env_center["center"], y=env_center["compost"],     marker_color="#4A7C1F"))
        fig3.update_layout(barmode="group", legend_orientation="h")
        farm_layout(fig3, "المؤشرات البيئية حسب المركز", height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with tabs_env[1]:
        st.markdown('<div class="section-header">📡 رادار إمكانية الاسترداد لكل محصول</div>', unsafe_allow_html=True)
        radar_df   = df.groupby("crop_type").agg(
            waste=("waste_tons","sum"), energy=("energy_kwh","mean"),
            co2=("co2_estimate","sum"), compost=("compost_tons","sum"),
            biogas=("biogas_m3","sum")
        ).reset_index()
        categories = ["المخلفات","الطاقة الحيوية","CO₂","الكمبوست","الغاز الحيوي"]
        radar_colors = list(CROP_COLORS.values())

        fig6 = go.Figure()
        for i, (_, row) in enumerate(radar_df.iterrows()):
            vals = [row["waste"], row["energy"]/100, row["co2"], row["compost"], row["biogas"]/100]
            max_v = max(vals) or 1
            vals_n = [v/max_v * 100 for v in vals]
            fig6.add_trace(go.Scatterpolar(
                r=vals_n + [vals_n[0]], theta=categories + [categories[0]],
                name=row["crop_type"], fill="toself", opacity=0.50,
                line=dict(color=radar_colors[i % len(radar_colors)], width=2)
            ))
        farm_layout(fig6, "تحليل رادار إمكانية الاسترداد لكل محصول", height=480)
        fig6.update_layout(polar=dict(
            bgcolor="rgba(240,232,208,0.5)",
            radialaxis=dict(visible=True, range=[0,100], color="#7A6040", gridcolor="rgba(74,124,31,0.2)"),
            angularaxis=dict(color="#4A3520", gridcolor="rgba(74,124,31,0.2)")
        ))
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="section-header">📊 إجمالي إمكانيات الاسترداد</div>', unsafe_allow_html=True)
        recovery = df.groupby("center").agg(
            compost=("compost_tons","sum"), energy_mwh=("energy_kwh", lambda x: x.sum()/1000)
        ).reset_index()
        fig_rec = go.Figure()
        fig_rec.add_trace(go.Bar(name="كمبوست (طن)", x=recovery["center"], y=recovery["compost"],
                                    marker_color="#4A7C1F"))
        fig_rec.add_trace(go.Bar(name="طاقة (MWh)", x=recovery["center"], y=recovery["energy_mwh"],
                                    marker_color="#D4A843"))
        fig_rec.update_layout(barmode="stack", xaxis_title="المركز", yaxis_title="الكمية")
        farm_layout(fig_rec, "إمكانيات الاسترداد: كمبوست + طاقة")
        st.plotly_chart(fig_rec, use_container_width=True)

    with tabs_env[2]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">🌧️ تأثير الأمطار على المخلفات</div>', unsafe_allow_html=True)
            fig4 = px.scatter(df, x="rainfall", y="waste_ratio",
                                color="crop_type", color_discrete_map=CROP_COLORS,
                                size="waste_tons", opacity=0.6,
                                trendline="ols")
            farm_layout(fig4, "الأمطار مقابل نسبة المخلفات")
            st.plotly_chart(fig4, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">💧 الرطوبة vs المخلفات (كثافة)</div>', unsafe_allow_html=True)
            fig5 = px.density_heatmap(df, x="humidity", y="waste_tons",
                                        nbinsx=28, nbinsy=28,
                                        color_continuous_scale=["#F5EDD8","#D4C898","#D4A843","#C8860A","#A0341A"])
            farm_layout(fig5, "توزيع الرطوبة مقابل المخلفات")
            st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 – الحصاد_والتربة
# ═══════════════════════════════════════════════════════════════════════════════
elif "الحصاد" in page or "التربة" in page:
    st.markdown('<div class="page-title">🚜 الحصاد والتربة – تحليل تفصيلي شامل</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🤖 الحصاد الآلي",     f"{(df['harvest_method']=='Mechanical').mean()*100:.1f}%")
    with col2: st.metric("👨‍🌾 الحصاد اليدوي",   f"{(df['harvest_method']=='Manual').mean()*100:.1f}%")
    with col3: st.metric("🪨 أغلب أنواع التربة", df["soil_type"].mode()[0])
    with col4: st.metric("📊 فارق نسبة المخلفات",
                            f"{abs(df[df['harvest_method']=='Mechanical']['waste_ratio'].mean() - df[df['harvest_method']=='Manual']['waste_ratio'].mean())*100:.1f}%",
                            delta="آلي مقابل يدوي")

    st.markdown("---")

    tabs_hs = st.tabs(["🚜 طرق الحصاد", "🪨 أنواع التربة", "🔬 تحليل مقارن", "🌐 ثلاثي الأبعاد"])
    
    with tabs_hs[0]:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-header">🚜 توزيع طرق الحصاد</div>', unsafe_allow_html=True)
            h_counts = df["harvest_method"].value_counts().reset_index()
            h_counts.columns = ["method","count"]
            fig = px.pie(h_counts, names="method", values="count",
                            color="method", color_discrete_map=HARVEST_COLORS, hole=0.50)
            fig.update_traces(textinfo="label+percent+value",
                                textfont=dict(size=13, family="Cairo"),
                                marker=dict(line=dict(color="#F5EDD8", width=2.5)),
                                pull=[0.04, 0.04])
            farm_layout(fig, "طرق الحصاد – التوزيع")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">📊 أداء طرق الحصاد لكل محصول</div>', unsafe_allow_html=True)
            hm_perf = df.groupby(["crop_type","harvest_method"]).agg(
                yield_avg=("yield_per_feddan","mean"),
                waste_avg=("waste_tons","mean")
            ).reset_index()
            fig_hm = px.scatter(hm_perf, x="yield_avg", y="waste_avg",
                                color="harvest_method", symbol="crop_type",
                                color_discrete_map=HARVEST_COLORS,
                                text="crop_type", size_max=12,
                                size=[20]*len(hm_perf))
            fig_hm.update_traces(textposition="top center", textfont=dict(size=10, family="Cairo"))
            farm_layout(fig_hm, "الإنتاجية مقابل المخلفات (حسب طريقة الحصاد)")
            st.plotly_chart(fig_hm, use_container_width=True)
        
        st.markdown('<div class="section-header">🔄 مقارنة تفصيلية: يدوي مقابل آلي</div>', unsafe_allow_html=True)
        mech_df   = df[df["harvest_method"]=="Mechanical"]
        manual_df = df[df["harvest_method"]=="Manual"]
        
        comp_data = pd.DataFrame({
            "المؤشر": ["متوسط الإنتاجية (ط/فدان)","متوسط المخلفات (طن)","متوسط النسبة %","متوسط المساحة (فدان)"],
            "الحصاد الآلي":  [mech_df["yield_per_feddan"].mean(), mech_df["waste_tons"].mean(),
                                mech_df["waste_ratio"].mean()*100, mech_df["area_feddan"].mean()],
            "الحصاد اليدوي": [manual_df["yield_per_feddan"].mean(), manual_df["waste_tons"].mean(),
                                manual_df["waste_ratio"].mean()*100, manual_df["area_feddan"].mean()],
        }).round(2)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="الحصاد الآلي",  x=comp_data["المؤشر"], y=comp_data["الحصاد الآلي"],
                                    marker_color=HARVEST_COLORS["Mechanical"]))
        fig_comp.add_trace(go.Bar(name="الحصاد اليدوي", x=comp_data["المؤشر"], y=comp_data["الحصاد اليدوي"],
                                    marker_color=HARVEST_COLORS["Manual"]))
        fig_comp.update_layout(barmode="group", xaxis_title="", yaxis_title="القيمة")
        farm_layout(fig_comp, "مقارنة مؤشرات الحصاد الآلي مقابل اليدوي", height=360)
        st.plotly_chart(fig_comp, use_container_width=True)

    with tabs_hs[1]:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-header">🪨 توزيع أنواع التربة</div>', unsafe_allow_html=True)
            s_counts = df["soil_type"].value_counts().reset_index()
            s_counts.columns = ["soil","count"]
            fig2 = px.pie(s_counts, names="soil", values="count",
                            color="soil", color_discrete_map=SOIL_COLORS, hole=0.50)
            fig2.update_traces(textinfo="label+percent+value",
                                textfont=dict(size=13, family="Cairo"),
                                marker=dict(line=dict(color="#F5EDD8", width=2.5)))
            farm_layout(fig2, "توزيع أنواع التربة")
            st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">📊 الإنتاجية حسب نوع التربة والمحصول</div>', unsafe_allow_html=True)
            soil_crop = df.groupby(["soil_type","crop_type"])["yield_per_feddan"].mean().reset_index()
            fig_sc = px.bar(soil_crop, x="soil_type", y="yield_per_feddan",
                            color="crop_type", barmode="group",
                            color_discrete_map=CROP_COLORS)
            fig_sc.update_layout(xaxis_title="نوع التربة", yaxis_title="متوسط الإنتاجية (طن/فدان)")
            farm_layout(fig_sc, "الإنتاجية: نوع التربة × المحصول")
            st.plotly_chart(fig_sc, use_container_width=True)
        
        st.markdown('<div class="section-header">🌡️ خريطة حرارية: التربة × المحصول × الإنتاجية</div>', unsafe_allow_html=True)
        pivot_sc = df.pivot_table(index="soil_type", columns="crop_type", values="yield_per_feddan", aggfunc="mean")
        fig_sch = px.imshow(pivot_sc,
                            color_continuous_scale=["#F5EDD8","#D4A843","#7AAB3A","#4A7C1F","#2D5016"],
                            text_auto=".1f", aspect="auto")
        fig_sch.update_traces(textfont=dict(size=13, family="Cairo"))
        farm_layout(fig_sch, "متوسط الإنتاجية (طن/فدان): التربة × المحصول", height=320)
        st.plotly_chart(fig_sch, use_container_width=True)

    with tabs_hs[2]:
        st.markdown('<div class="section-header">📊 تأثير التربة وطريقة الحصاد على الإنتاج والمخلفات</div>', unsafe_allow_html=True)
        cross = df.groupby(["soil_type","harvest_method"]).agg(
            yield_avg=("yield_per_feddan","mean"),
            waste_avg=("waste_tons","mean"),
        ).reset_index()

        col_a, col_b = st.columns(2)
        with col_a:
            fig3 = px.bar(cross, x="soil_type", y="yield_avg",
                            color="harvest_method", barmode="group",
                            color_discrete_map=HARVEST_COLORS)
            fig3.update_layout(xaxis_title="نوع التربة", yaxis_title="متوسط الإنتاجية (طن/فدان)")
            farm_layout(fig3, "الإنتاجية: التربة × طريقة الحصاد")
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            fig4 = px.bar(cross, x="soil_type", y="waste_avg",
                            color="harvest_method", barmode="group",
                            color_discrete_map=HARVEST_COLORS)
            fig4.update_layout(xaxis_title="نوع التربة", yaxis_title="متوسط المخلفات (طن)")
            farm_layout(fig4, "المخلفات: التربة × طريقة الحصاد")
            st.plotly_chart(fig4, use_container_width=True)

        
        

        plt.suptitle("مقارنة المؤشرات الرئيسية: التربة والحصاد",
                    color="#2D5016", fontsize=13, y=1.01, fontweight="bold")
        plt.tight_layout()
        

    with tabs_hs[3]:
        st.markdown('<div class="section-header">🌐 تحليل ثلاثي الأبعاد: المساحة – الإنتاج – المخلفات</div>', unsafe_allow_html=True)
        fig5 = px.scatter_3d(df, x="area_feddan", y="total_yield", z="waste_tons",
                            color="soil_type", symbol="harvest_method",
                            color_discrete_map=SOIL_COLORS, opacity=0.72, size_max=7,
                            hover_data=["crop_type","center","yield_per_feddan"])
        farm_layout(fig5, "", height=550)
        fig5.update_layout(scene=dict(
            bgcolor="rgba(240,232,208,0.5)",
            xaxis=dict(backgroundcolor="rgba(240,232,208,0.5)", gridcolor="#C8B090", color="#4A3520", title="المساحة"),
            yaxis=dict(backgroundcolor="rgba(240,232,208,0.5)", gridcolor="#C8B090", color="#4A3520", title="الإنتاج"),
            zaxis=dict(backgroundcolor="rgba(240,232,208,0.5)", gridcolor="#C8B090", color="#4A3520", title="المخلفات"),
        ))
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 – الخريطة_الجغرافية
# ═══════════════════════════════════════════════════════════════════════════════
elif "الخريطة" in page or "خريطة" in page:
    st.markdown('<div class="page-title">🗺️ الخريطة الجغرافية – توزيع المزارع في الشرقية</div>', unsafe_allow_html=True)

    center_coords = {
        "Zagazig":       (30.5965, 31.5020),
        "Belbeis":       (30.4183, 31.5615),
        "Abu Hammad":    (30.5237, 31.6817),
        "Minya El Qamh": (30.5180, 31.3487),
        "Husseiniya":    (30.7700, 31.8800),
        "Faqous":        (30.7318, 31.8052),
    }

    df["lat"] = df["center"].map(lambda c: center_coords.get(c, (30.5965, 31.5020))[0])
    df["lon"] = df["center"].map(lambda c: center_coords.get(c, (30.5965, 31.5020))[1])
    np.random.seed(42)
    df["lat"] = df["lat"] + np.random.uniform(-0.04, 0.04, len(df))
    df["lon"] = df["lon"] + np.random.uniform(-0.04, 0.04, len(df))

    col1, col2, col3, col4 = st.columns(4)
    with col1: map_metric = st.selectbox("🎨 متغير اللون",      ["waste_tons","total_yield","waste_ratio","area_feddan","temperature","humidity"])
    with col2: map_size   = st.selectbox("📏 متغير الحجم",      ["waste_tons","total_yield","area_feddan","yield_per_feddan"])
    with col3: map_style  = st.selectbox("🗺️ نمط الخريطة",     ["carto-positron","open-street-map","carto-darkmatter"])
    with col4: map_crop   = st.multiselect("🌾 تصفية المحصول",  df["crop_type"].unique(), default=list(df["crop_type"].unique()))

    df_map = df[df["crop_type"].isin(map_crop)] if map_crop else df

    st.markdown("---")
    st.markdown('<div class="section-header">📍 خريطة توزيع المزارع التفصيلية</div>', unsafe_allow_html=True)

    fig_map = px.scatter_mapbox(
        df_map, lat="lat", lon="lon",
        color=map_metric, size=map_size,
        hover_name="center",
        hover_data={"crop_type":True,"harvest_method":True,"soil_type":True,
                    "waste_tons":":.1f","total_yield":":.1f","area_feddan":":.1f",
                    "lat":False,"lon":False},
        color_continuous_scale=["#D4C898","#D4A843","#C8860A","#A0522D","#6B2A1A"],
        size_max=18, zoom=8.5,
        center={"lat":30.60,"lon":31.60},
        mapbox_style=map_style, opacity=0.78,
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=560,
        margin=dict(l=0,r=0,t=0,b=0),
        coloraxis_colorbar=dict(
            title=map_metric,
            tickfont=dict(color="#4A3520", family="Cairo"),
            title_font=dict(color="#4A3520", family="Cairo")
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🟡 خريطة الفقاعات – إجمالي المخلفات لكل مركز</div>', unsafe_allow_html=True)

    center_agg_map = df.groupby("center").agg(
        waste=("waste_tons","sum"), yield_=("total_yield","sum"),
        farms=("farm_id","count"), area=("area_feddan","sum")
    ).reset_index()
    center_agg_map["lat"] = center_agg_map["center"].map(lambda c: center_coords.get(c,(30.6,31.6))[0])
    center_agg_map["lon"] = center_agg_map["center"].map(lambda c: center_coords.get(c,(30.6,31.6))[1])

    fig_map2 = px.scatter_mapbox(
        center_agg_map, lat="lat", lon="lon",
        size="waste", color="waste", hover_name="center",
        hover_data={"farms":True,"yield_":":.0f","waste":":.0f","area":":.0f","lat":False,"lon":False},
        color_continuous_scale=["#D4A843","#C8860A","#A0341A"],
        size_max=55, zoom=8.5,
        center={"lat":30.60,"lon":31.60},
        mapbox_style=map_style, text="center",
    )
    fig_map2.update_traces(textposition="top center",
                            textfont=dict(size=12, color="#1C1008", family="Cairo"))
    fig_map2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=480,
        margin=dict(l=0,r=0,t=0,b=0),
        coloraxis_colorbar=dict(title="المخلفات (طن)", tickfont=dict(color="#4A3520")),
    )
    st.plotly_chart(fig_map2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📋 ملخص المراكز الجغرافي</div>', unsafe_allow_html=True)

    summary_map = df.groupby("center").agg(
        عدد_المزارع=("farm_id","count"),
        اجمالي_المساحة=("area_feddan","sum"),
        اجمالي_الانتاج=("total_yield","sum"),
        اجمالي_المخلفات=("waste_tons","sum"),
        متوسط_النسبة=("waste_ratio",lambda x: x.mean()*100),
        اكثر_محصول=("crop_type", lambda x: x.mode()[0])
    ).round(2).reset_index()
    summary_map.columns = ["المركز","عدد المزارع","المساحة (فدان)","الإنتاج (طن)","المخلفات (طن)","نسبة المخلفات %","أكثر محصول"]
    st.dataframe(summary_map, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 – نموذج_التنبؤ
# ═══════════════════════════════════════════════════════════════════════════════
elif "نموذج" in page or "التنبؤ" in page:
    st.markdown('<div class="page-title">🤖 نموذج الذكاء الاصطناعي – التنبؤ بكميات المخلفات</div>', unsafe_allow_html=True)

    @st.cache_data
    def prepare_ml_data(dataframe):
        df_ml = dataframe.copy()
        le_crop    = LabelEncoder(); le_harvest = LabelEncoder()
        le_soil    = LabelEncoder(); le_center  = LabelEncoder()
        df_ml["crop_enc"]    = le_crop.fit_transform(df_ml["crop_type"])
        df_ml["harvest_enc"] = le_harvest.fit_transform(df_ml["harvest_method"])
        df_ml["soil_enc"]    = le_soil.fit_transform(df_ml["soil_type"])
        df_ml["center_enc"]  = le_center.fit_transform(df_ml["center"])
        return df_ml, le_crop, le_harvest, le_soil, le_center

    df_ml, le_crop, le_harvest, le_soil, le_center = prepare_ml_data(df_raw)

    features = [
        "area_feddan","yield_per_feddan","total_yield","waste_ratio",
        "temperature","humidity","rainfall",
        "crop_enc","harvest_enc","soil_enc","center_enc"
    ]
    target = "waste_tons"
    X = df_ml[features]; y = df_ml[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_rf(X_tr, y_tr):
        rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                    min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr); return rf

    @st.cache_resource
    def train_gbm(X_tr, y_tr):
        gb = GradientBoostingRegressor(n_estimators=150, max_depth=6,
                                        learning_rate=0.1, random_state=42)
        gb.fit(X_tr, y_tr); return gb

    model_rf  = train_rf(X_train, y_train)
    model_gbm = train_gbm(X_train, y_train)

    y_pred_rf  = model_rf.predict(X_test)
    y_pred_gbm = model_gbm.predict(X_test)

    r2_rf    = r2_score(y_test, y_pred_rf);   mae_rf  = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf  = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_gbm   = r2_score(y_test, y_pred_gbm);  mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
    rmse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))

    model  = model_rf
    y_pred = y_pred_rf


    st.markdown("---")
    st.markdown('<div class="section-header">🔮 تنبأ بكمية المخلفات لمزرعتك</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        p_center  = st.selectbox("📍 المركز",        df_raw["center"].unique())
        p_crop    = st.selectbox("🌾 نوع المحصول",   df_raw["crop_type"].unique())
    with col_f2:
        p_harvest = st.selectbox("🚜 طريقة الحصاد",  df_raw["harvest_method"].unique())
        p_soil    = st.selectbox("🪨 نوع التربة",     df_raw["soil_type"].unique())
        price_per_ton = st.slider("💰 سعر المخلفات/طن (جنيه)", 100, 1000, 400, 50)
    with col_f3:
        p_area    = st.number_input("📐 المساحة (فدان)",  min_value=1.0,  max_value=20.0, value=8.0,  step=0.5)
        p_ypf     = st.number_input("📊 الإنتاجية/فدان",  min_value=1.0,  max_value=50.0, value=10.0, step=0.5)
        p_total   = p_area * p_ypf
        st.metric("🌾 الإنتاج المتوقع", f"{p_total:.1f} طن")
    with col_f4:
        p_ratio    = st.slider("⚠️ نسبة المخلفات",   0.05, 0.60, 0.25, 0.01, format="%.2f")
        p_temp     = st.slider("🌡️ الحرارة (°C)",    15.0, 45.0, 30.0, 0.5)
        p_humidity = st.slider("💧 الرطوبة (%)",      30.0, 80.0, 55.0, 1.0)
        p_rain     = st.slider("🌧️ الأمطار (mm)",      0.0, 10.0,  5.0, 0.1)

    st.markdown("---")
    if st.button("🔮 تنبؤ بكمية المخلفات الآن"):
        input_data = pd.DataFrame([[
            p_area, p_ypf, p_total, p_ratio,
            p_temp, p_humidity, p_rain,
            le_crop.transform([p_crop])[0],
            le_harvest.transform([p_harvest])[0],
            le_soil.transform([p_soil])[0],
            le_center.transform([p_center])[0]
        ]], columns=features)

        pred_rf_val  = model_rf.predict(input_data)[0]
        pred_gbm_val = model_gbm.predict(input_data)[0]
        prediction   = pred_rf_val

        st.markdown("---")
        res1, res2, res3, res4 = st.columns(4)
        with res1: st.metric("كمية المخلفات المتنباء بها",    f"{pred_rf_val:.2f} طن")
        
        with res2: st.metric(" متوسط المخلفات للمزارع المشابهة", f"{df_raw['waste_tons'].mean():.2f} طن")
        with res3:
            diff = prediction - df_raw['waste_tons'].mean()
            st.metric("📈 الفرق عن المتوسط", f"{diff:+.2f} طن",
                    delta="أعلى" if diff > 0 else "أقل")
        with res4:
            cost = prediction * price_per_ton
            st.metric("💰 التكلفة المتوقعة", f"{round(cost):,} جنيها")

        

        q75 = df_raw["waste_tons"].quantile(0.75)
        q25 = df_raw["waste_tons"].quantile(0.25)
        if prediction > q75:
            advice = (f"⚠️ <b>كمية مرتفعة ({prediction:.1f} طن)</b> — يُنصح بمراجعة طريقة الحصاد "
                        f"والتفكير في تحويل المخلفات لكمبوست أو طاقة حيوية. "
                        f"الحصاد الآلي قد يقلل المخلفات بنسبة تصل إلى 20%.")
            cls = "danger"
        elif prediction > q25:
            advice = (f"ℹ️ <b>كمية متوسطة ({prediction:.1f} طن)</b> — يمكن تحسينها بتطوير التخزين "
                        f"ومعالجة المخلفات، واستخدام تقنيات الحصاد الموفِّرة للمحصول.")
            cls = ""
        else:
            advice = (f"✅ <b>كمية منخفضة ({prediction:.1f} طن)</b> — أداء ممتاز! "
                        f"يمكن الاستفادة من هذه المخلفات في إنتاج سماد عضوي عالي الجودة لتحسين خصوبة التربة.")
            cls = "success"
        st.markdown(f'<div class="insight-box {cls}">{advice}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 – توصيات_المزارع
# ═══════════════════════════════════════════════════════════════════════════════
elif "توصيات" in page:
    st.markdown('<div class="page-title">💡 توصيات عملية للمزارع – دليل إدارة المخلفات</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="insight-box success">🌱 <b>مرحباً بك في دليل التوصيات الزراعية</b> — هنا ستجد نصائح عملية مبنية على تحليل بيانات مزارع الشرقية لتقليل الهدر وزيادة الاستفادة من مخلفاتك الزراعية.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    st.markdown('<div class="section-header">🧮 حاسبة الاستفادة من مخلفاتك</div>', unsafe_allow_html=True)
    
    calc_col1, calc_col2 = st.columns([1, 1])
    with calc_col1:
        user_waste = st.number_input("أدخل كمية مخلفاتك (طن)", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
        user_crop  = st.selectbox("نوع محصولك", df["crop_type"].unique(), key="rec_crop")
    
    with calc_col2:
        compost_yield  = user_waste * 0.40
        energy_kwh     = user_waste * 350
        biogas_m3      = user_waste * 120
        co2_saved      = user_waste * 0.9
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2D5016,#1E3510);border-radius:14px;padding:20px;color:#F5EDD8;">
            <div style="font-size:1.0rem;font-weight:800;color:#D4A843;margin-bottom:14px;text-align:center;">
                ♻️ إمكانيات الاستفادة من {user_waste:.1f} طن مخلفات
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div style="text-align:center;background:rgba(255,255,255,0.08);padding:10px;border-radius:10px;">
                    <div style="font-size:1.5rem;">🌱</div>
                    <div style="color:#D4A843;font-size:0.8rem;font-weight:700;">سماد عضوي</div>
                    <div style="font-size:1.2rem;font-weight:900;">{compost_yield:.1f} طن</div>
                </div>
                <div style="text-align:center;background:rgba(255,255,255,0.08);padding:10px;border-radius:10px;">
                    <div style="font-size:1.5rem;">⚡</div>
                    <div style="color:#D4A843;font-size:0.8rem;font-weight:700;">طاقة حيوية</div>
                    <div style="font-size:1.2rem;font-weight:900;">{energy_kwh:,.0f} kWh</div>
                </div>
                <div style="text-align:center;background:rgba(255,255,255,0.08);padding:10px;border-radius:10px;">
                    <div style="font-size:1.5rem;">🔥</div>
                    <div style="color:#D4A843;font-size:0.8rem;font-weight:700;">غاز حيوي</div>
                    <div style="font-size:1.2rem;font-weight:900;">{biogas_m3:,.0f} م³</div>
                </div>
                <div style="text-align:center;background:rgba(255,255,255,0.08);padding:10px;border-radius:10px;">
                    <div style="font-size:1.5rem;">🌍</div>
                    <div style="color:#D4A843;font-size:0.8rem;font-weight:700;">CO₂ يمكن توفيره</div>
                    <div style="font-size:1.2rem;font-weight:900;">{co2_saved:.1f} طن</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown('<div class="section-header">🌾 توصيات مخصصة لمحصولك</div>', unsafe_allow_html=True)
    
    crop_tips = {
        "Wheat": {
            "icon": "🌾",
            "tips": [
                ("🚜 الحصاد الآلي", "استخدم الحصادات الآلية الحديثة لتقليل الهدر بنسبة 15-20%"),
                ("📦 قش القمح", "قش القمح له قيمة عالية: استخدمه علفاً للماشية أو سماداً عضوياً"),
                ("🔄 التدوير", "نسب تدوير قش القمح يمكن أن تصل 85% مع التقنيات الحديثة"),
                ("⏰ التوقيت", "احرص على حصاد القمح في الوقت المناسب لتجنب الهدر"),
            ]
        },
        "Rice": {
            "icon": "🍚",
            "tips": [
                ("🔥 التبن", "لا تحرق تبن الأرز – هذا يضر البيئة ويهدر موارد قيمة"),
                ("🌱 السماد", "تبن الأرز ممتاز لصنع الكمبوست وتحسين خصوبة التربة"),
                ("💧 الغمر المتحكم", "استخدم نظام الغمر المتحكم لتقليل مخلفات الجذور"),
                ("🏭 الطاقة", "تبن الأرز يمكن تحويله لطاقة حرارية بكفاءة عالية"),
            ]
        },
        "Maize": {
            "icon": "🌽",
            "tips": [
                ("🐄 الأعلاف", "قش الذرة مصدر ممتاز للأعلاف الخشنة للماشية"),
                ("🔬 السيلاج", "حول مخلفات الذرة لسيلاج عالي الجودة للتخزين الطويل"),
                ("🌱 التغطية", "استخدم بقايا الذرة كغطاء للتربة لتقليل التبخر"),
                ("⚡ الغاز", "مخلفات الذرة تعطي إنتاجاً ممتازاً من الغاز الحيوي"),
            ]
        },
        "Sugar Beet": {
            "icon": "🌿",
            "tips": [
                ("🐄 الأعلاف", "أوراق الشمندر السكري علف ممتاز للماشية"),
                ("♻️ اللب", "لب الشمندر بعد استخلاص السكر يمكن تجفيفه وبيعه"),
                ("🌍 التربة", "بقايا الشمندر تُحسّن بنية التربة الطينية"),
                ("📊 التتبع", "راقب نسبة الهدر في عمليات نقل وتخزين المحصول"),
            ]
        },
        "Potato": {
            "icon": "🥔",
            "tips": [
                ("❄️ التخزين", "التخزين السليم في بيئة باردة يقلل الهدر بنسبة 30%"),
                ("🌱 البقايا", "بقايا البطاطس وأوراقها ممتازة لإنتاج الكمبوست"),
                ("🔍 التفتيش", "افحص المحصول قبل التخزين لعزل المصابة وتقليل الهدر"),
                ("💊 الفطريات", "استخدام المعاملات الوقائية يقلل الخسائر الفطرية"),
            ]
        }
    }

    selected_crop_tips = crop_tips.get(user_crop, crop_tips["Wheat"])
    
    tip_cols = st.columns(4)
    for i, (title, desc) in enumerate(selected_crop_tips["tips"]):
        with tip_cols[i]:
            st.markdown(f"""
            <div class="tip-card">
                <div class="tip-icon">{title.split()[0]}</div>
                <h4>{" ".join(title.split()[1:])}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown('<div class="section-header">📋 أفضل الممارسات العامة لإدارة المخلفات</div>', unsafe_allow_html=True)

    practices = [
        ("📅 التخطيط المسبق", "ضع خطة واضحة لإدارة المخلفات قبل موسم الحصاد بشهرين على الأقل"),
        ("📊 قياس وتوثيق", "سجّل كميات المخلفات لكل محصول لتحسين التخطيط المستقبلي"),
        ("🤝 التعاون الجماعي", "تعاون مع المزارعين المجاورين لإنشاء مشاريع تدوير مشتركة"),
        ("🏭 الشركاء التجاريين", "ابحث عن شركات تشتري المخلفات الزراعية كمادة خام للصناعة"),
        ("💡 التقنيات الحديثة", "تابع التقنيات الجديدة في تحويل المخلفات الزراعية للمنتجات ذات القيمة"),
        ("🌱 الزراعة العضوية", "الكمبوست الناتج من مخلفاتك يمكن استخدامه لتحويل مزرعتك للزراعة العضوية"),
    ]

    for i in range(0, len(practices), 2):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if i < len(practices):
                title, desc = practices[i]
                st.markdown(f'<div class="insight-box"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
        with col_p2:
            if i+1 < len(practices):
                title, desc = practices[i+1]
                st.markdown(f'<div class="insight-box success"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown('<div class="section-header">🎯 أهداف تقليل المخلفات – مؤشرات الأداء</div>', unsafe_allow_html=True)
    
    targets = {
        "القمح": 15, "الأرز": 18, "الذرة": 14, "الشمندر": 12, "البطاطس": 20
    }
    
    target_cols = st.columns(5)
    for i, (crop_ar, target_pct) in enumerate(targets.items()):
        crop_en = list(CROP_COLORS.keys())[i]
        actual  = df[df["crop_type"]==crop_en]["waste_ratio"].mean() * 100
        delta_val = actual - target_pct
        color = "#A0341A" if delta_val > 0 else "#4A7C1F"
        
        with target_cols[i]:
            st.markdown(f"""
            <div style="background:#F0E8D0;border:1px solid #C8B090;border-radius:12px;padding:14px;text-align:center;">
                <div style="font-size:1.5rem;">{["🌾","🍚","🌽","🌿","🥔"][i]}</div>
                <div style="color:#2D5016;font-weight:800;font-size:0.9rem;">{crop_ar}</div>
                <div style="color:{color};font-size:1.3rem;font-weight:900;">{actual:.1f}%</div>
                <div style="color:#7A6040;font-size:0.75rem;">الهدف: {target_pct}%</div>
                <div style="color:{color};font-size:0.78rem;font-weight:700;">
                    {"▲" if delta_val > 0 else "▼"} {abs(delta_val):.1f}% عن الهدف
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""<div style='text-align:center;color:#7A6040;font-size:0.82rem;padding:12px;
    background:linear-gradient(90deg,rgba(45,80,22,0.04),rgba(212,168,67,0.06),rgba(45,80,22,0.04));
    border-radius:10px;border:1px solid rgba(200,176,144,0.3);'>
    🌿 لوحة إدارة المخلفات الزراعية – محافظة الشرقية |
    إجمالي السجلات: <b style="color:#4A7C1F;">{len(df_raw):,}</b> مزرعة |
    جميع الحقوق محفوظة © 2025
    </div>""",
    unsafe_allow_html=True
)