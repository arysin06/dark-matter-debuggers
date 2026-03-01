import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from io import StringIO


API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="Stellar Verification Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Exoplanet Verification Dashboard - Stellar Analytics Phase 1"
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
        :root {
            --primary-color: #6B5FFF;
            --secondary-color: #FF6B6B;
            --success-color: #2ECC71;
            --warning-color: #F39C12;
            --danger-color: #E74C3C;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
            font-weight: bold;
        }
        
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 0;
        }
        
        .section-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            border-left: 5px solid #FF6B6B;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 0.5rem;
        }
        
        .status-confirmed {
            background-color: #2ECC71;
            color: white;
        }
        
        .status-false-positive {
            background-color: #E74C3C;
            color: white;
        }
        
        .status-candidate {
            background-color: #F39C12;
            color: white;
        }
        
        .info-box {
            background: #E8F4F8;
            border-left: 4px solid #667eea;
            padding: 1.25rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .success-box {
            background: #E8F8F5;
            border-left: 4px solid #2ECC71;
            padding: 1.25rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .error-box {
            background: #FADBD8;
            border-left: 4px solid #E74C3C;
            padding: 1.25rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .form-section {
            background: #F8F9FA;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            border: 1px solid #E0E0E0;
        }
        
        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        .dataframe-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
        }
        
        [data-testid="stSidebar"] .css-1d391kg {
            padding: 2rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
    <div class="main-header">
        <h1>🌟 Stellar Verification Dashboard</h1>
        <p>Interactive Exoplanet Analysis & Prediction Platform</p>
    </div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path="supernova_dataset.csv"):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    return df


df = load_data()


def mission_brief():
    st.markdown('<div class="section-header"><h2>🚀 Mission Brief</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        In the year 2236, humanity's **Stellar Verification Program** scans exoplanet candidates across the galaxy. 
        This dashboard helps mission scientists inspect transit signals, explore datasets, and verify whether signals are 
        **CONFIRMED** exoplanets or **FALSE POSITIVES**.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Quick Start Guide")
        with st.expander("🔍 How to use this dashboard:", expanded=False):
            st.markdown("""
            1. **Data Insights**: Explore dataset distributions and relationships
            2. **Model Prediction**: Make single predictions using the interactive form
            3. **Batch Predictions**: Upload CSV files for bulk predictions
            4. **System Architecture**: View model performance metrics and system design
            """)
    
    with col2:
        st.markdown("### 📊 Key Statistics")
        metrics_row1, metrics_row2 = st.columns(2)
        with metrics_row1:
            st.metric("🎯 Classification F1", "0.9113")
        with metrics_row2:
            st.metric("📐 Regression RMSE", "0.654 R⊕")



def data_insights():
    st.markdown('<div class="section-header"><h2>📊 Data Insights</h2></div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("⚠️ Dataset not found in workspace. Place `supernova_dataset.csv` beside this app.")
        return

    # Dataset Overview Section
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Rows</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        confirmed_count = len(df[df['koi_disposition'] == 'CONFIRMED']) if 'koi_disposition' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confirmed</div>
            <div class="metric-value">{confirmed_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        false_positive_count = len(df[df['koi_disposition'] == 'FALSE POSITIVE']) if 'koi_disposition' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">False Positives</div>
            <div class="metric-value">{false_positive_count}</div>
        </div>
        """, unsafe_allow_html=True)

    # Data Exploration Section
    st.markdown("### 🔍 Feature Distributions")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 koi_period", "📉 koi_depth", "🌡️ st_teff", "📊 Data Preview"])

    with tab1:
        fig = px.histogram(df, x='koi_period', nbins=60, marginal='box', title='KOI Period Distribution',
                          color_discrete_sequence=['#667eea'], opacity=0.7)
        fig.update_layout(hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Mean: {df['koi_period'].mean():.2f} | Median: {df['koi_period'].median():.2f} | Std: {df['koi_period'].std():.2f}")

    with tab2:
        fig = px.histogram(df, x='koi_depth', nbins=60, title='KOI Depth Distribution',
                          color_discrete_sequence=['#764ba2'], opacity=0.7)
        fig.update_layout(hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Mean: {df['koi_depth'].mean():.2f} | Median: {df['koi_depth'].median():.2f}")

    with tab3:
        fig = px.histogram(df, x='st_teff', nbins=60, title='Stellar Effective Temperature Distribution',
                          color_discrete_sequence=['#FF6B6B'], opacity=0.7)
        fig.update_layout(hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Mean: {df['st_teff'].mean():.0f} K | Range: {df['st_teff'].min():.0f} - {df['st_teff'].max():.0f} K")

    with tab4:
        st.markdown("**Data Sample (first 50 rows)**")
        st.dataframe(df.head(50), use_container_width=True, height=400)

    # Relationships Section
    st.markdown("### 🔗 Feature Relationships")
    r1, r2 = st.columns(2)
    
    with r1:
        if 'koi_prad' in df.columns and 'koi_period' in df.columns:
            fig = px.scatter(df, x='koi_period', y='koi_prad', color='koi_disposition',
                           hover_data=['kepid'], title='Planetary Radius vs Orbital Period',
                           color_discrete_map={
                               'CONFIRMED': '#2ECC71',
                               'FALSE POSITIVE': '#E74C3C',
                               'CANDIDATE': '#F39C12'
                           })
            fig.update_layout(hovermode='closest', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    with r2:
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, color_continuous_scale='RdBu_r', title='Feature Correlation Matrix',
                       zmin=-1, zmax=1)
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    # Class Distribution
    st.markdown("### 🎯 Class Distribution")
    if 'koi_disposition' in df.columns:
        col1, col2 = st.columns([1, 1])
        with col1:
            disposition_counts = df['koi_disposition'].value_counts()
            fig = px.pie(values=disposition_counts.values, names=disposition_counts.index,
                        title='Signal Classification Distribution',
                        color_discrete_map={
                            'CONFIRMED': '#2ECC71',
                            'FALSE POSITIVE': '#E74C3C',
                            'CANDIDATE': '#F39C12'
                        })
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Distribution Summary**")
            for label, count in disposition_counts.items():
                pct = 100 * count / len(df)
                st.write(f"**{label}**: {count} ({pct:.1f}%)")


def prediction_form():
    st.markdown('<div class="section-header"><h2>🔮 Single Signal Verification</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Enter transit and stellar attributes to predict whether a signal is CONFIRMED or a FALSE POSITIVE. 
    The model analyzes 14 key features and provides confidence scores.
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.info("ℹ️ If you provide no dataset, default values will be used where possible.")

    fields = [
        'kepid','koi_disposition','koi_period','koi_duration','koi_depth','koi_impact','koi_model_snr',
        'koi_num_transits','koi_ror','koi_prad','st_teff','st_logg','st_met','st_mass','st_radius','st_dens',
        'teff_err1','teff_err2','logg_err1','logg_err2','feh_err1','feh_err2','mass_err1','mass_err2','radius_err1','radius_err2'
    ]

    defaults = {}
    if not df.empty:
        for f in fields:
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                defaults[f] = float(df[f].median())
            elif f in df.columns:
                defaults[f] = str(df[f].mode().iat[0]) if not df[f].mode().empty else ''
            else:
                defaults[f] = 0 if f != 'koi_disposition' else 'CANDIDATE'
    else:
        for f in fields:
            defaults[f] = 0 if f != 'koi_disposition' else 'CANDIDATE'

    with st.form(key='predict_form'):
        # Organize form into logical sections
        st.markdown("### 🌍 Transit Properties")
        col1, col2 = st.columns(2)
        transit_fields = ['koi_period', 'koi_duration', 'koi_depth', 'koi_impact', 'koi_model_snr', 'koi_num_transits', 'koi_ror', 'koi_prad']
        
        inputs = {}
        for i, f in enumerate(transit_fields):
            col = col1 if i % 2 == 0 else col2
            with col:
                val = defaults.get(f, 0)
                inputs[f] = st.number_input(f"**{f}**", value=float(val), step=0.01, 
                                           help=f"Value for {f}")

        st.markdown("### ⭐ Stellar Properties")
        col1, col2 = st.columns(2)
        stellar_fields = ['st_teff', 'st_logg', 'st_met', 'st_mass', 'st_radius', 'st_dens']
        
        for i, f in enumerate(stellar_fields):
            col = col1 if i % 2 == 0 else col2
            with col:
                val = defaults.get(f, 0)
                inputs[f] = st.number_input(f"**{f}**", value=float(val), step=0.01)

        st.markdown("### 🔬 Measurement Errors")
        col1, col2, col3 = st.columns(3)
        error_fields = ['teff_err1', 'teff_err2', 'logg_err1', 'logg_err2', 'feh_err1', 
                       'feh_err2', 'mass_err1', 'mass_err2', 'radius_err1', 'radius_err2']
        
        for i, f in enumerate(error_fields):
            col = [col1, col2, col3][i % 3]
            with col:
                val = defaults.get(f, 0)
                inputs[f] = st.number_input(f"**{f}**", value=float(val), step=0.001)

        st.markdown("### 📝 Metadata")
        col1, col2 = st.columns(2)
        with col1:
            inputs['kepid'] = st.number_input("**kepid**", value=int(defaults.get('kepid', 0)))
        with col2:
            inputs['koi_disposition'] = st.selectbox('**koi_disposition**', 
                                                     options=['CANDIDATE','CONFIRMED','FALSE POSITIVE'], 
                                                     index=0)

        submit = st.form_submit_button("🚀 Verify Signal", use_container_width=True)

    if submit:
        with st.spinner("🔄 Analyzing signal..."):
            payload = {k: v for k, v in inputs.items()}

            try:
                r = requests.post(API_URL, json=payload, timeout=10)
                r.raise_for_status()
                resp = r.json()
            except Exception as e:
                st.error(f"❌ Prediction request failed: {e}")
                return

            if resp.get('status') != 'success':
                st.error(f"❌ Model returned error: {resp}")
                return

            data = resp.get('data', {})
            disposition = data.get('disposition_prediction', 'UNKNOWN')
            prob = data.get('disposition_probability', None)
            pred_radius = data.get('predicted_radius_earth', None)

            # Display results with styling
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns([1.2, 1, 1])
            
            if disposition == 'CONFIRMED':
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Signal Verified: CONFIRMED</h3>
                    <p>This signal shows strong characteristics of a real exoplanet.</p>
                </div>
                """, unsafe_allow_html=True)
            elif disposition == 'FALSE POSITIVE':
                st.markdown(f"""
                <div class="error-box">
                    <h3>⚠️ Alert: FALSE POSITIVE DETECTED</h3>
                    <p>This signal does not exhibit exoplanet characteristics.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <h3>ℹ️ Classification: {disposition}</h3>
                </div>
                """, unsafe_allow_html=True)

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Disposition</div>
                    <div class="metric-value">{disposition}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                if prob is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with metric_col3:
                if pred_radius is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Estimated Radius</div>
                        <div class="metric-value">{pred_radius:.3f}R⊕</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📋 Detailed Response"):
                st.json(resp)


def model_performance_and_architecture():
    st.markdown('<div class="section-header"><h2>⚙️ Model Performance & Architecture</h2></div>', unsafe_allow_html=True)

    f1_score_val = 0.9113
    rmse_val = 0.6536
    roc_auc_val = 0.9834
    r2_val = 0.9601

    # Performance Metrics
    st.markdown("### 📈 Model Performance Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Classification F1</div>
            <div class="metric-value">{f1_score_val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ROC-AUC Score</div>
            <div class="metric-value">{roc_auc_val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Regression RMSE</div>
            <div class="metric-value">{rmse_val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{r2_val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Training Details
    st.markdown("### 📚 Training Details")
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        st.markdown("""
        **Task A: Signal Classification**
        - Target: koi_disposition (CONFIRMED vs FALSE POSITIVE)
        - Training Samples: 6,580
        - Test Samples: 1,517
        - Model: Gradient Boosting + SMOTE
        - Features Selected: 14
        """)
    
    with train_col2:
        st.markdown("""
        **Task B: Radius Regression**
        - Target: koi_prad (Planetary Radius)
        - Training Samples: 2,195
        - Test Samples: 549
        - Model: Gradient Boosting
        - Features Selected: 14
        """)

    # System Architecture
    st.markdown("### 🏗️ System Architecture")
    
    arch_col1, arch_col2 = st.columns([1.5, 1])
    
    with arch_col1:
        st.markdown("""
        <div class="info-box">
        <h4>Data Flow</h4>
        <ol>
            <li><strong>Frontend</strong>: Streamlit interactive dashboard</li>
            <li><strong>API Layer</strong>: Flask REST API (/predict endpoint)</li>
            <li><strong>Models</strong>: Serialized scikit-learn pipelines (.joblib)</li>
            <li><strong>Processing</strong>: Feature scaling, encoding, inference</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_col2:
        st.markdown("""
        **Key Components**
        - 🔄 Request handling
        - 📊 Data preprocessing
        - 🤖 Model inference
        - 📈 Confidence scoring
        """)

    # Process Flow Visualization
    st.markdown("### 📊 Process Flow")
    
    fig = go.Figure()
    
    fig.add_trace(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["User Input", "Data Validation", "Feature Engineering", 
                   "Task A (Classification)", "Task B (Regression)", "Confidence Scoring", "Response"],
            color=["#667eea", "#764ba2", "#FF6B6B", "#2ECC71", "#F39C12", "#667eea", "#764ba2"]
        ),
        link=dict(
            source=[0, 1, 2, 2, 3, 4, 5],
            target=[1, 2, 3, 4, 5, 5, 6],
            value=[1, 1, 1, 1, 1, 1, 1]
        )
    ))
    
    fig.update_layout(title="Signal Processing Pipeline", font=dict(size=11),
                     template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance Info
    st.markdown("### 🎯 Key Features")
    
    feat_col1, feat_col2 = st.columns(2)
    
    with feat_col1:
        st.markdown("""
        **Transit Properties:**
        - koi_period (Orbital Period)
        - koi_duration (Transit Duration)
        - koi_depth (Transit Depth)
        - koi_model_snr (Signal-to-Noise Ratio)
        - koi_num_transits (Number of Transits)
        - koi_impact (Impact Parameter)
        - koi_ror (Radius of Planet/Star Ratio)
        """)
    
    with feat_col2:
        st.markdown("""
        **Stellar Properties:**
        - st_teff (Effective Temperature)
        - st_logg (Surface Gravity)
        - st_met (Metallicity)
        - st_mass (Stellar Mass)
        - st_radius (Stellar Radius)
        - st_dens (Stellar Density)
        """)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>✨ Advanced Features:</strong> The pipeline includes relative error measurements, 
    stellar density calculations, and signal strength interactions to maximize predictive power.
    </div>
    """, unsafe_allow_html=True)


def batch_upload_section():
    st.markdown('<div class="section-header"><h2>📦 Batch Predictions</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Upload a CSV file with the same schema as the training dataset to process multiple candidates at once.
    The system will return predictions for all signals with confidence scores and radius estimates.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("📤 Upload CSV file for batch prediction", type=['csv'],
                                   help="File must contain the same columns as supernova_dataset.csv")
    
    with col2:
        st.markdown("**📋 Expected Format**")
        st.caption("26 columns including transit & stellar properties")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Could not read uploaded CSV: {e}")
            return

        st.success(f"✅ Batch loaded: {len(batch_df)} rows")

        # Configuration Section
        st.markdown("### ⚙️ Batch Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            max_rows = st.slider('Max rows to predict', 10, min(500, max(10, len(batch_df))), 
                                value=min(100, len(batch_df)),
                                help="Limit to avoid long processing times")
        
        with config_col2:
            preview_enabled = st.checkbox("Show data preview", value=True)

        batch_df = batch_df.head(max_rows).copy()

        if preview_enabled:
            st.markdown("**Data Preview**")
            st.dataframe(batch_df.head(10), use_container_width=True)

        # Batch Prediction Processing
        if st.button("🚀 Start Batch Prediction", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in batch_df.iterrows():
                payload = {c: (row[c] if c in row else None) for c in batch_df.columns}
                try:
                    r = requests.post(API_URL, json=payload, timeout=8)
                    r.raise_for_status()
                    resp = r.json()
                    if resp.get('status') == 'success':
                        data = resp.get('data', {})
                        results.append({
                            'kepid': payload.get('kepid', f'row_{i}'),
                            'disposition': data.get('disposition_prediction'),
                            'confidence': data.get('disposition_probability'),
                            'pred_radius': data.get('predicted_radius_earth')
                        })
                    else:
                        results.append({'kepid': payload.get('kepid', f'row_{i}'), 'disposition': 'ERROR',
                                      'confidence': None, 'pred_radius': None})
                except Exception:
                    results.append({'kepid': payload.get('kepid', f'row_{i}'), 'disposition': 'ERROR',
                                  'confidence': None, 'pred_radius': None})
                
                progress = int((i+1)/len(batch_df)*100)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i+1}/{len(batch_df)} signals...")

            progress_bar.empty()
            status_text.empty()

            res_df = pd.DataFrame(results)
            
            # Results Summary
            st.markdown("### 📊 Batch Results Summary")
            
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            
            total_processed = len([r for r in results if r.get('disposition') != 'ERROR'])
            confirmed = len([r for r in results if r.get('disposition') == 'CONFIRMED'])
            false_positive = len([r for r in results if r.get('disposition') == 'FALSE POSITIVE'])
            errors = len([r for r in results if r.get('disposition') == 'ERROR'])
            
            with result_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Processed</div>
                    <div class="metric-value">{total_processed}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confirmed</div>
                    <div class="metric-value" style="color: #2ECC71;">{confirmed}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">False Positive</div>
                    <div class="metric-value" style="color: #E74C3C;">{false_positive}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Errors</div>
                    <div class="metric-value">{errors}</div>
                </div>
                """, unsafe_allow_html=True)

            # Results Table
            st.markdown("### 📋 Detailed Results")
            
            display_df = res_df.copy()
            if 'confidence' in display_df.columns:
                display_df['confidence'] = display_df['confidence'].apply(
                    lambda x: f"{x:.1%}" if x is not None else "N/A"
                )
            if 'pred_radius' in display_df.columns:
                display_df['pred_radius'] = display_df['pred_radius'].apply(
                    lambda x: f"{x:.3f}R⊕" if x is not None else "N/A"
                )
            
            st.dataframe(display_df, use_container_width=True, height=400)

            # Visualization
            st.markdown("### 📈 Visualization")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if not res_df.empty and 'disposition' in res_df.columns:
                    disposition_valid = res_df[res_df['disposition'] != 'ERROR']
                    if not disposition_valid.empty:
                        fig = px.pie(
                            values=disposition_valid['disposition'].value_counts().values,
                            names=disposition_valid['disposition'].value_counts().index,
                            title='Signal Classification Distribution',
                            color_discrete_map={
                                'CONFIRMED': '#2ECC71',
                                'FALSE POSITIVE': '#E74C3C',
                                'CANDIDATE': '#F39C12'
                            }
                        )
                        fig.update_layout(template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                if not res_df.empty and 'confidence' in res_df.columns:
                    confidence_data = pd.to_numeric(
                        res_df['confidence'].str.rstrip('%'), errors='coerce'
                    ) / 100
                    fig = px.histogram(
                        x=confidence_data,
                        nbins=20,
                        title='Confidence Score Distribution',
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(template='plotly_white',
                                    xaxis_title='Confidence Score',
                                    yaxis_title='Frequency')
                    st.plotly_chart(fig, use_container_width=True)

            # Download Results
            st.markdown("### 📥 Export Results")
            csv = res_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


def main():
    # Enhanced Sidebar Navigation
    st.sidebar.markdown("""
    <div style="background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2>🚀 Navigation</h2>
        <p style="opacity: 0.9;">Select a section to explore</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "📍 Go to Section",
        ["🏠 Introduction", "📊 Data Insights", "🔮 Prediction", "📦 Batch Mode", "⚙️ Architecture"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    
    # Sidebar Info
    st.sidebar.markdown("""
    ### 📚 About This Dashboard
    
    **Stellar Verification Program** helps scientists validate exoplanet candidates using 
    advanced machine learning models trained on NASA Kepler transit data.
    
    **Key Models:**
    - Task A: Signal Classification (CONFIRMED vs FALSE POSITIVE)
    - Task B: Planetary Radius Prediction
    
    **Performance:**
    - F1-Score: 0.9113 ✓
    - RMSE: 0.654 R⊕
    """)
    
    st.sidebar.markdown("---")
    
    # Quick Stats
    if not df.empty:
        st.sidebar.markdown("### 📈 Dataset Stats")
        st.sidebar.metric("Total Records", f"{len(df):,}")
        st.sidebar.metric("Features", df.shape[1])

    # Route to page
    if page == "🏠 Introduction":
        mission_brief()
    elif page == "📊 Data Insights":
        data_insights()
    elif page == "🔮 Prediction":
        prediction_form()
    elif page == "📦 Batch Mode":
        batch_upload_section()
    elif page == "⚙️ Architecture":
        model_performance_and_architecture()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9rem;">
        <p>🌌 Stellar Analytics Dashboard | Phase 1 ML Pipeline</p>
        <p>Powered by Streamlit + Flask + scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
