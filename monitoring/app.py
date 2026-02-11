"""
Dashboard de Monitoring avec Streamlit
Affiche les métriques de production et détecte le data drift
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import os
from scipy import stats

# Configuration de la page
st.set_page_config(
    page_title="Monitoring - Scoring API",
    page_icon="📊",
    layout="wide"
)

# Titre principal
st.title("📊 Dashboard de Monitoring - API de Scoring")
st.markdown("---")


@st.cache_data(ttl=60)
def load_production_logs():
    """
    Charge les logs de production
    """
    logs_file = "logs/production_logs.json"
    
    if not os.path.exists(logs_file):
        return pd.DataFrame()
    
    try:
        logs = []
        with open(logs_file, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        if not logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(logs)
        
        # Parser le timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extraire les features d'input
        input_df = pd.json_normalize(df['input'])
        output_df = pd.json_normalize(df['output'])
        
        # Combiner
        result = pd.concat([df[['timestamp', 'model_version']], input_df, output_df], axis=1)
        
        return result
    except Exception as e:
        st.error(f"Erreur lors du chargement des logs: {e}")
        return pd.DataFrame()


@st.cache_data
def load_reference_data():
    """
    Charge les données de référence (entraînement)
    """
    ref_file = "output/dataset_train_top40.csv"
    
    try:
        return pd.read_csv(ref_file)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de référence: {e}")
        return pd.DataFrame()


def calculate_drift_score(reference, production, column):
    """
    Calcule un score de drift simple avec le test de Kolmogorov-Smirnov
    """
    
    
    if column not in reference.columns or column not in production.columns:
        return None, None
    
    ref_data = reference[column].dropna()
    prod_data = production[column].dropna()
    
    if len(ref_data) == 0 or len(prod_data) == 0:
        return None, None
    
    # Test KS
    statistic, pvalue = stats.ks_2samp(ref_data, prod_data)
    
    return statistic, pvalue


# Charger les données
production_df = load_production_logs()
reference_df = load_reference_data()

# Vérifier si on a des données
if production_df.empty:
    st.warning("⚠️ Aucune donnée de production disponible. Lancez l'API et effectuez des prédictions pour voir les statistiques.")
    st.info("💡 Pour tester l'API: `curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{...}'`")
    st.stop()

# Sidebar pour les filtres
st.sidebar.header("⚙️ Filtres")

# Filtre de temps
time_range = st.sidebar.selectbox(
    "Période",
    ["Dernière heure", "Dernières 24h", "Derniers 7 jours", "Tout"]
)

# Appliquer le filtre temporel
if time_range != "Tout":
    now = datetime.now()
    if time_range == "Dernière heure":
        start_time = now - timedelta(hours=1)
    elif time_range == "Dernières 24h":
        start_time = now - timedelta(days=1)
    else:  # 7 jours
        start_time = now - timedelta(days=7)
    
    production_df = production_df[production_df['timestamp'] >= start_time]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 **{len(production_df)}** prédictions analysées")

# === SECTION 1: MÉTRIQUES CLÉS ===
st.header("🎯 Métriques Clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Prédictions",
        value=len(production_df)
    )

with col2:
    approved = (production_df['decision'] == 'APPROVED').sum() if 'decision' in production_df.columns else 0
    approval_rate = (approved / len(production_df) * 100) if len(production_df) > 0 else 0
    st.metric(
        label="Taux Crédit Accepté",
        value=f"{approval_rate:.1f}%"
    )

with col3:
    avg_time = production_df['inference_time_ms'].mean() if 'inference_time_ms' in production_df.columns else 0
    st.metric(
        label="Temps d'Inférence Moyen",
        value=f"{avg_time:.2f} ms"
    )

with col4:
    if 'score' in production_df.columns:
        LOW, HIGH = 0.15, 0.25

        confident_mask = (production_df['score'] < LOW) | (production_df['score'] > HIGH)
        confidence_rate = confident_mask.mean() * 100

        if confidence_rate >= 90:
            status = "🟢 VERT"
        elif confidence_rate >= 80:
            status = "🟠 ORANGE"
        else:
            status = "🔴 ROUGE"

        st.metric(
            label="Taux de Confiance Modèle",
            value=f"{status}",
            delta=f"{confidence_rate:.1f}% prédictions confiantes"
        )

st.markdown("---")

# === SECTION 2: DISTRIBUTION DES SCORES ===
st.header("📈 Distribution des Scores")

col1, col2 = st.columns(2)

with col1:
    if 'score' in production_df.columns:
        fig_hist = px.histogram(
            production_df,
            x='score',
            nbins=30,
            title="Distribution des Scores de Prédiction",
            labels={'score': 'Score', 'count': 'Fréquence'}
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    if 'decision' in production_df.columns:
        decision_counts = production_df['decision'].value_counts()
        fig_pie = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="Répartition des Décisions",
            color=decision_counts.index,
            color_discrete_map={'APPROVED': 'green', 'REJECTED': 'red'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# === SECTION 3: TEMPS D'INFÉRENCE ===
st.header("⚡ Performance de l'API")

if 'inference_time_ms' in production_df.columns and 'timestamp' in production_df.columns:
    fig_time = px.line(
        production_df.sort_values('timestamp'),
        x='timestamp',
        y='inference_time_ms',
        title="Évolution du Temps d'Inférence",
        labels={'timestamp': 'Temps', 'inference_time_ms': 'Temps (ms)'}
    )
    fig_time.add_hline(
        y=production_df['inference_time_ms'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Moyenne: {production_df['inference_time_ms'].mean():.2f} ms"
    )
    st.plotly_chart(fig_time, use_container_width=True)

st.markdown("---")

# === SECTION 4: DATA DRIFT ===
st.header("🔍 Détection de Data Drift")

st.info("""
**Data Drift**: Changement dans la distribution des données d'entrée entre l'entraînement et la production.
Un drift significatif peut indiquer que le modèle doit être ré-entraîné.
""")

# Features à analyser
features = [
'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN', 'ACTIVE_AMT_CREDIT_SUM_MAX', 'ACTIVE_DAYS_CREDIT_MAX', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'ANNUITY_INCOME_PERC', 'APPROVED_AMT_ANNUITY_MEAN', 'APPROVED_CNT_PAYMENT_MEAN', 'APPROVED_DAYS_DECISION_MAX', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'BURO_DAYS_CREDIT_MAX', 'BURO_DAYS_CREDIT_MEAN', 'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'CLOSED_AMT_CREDIT_SUM_MAX', 'CLOSED_DAYS_CREDIT_ENDDATE_MAX', 'CLOSED_DAYS_CREDIT_MAX', 'CLOSED_DAYS_CREDIT_VAR', 'CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERC', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'INSTAL_AMT_PAYMENT_MEAN', 'INSTAL_AMT_PAYMENT_MIN', 'INSTAL_AMT_PAYMENT_SUM', 'INSTAL_DBD_MAX', 'INSTAL_DBD_SUM', 'INSTAL_DPD_MEAN', 'INSTAL_PAYMENT_PERC_MEAN', 'OWN_CAR_AGE', 'PAYMENT_RATE', 'POS_MONTHS_BALANCE_SIZE', 'PREV_CNT_PAYMENT_MEAN'
]

# Calculer le drift pour chaque feature
drift_results = []

for feature in features:
    if feature in production_df.columns and feature in reference_df.columns:
        ks_stat, p_value = calculate_drift_score(reference_df, production_df, feature)
        if ks_stat is not None:
            drift_results.append({
                'Feature': feature,
                'KS Statistic': ks_stat,
                'P-Value': p_value,
                'Drift Détecté': '🔴 OUI' if p_value < 0.05 else '🟢 NON'
            })

if drift_results:
    drift_df = pd.DataFrame(drift_results)
    st.dataframe(drift_df, use_container_width=True)
    
    # Alertes
    drifted_features = [r['Feature'] for r in drift_results if r['P-Value'] < 0]
    if drifted_features:
        st.error(f"⚠️ **ALERTE DRIFT**: Drift détecté sur {len(drifted_features)} feature(s): {', '.join(drifted_features)}")
        st.warning("💡 **Recommandation**: Considérez un ré-entraînement du modèle avec des données récentes.")
    else:
        st.success("✅ Aucun drift significatif détecté. Le modèle semble stable.")

# Comparaison visuelle des distributions
st.subheader("📊 Comparaison des Distributions (Référence vs Production)")

selected_feature = st.selectbox("Sélectionnez une feature à analyser:", features)

if selected_feature in production_df.columns and selected_feature in reference_df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ref = px.histogram(
            reference_df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution de Référence - {selected_feature}",
            color_discrete_sequence=['blue']
        )
        st.plotly_chart(fig_ref, use_container_width=True)
    
    with col2:
        fig_prod = px.histogram(
            production_df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution en Production - {selected_feature}",
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig_prod, use_container_width=True)

st.markdown("---")

# === SECTION 5: LOGS RÉCENTS ===
st.header("📋 Logs Récents")

n_logs = st.slider("Nombre de logs à afficher:", 5, 50, 10)

if not production_df.empty:
    recent_logs = production_df.sort_values('timestamp', ascending=False).head(n_logs)
    st.dataframe(recent_logs, use_container_width=True)

# === SECTION 6: EXPORT ===
st.markdown("---")
st.header("💾 Export des Données")

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Télécharger les logs (CSV)"):
        csv = production_df.to_csv(index=False)
        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name=f"production_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📊 Télécharger le rapport de drift (CSV)"):
        if drift_results:
            drift_csv = pd.DataFrame(drift_results).to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=drift_csv,
                file_name=f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Auto-refresh
st.markdown("---")
if st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False):
    st.rerun()
