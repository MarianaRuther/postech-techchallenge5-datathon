"""
Datathon FIAP Fase 5 — Associação Passos Mágicos
=================================================

App Streamlit que serve o modelo de previsão de risco de defasagem.

Tutoria insere os indicadores de um aluno → recebe:
- Probabilidade de risco no próximo ano
- Classificação em 4 níveis (Baixo / Observação / Atenção / Urgente)
- Top 5 fatores que mais empurraram pra risco e top 5 que protegeram (SHAP local)
- Recomendação de ação por nível

Usa o modelo do notebook 04 (LogReg) e os artefatos do notebook 06 (SHAP summary).

Deploy: Streamlit Community Cloud — https://share.streamlit.io
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================================
st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "App de previsão de risco de defasagem da Associação Passos Mágicos.\n\n"
            "Datathon FIAP Fase 5 — Deep Learning and Unstructured Data."
        ),
    },
)


# =========================================================================
# PATHS — caminhos relativos (funciona local e no Streamlit Cloud)
# =========================================================================
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


# =========================================================================
# CARREGAR ARTEFATOS (com cache pra não recarregar a cada interação)
# =========================================================================
@st.cache_resource
def carregar_modelo():
    """Carrega pipeline (impute + scale + LogReg) treinado no notebook 04."""
    return joblib.load(MODELS_DIR / "modelo_risco_v1.pkl")


@st.cache_resource
def carregar_metadados():
    """Carrega thresholds, feature_names e summary do SHAP."""
    with open(MODELS_DIR / "thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)
    with open(MODELS_DIR / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    shap_summary_path = MODELS_DIR / "shap_summary.json"
    shap_summary = None
    if shap_summary_path.exists():
        with open(shap_summary_path, encoding="utf-8") as f:
            shap_summary = json.load(f)
    return thresholds, feature_names, shap_summary


modelo = carregar_modelo()
thresholds, FEATURE_NAMES, shap_summary = carregar_metadados()


# =========================================================================
# HEADER
# =========================================================================
st.title("🎓 Passos Mágicos — Previsão de Risco de Defasagem")
st.caption(
    "Modelo treinado em dados de 2022→2023 e validado em 2023→2024. "
    "Datathon FIAP Fase 5."
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Modelo", thresholds["modelo_selecionado"])
col_b.metric(
    "PR-AUC (teste)",
    f"{shap_summary['metricas_teste']['pr_auc']:.3f}" if shap_summary else "—",
)
col_c.metric(
    "Precisão @ Atenção",
    f"{shap_summary['metricas_teste']['precision_atencao']:.0%}" if shap_summary else "—",
)
col_d.metric(
    "Recall @ Atenção",
    f"{shap_summary['metricas_teste']['recall_atencao']:.0%}" if shap_summary else "—",
)

st.divider()

# Placeholder pros próximos pedaços
st.info(
    "🚧 **Pedaço 1 do app — skeleton em construção.** "
    "Os próximos passos vão adicionar: inputs do aluno (sidebar), predição com semáforo, "
    "SHAP local e estatísticas agregadas."
)

# Sanity de carregamento
with st.expander("🔧 Diagnóstico (dev) — confirma que tudo carregou"):
    st.write(f"**Modelo carregado:** `{type(modelo).__name__}` com {len(FEATURE_NAMES)} features")
    st.write(f"**Thresholds:** Observação={thresholds['observacao']:.3f} | "
             f"Atenção={thresholds['atencao']:.3f} | "
             f"Urgente={thresholds['urgente']:.3f}")
    if shap_summary:
        st.write(f"**SHAP summary:** {len(shap_summary['logreg_coefficients'])} coeficientes salvos")
        st.write(f"**Top 5 features (consenso):** {', '.join(shap_summary['top5_consenso'])}")
    st.write("**Features esperadas pelo modelo (na ordem):**")
    st.code(", ".join(FEATURE_NAMES), language="text")
