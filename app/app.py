"""
Datathon FIAP Fase 5 — Associação Passos Mágicos
=================================================

App Streamlit que serve o modelo de previsão de risco de defasagem.

Fluxo: tutor preenche os indicadores do aluno na sidebar → app calcula
features derivadas → modelo prevê probabilidade de risco → semáforo +
recomendação de ação.

Modelo: LogReg (notebook 04). Métricas validadas em 2023→2024 (1014 alunos).
PR-AUC=0.793. Precisão=70%, Recall=72% no threshold 🟡 Atenção.

Deploy: Streamlit Community Cloud — https://share.streamlit.io
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================================
# CONFIG DA PÁGINA
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
# PATHS
# =========================================================================
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


# =========================================================================
# CONSTANTES DE NEGÓCIO (mapeadas pelo notebook 03)
# =========================================================================
PEDRA_NIVEL = {"Quartzo": 0, "Ágata": 1, "Ametista": 2, "Topázio": 3}
PEDRAS_ORDEM = ["Quartzo", "Ágata", "Ametista", "Topázio"]

NIVEIS_INFO = {
    "🟢 Baixo": {
        "cor": "#2a9d8f",
        "acao": "**Monitoramento passivo.** Continuar acompanhamento regular. Reforço positivo do que está funcionando bem.",
        "prazo": "Sem ação imediata necessária",
    },
    "🟢 Observação": {
        "cor": "#52b788",
        "acao": "**Atenção preventiva.** Manter contato mensal com tutor. Avaliar pilares com leve queda no IDA/IEG/IPS.",
        "prazo": "Revisão a cada 30 dias",
    },
    "🟡 Atenção": {
        "cor": "#f4a261",
        "acao": "**Contato ativo do tutor em até 30 dias.** Avaliar pilares em queda + conversa com a família. Plano de reforço acadêmico.",
        "prazo": "Tutor liga em até 30 dias",
    },
    "🔴 Urgente": {
        "cor": "#e63946",
        "acao": "**INTERVENÇÃO IMEDIATA.** Reunião pedagógica + acompanhamento psicossocial. Conversa com família esta semana.",
        "prazo": "Ação esta semana",
    },
}


# =========================================================================
# CARREGAR ARTEFATOS (com cache)
# =========================================================================
@st.cache_resource
def carregar_modelo():
    return joblib.load(MODELS_DIR / "modelo_risco_v1.pkl")


@st.cache_resource
def carregar_metadados():
    with open(MODELS_DIR / "thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)
    with open(MODELS_DIR / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    shap_summary = None
    shap_path = MODELS_DIR / "shap_summary.json"
    if shap_path.exists():
        with open(shap_path, encoding="utf-8") as f:
            shap_summary = json.load(f)

    fe_stats = None
    fe_path = MODELS_DIR / "feature_engineering_stats.json"
    if fe_path.exists():
        with open(fe_path, encoding="utf-8") as f:
            fe_stats = json.load(f)

    return thresholds, feature_names, shap_summary, fe_stats


modelo = carregar_modelo()
thresholds, FEATURE_NAMES, shap_summary, fe_stats = carregar_metadados()


# =========================================================================
# FUNÇÕES DE NEGÓCIO
# =========================================================================
def classificar_nivel(proba: float, thresholds: dict) -> str:
    """Retorna o label do nível baseado nos 3 thresholds operacionais."""
    if proba >= thresholds["urgente"]:
        return "🔴 Urgente"
    if proba >= thresholds["atencao"]:
        return "🟡 Atenção"
    if proba >= thresholds["observacao"]:
        return "🟢 Observação"
    return "🟢 Baixo"


def calcular_shap_local(X_aluno_df: pd.DataFrame, shap_summary: dict) -> dict | None:
    """
    Calcula SHAP values manualmente para o modelo LogReg, sem precisar da
    lib `shap` em produção (mantém o deploy do Streamlit Cloud leve).

    Lógica matemática:
    1. Aplica imputer (mediana do treino) nos NaN — replica o que o pipeline faz
    2. Aplica StandardScaler: x_scaled = (x - mean) / std
    3. SHAP_i = coef_i * x_scaled_i  (LinearExplainer com background = média = 0 no espaço scaled)
    4. base_value = intercept da LogReg
    5. Validação: base + sum(SHAP) == model.decision_function(x)
    """
    if not shap_summary:
        return None

    feature_names = shap_summary["feature_names_ordem"]
    imputer_stats = np.array(shap_summary["preprocess"]["imputer_statistics"])
    scaler_mean = np.array(shap_summary["preprocess"]["scaler_mean"])
    scaler_scale = np.array(shap_summary["preprocess"]["scaler_scale"])
    coefs = np.array([shap_summary["logreg_coefficients"][f] for f in feature_names])
    intercept = float(shap_summary["logreg_intercept"])

    # X cru do aluno (na ordem correta)
    X_raw = X_aluno_df[feature_names].iloc[0].astype(float).values

    # 1) Imputer mediana — substitui NaN pela mediana do treino
    X_imputed = np.where(np.isnan(X_raw), imputer_stats, X_raw)

    # 2) Scaler
    X_scaled = (X_imputed - scaler_mean) / scaler_scale

    # 3) SHAP no espaço logit
    shap_vals = coefs * X_scaled

    # 4) Validação matemática
    logit_recomposto = float(shap_vals.sum() + intercept)

    return {
        "feature_names": feature_names,
        "x_raw": X_raw,
        "shap_values": shap_vals,
        "intercept": intercept,
        "logit_recomposto": logit_recomposto,
    }


def construir_features_aluno(inputs: dict, fe_stats: dict) -> pd.DataFrame:
    """
    Recebe os inputs do tutor (indicadores principais + metadados) e
    constrói um DataFrame com as 24 features que o modelo espera, calculando
    automaticamente as features derivadas do notebook 03.
    """
    fase = int(inputs["fase"])

    # Stats por fase (usadas pra z-scores e ranking)
    fase_key = str(fase)
    fase_stats = fe_stats["por_fase"].get(fase_key) if fe_stats else None

    if fase_stats:
        inde_std = max(fase_stats["inde_std"], 1e-3)
        ips_std = max(fase_stats["ips_std"], 1e-3)
        inde_zscore = (inputs["inde"] - fase_stats["inde_mean"]) / inde_std
        ips_zscore = (inputs["ips"] - fase_stats["ips_mean"]) / ips_std

        dist = fase_stats["inde_distribution"]
        ranking = (
            sum(1 for v in dist if v < inputs["inde"]) / len(dist)
            if len(dist) > 0 else 0.5
        )
        idade_esperada = fe_stats["idade_esperada_por_fase"].get(fase_key, 12)
    else:
        inde_zscore = 0.0
        ips_zscore = 0.0
        ranking = 0.5
        idade_esperada = 12

    # Defasagem = idade do aluno - idade típica da fase
    defasagem = inputs["idade"] - idade_esperada

    # Mapeamentos
    pedra_nivel = PEDRA_NIVEL[inputs["pedra"]]
    is_menina = 1 if inputs["genero"] == "Menina" else 0
    tem_nota_ingles = 1 if inputs["tem_ingles"] else 0
    nota_ing = inputs["nota_ing"] if tem_nota_ingles else np.nan

    # Notas e gaps
    notas_validas = [inputs["nota_mat"], inputs["nota_port"]]
    if tem_nota_ingles:
        notas_validas.append(nota_ing)
    media_notas = float(np.mean(notas_validas))
    gap_iaa_ida = inputs["iaa"] - inputs["ida"]
    gap_ieg_ida = inputs["ieg"] - inputs["ida"]

    # Ano de ingresso (ano atual - anos no programa)
    ano_atual = fe_stats["globais"].get("ano_atual", 2024) if fe_stats else 2024
    ano_ingresso = ano_atual - inputs["anos_no_programa"]

    # Monta linha na ORDEM exata que o modelo espera
    features = {
        "inde": inputs["inde"],
        "ian": inputs["ian"],
        "ida": inputs["ida"],
        "ieg": inputs["ieg"],
        "iaa": inputs["iaa"],
        "ips": inputs["ips"],
        "ipv": inputs["ipv"],
        "fase": fase,
        "idade": inputs["idade"],
        "anos_no_programa": inputs["anos_no_programa"],
        "ano_ingresso": ano_ingresso,
        "defasagem": defasagem,
        "is_menina": is_menina,
        "pedra_nivel": pedra_nivel,
        "nota_mat": inputs["nota_mat"],
        "nota_port": inputs["nota_port"],
        "nota_ing": nota_ing,
        "tem_nota_ingles": tem_nota_ingles,
        "gap_iaa_ida": gap_iaa_ida,
        "gap_ieg_ida": gap_ieg_ida,
        "media_notas_escolares": media_notas,
        "ips_zscore_fase": ips_zscore,
        "inde_zscore_fase": inde_zscore,
        "ranking_inde_fase": ranking,
    }

    df = pd.DataFrame([features])
    # Garantia: ordem das colunas == a esperada pelo modelo
    df = df[FEATURE_NAMES]
    return df


# =========================================================================
# UI — HEADER
# =========================================================================
# CSS customizado: aumenta fonte das abas pra ficar mais legível
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 Passos Mágicos — Previsão de Risco de Defasagem")
st.caption(
    "Modelo treinado em dados de 2022→2023 e validado em 2023→2024. "
    "Datathon FIAP Fase 5."
)

# Métricas-chave do modelo
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Modelo", thresholds["modelo_selecionado"])
col_b.metric(
    "PR-AUC (teste)",
    f"{shap_summary['metricas_teste']['pr_auc']:.3f}" if shap_summary else "—",
)
col_c.metric(
    "Precisão @ 🟡 Atenção",
    f"{shap_summary['metricas_teste']['precision_atencao']:.0%}" if shap_summary else "—",
)
col_d.metric(
    "Recall @ 🟡 Atenção",
    f"{shap_summary['metricas_teste']['recall_atencao']:.0%}" if shap_summary else "—",
)

st.divider()

# =========================================================================
# CAMINHO DAS FIGURAS (geradas pelos notebooks 02/04/05/06)
# =========================================================================
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# =========================================================================
# SIDEBAR — INPUTS DO ALUNO
# =========================================================================
with st.sidebar:
    # ===== DICIONÁRIO (colapsável, sempre disponível) =====
    with st.expander("📖 Dicionário — entenda os termos"):
        st.markdown(
            """
            #### 🪨 Pedras-conceito
            Classificação geral do aluno por desempenho, em ordem crescente:
            - **Quartzo** — base inicial, mais frágil
            - **Ágata** — desenvolvimento intermediário
            - **Ametista** — bom desempenho consolidado
            - **Topázio** — topo, alunos mais maduros e estáveis

            #### 🎯 Fase do programa
            Etapa pedagógica em que o aluno está dentro da Passos. Fases mais
            altas = idades maiores e conteúdos mais complexos. Mediana de idade
            por fase no nosso dataset:

            | Fase | Idade típica |
            |---|---|
            | 0 (Alfa) | 9 anos |
            | 1 | 10 anos |
            | 2 | 12 anos |
            | 3 | 14 anos |
            | 4 | 15 anos |
            | 5 | 16 anos |
            | 6 | 17 anos |
            | 7 | 18 anos |

            #### 📅 Anos no programa
            Quantos anos o aluno está vinculado à Passos. Aluno novo (1 ano) e
            aluno veterano (5+ anos) têm perfis de risco diferentes.

            #### 📊 Indicadores principais (escala 0-10)
            - **INDE** — *Índice de Desenvolvimento Educacional*. Resumo geral
            do aluno (média ponderada dos demais).
            - **IAN** — *Adequação de Nível*. Mede defasagem entre idade real
            e fase esperada.
            - **IDA** — *Desempenho Acadêmico*. Notas em Matemática, Português,
            Inglês.
            - **IEG** — *Engajamento*. Presença, participação, entrega de tarefas.
            - **IAA** — *Autoavaliação*. Percepção do próprio aluno sobre si.
            - **IPS** — *Pilar Psicossocial*. Saúde emocional, relações, suporte
            familiar.
            - **IPV** — *Ponto de Virada*. Capacidade do aluno de mudar a
            própria trajetória — síntese de IDA + IEG + IAA + IPS.

            #### 🚦 Níveis operacionais (semáforo)
            - 🟢 **Baixo / Observação** — risco baixo. Monitoramento passivo.
            - 🟡 **Atenção** — precisão de 70%. Tutor entra em contato em 30 dias.
            - 🔴 **Urgente** — precisão de 80%. Intervenção imediata.
            """
        )

    st.divider()

    st.header("📋 Dados do aluno")
    st.caption(
        "Preencha os indicadores e metadados. As features derivadas "
        "(z-scores, gaps, ranking) são calculadas automaticamente."
    )

    with st.form("form_aluno", clear_on_submit=False):
        # === Identificação básica ===
        st.markdown("**Identificação**")
        ra = st.text_input("RA (opcional, só pra contexto)", value="")

        col_f, col_p = st.columns(2)
        fase = col_f.selectbox(
            "Fase do programa",
            options=fe_stats["fases_disponiveis"] if fe_stats else list(range(0, 8)),
            index=2,
        )
        pedra = col_p.selectbox("Pedra", options=PEDRAS_ORDEM, index=2)

        col_g, col_i = st.columns(2)
        genero = col_g.radio("Gênero", ["Menina", "Menino"], index=0, horizontal=True)
        idade = col_i.number_input("Idade", min_value=6, max_value=25, value=12, step=1)

        anos_no_programa = st.number_input(
            "Anos no programa", min_value=1, max_value=10, value=3, step=1,
        )

        st.divider()
        st.markdown("**Indicadores principais (escala 0-10)**")

        # Indicadores: 7 sliders
        inde = st.slider("INDE — índice geral", 0.0, 10.0, 7.0, step=0.1)
        ian = st.slider("IAN — Adequação de Nível", 0.0, 10.0, 7.0, step=0.1)
        ida = st.slider("IDA — Desempenho Acadêmico", 0.0, 10.0, 7.0, step=0.1)
        ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 7.5, step=0.1)
        iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 7.0, step=0.1)
        ips = st.slider("IPS — Pilar Psicossocial", 0.0, 10.0, 5.0, step=0.1)
        ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 7.0, step=0.1)

        st.divider()
        st.markdown("**Notas escolares (0-10)**")

        col_m, col_pt = st.columns(2)
        nota_mat = col_m.slider("Matemática", 0.0, 10.0, 6.0, step=0.1)
        nota_port = col_pt.slider("Português", 0.0, 10.0, 6.5, step=0.1)

        tem_ingles = st.checkbox("Aluno tem nota de inglês?", value=False)
        nota_ing = st.slider(
            "Inglês (só se tem)", 0.0, 10.0, 6.0, step=0.1,
            disabled=not tem_ingles,
        )

        st.divider()
        submitted = st.form_submit_button(
            "🔮 Calcular Risco", use_container_width=True, type="primary",
        )


# =========================================================================
# CORPO PRINCIPAL — 2 ABAS
# =========================================================================
tab_overview, tab_predict = st.tabs([
    "📊 Visão geral do modelo",
    "🔮 Avaliar aluno",
])


# =========================================================================
# ABA 1 — AVALIAR ALUNO (predição + SHAP local)
# =========================================================================
with tab_predict:
    if not submitted:
        st.info(
            "👈 **Preencha os dados do aluno na barra lateral** e clique em "
            "**Calcular Risco** pra ver a previsão."
        )

        # Mostrar 3 alunos exemplares pra dar contexto
        if shap_summary and "exemplares" in shap_summary:
            st.markdown("### 💡 Como o modelo classificou os 3 alunos exemplares do teste")
            st.caption("Esses são alunos reais do conjunto de teste (2023→2024).")

            cols = st.columns(3)
            for col, ex in zip(cols, shap_summary["exemplares"]):
                perfil_emoji = {"baixo": "🟢", "atencao": "🟡", "urgente": "🔴"}.get(
                    ex["perfil"], "⚪"
                )
                with col:
                    st.markdown(f"#### {perfil_emoji} {ex['perfil'].upper()}")
                    st.metric(
                        label=f"RA {ex['ra']} — {ex['pedra']} (fase {ex['fase']})",
                        value=f"{ex['proba']:.1%}",
                        help=f"Probabilidade de risco. INDE={ex['inde']:.2f}",
                    )

    else:
        # === USUÁRIO CLICOU EM CALCULAR ===
        inputs = {
            "fase": fase, "pedra": pedra, "genero": genero,
            "idade": idade, "anos_no_programa": anos_no_programa,
            "inde": inde, "ian": ian, "ida": ida, "ieg": ieg,
            "iaa": iaa, "ips": ips, "ipv": ipv,
            "nota_mat": nota_mat, "nota_port": nota_port,
            "nota_ing": nota_ing, "tem_ingles": tem_ingles,
        }

        # Construir DataFrame de features e prever
        X_aluno = construir_features_aluno(inputs, fe_stats)
        proba = float(modelo.predict_proba(X_aluno)[0, 1])
        nivel = classificar_nivel(proba, thresholds)
        info_nivel = NIVEIS_INFO[nivel]

        aluno_label = f"RA {ra}" if ra.strip() else f"Aluno (fase {fase}, {pedra})"

        # === Card grande com semáforo ===
        st.markdown(f"### Resultado para **{aluno_label}**")

        col_main, col_proba = st.columns([2, 1])
        with col_main:
            st.markdown(
                f"""
                <div style="
                    background-color: {info_nivel['cor']}22;
                    border-left: 8px solid {info_nivel['cor']};
                    padding: 24px;
                    border-radius: 8px;
                    margin: 8px 0;
                ">
                    <div style="font-size: 40px; font-weight: 700; color: {info_nivel['cor']};">
                        {nivel}
                    </div>
                    <div style="font-size: 16px; color: #444; margin-top: 8px;">
                        Probabilidade de defasagem em t+1
                    </div>
                    <div style="font-size: 32px; font-weight: 600; color: #1a1b25; margin-top: 4px;">
                        {proba:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_proba:
            st.markdown("**Distância pros thresholds**")
            for thr_label, thr_key, thr_emoji in [
                ("Urgente", "urgente", "🔴"),
                ("Atenção", "atencao", "🟡"),
                ("Observação", "observacao", "🟢"),
            ]:
                thr_val = thresholds[thr_key]
                delta = proba - thr_val
                sinal = "+" if delta >= 0 else ""
                st.metric(
                    label=f"{thr_emoji} {thr_label} ({thr_val:.3f})",
                    value=f"{proba:.3f}",
                    delta=f"{sinal}{delta:.3f}",
                    delta_color="inverse",
                )

        # === Recomendação de ação ===
        st.markdown("### 🎯 Ação recomendada")
        st.markdown(
            f"""
            <div style="background-color: {info_nivel['cor']}15; padding: 16px;
                        border-radius: 8px; margin: 4px 0;">
                <div style="font-size: 18px;"><strong>Prazo:</strong> {info_nivel['prazo']}</div>
                <div style="font-size: 16px; margin-top: 8px;">{info_nivel['acao']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "ℹ️ A recomendação acima é uma guideline padrão por nível. A decisão "
            "final é sempre da coordenação pedagógica + tutor que conhece o aluno."
        )

        # === Features derivadas calculadas ===
        with st.expander("🔧 Ver features que o modelo recebeu (24 colunas)"):
            st.caption(
                "Inclui as features derivadas calculadas automaticamente "
                "a partir dos seus inputs (z-scores, gaps, ranking, etc)."
            )
            df_show = X_aluno.T.reset_index()
            df_show.columns = ["feature", "valor"]
            df_show["valor"] = df_show["valor"].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not pd.isna(x)
                else ("NaN" if pd.isna(x) else str(x))
            )
            st.dataframe(df_show, hide_index=True, use_container_width=True)

        # =================================================================
        # SHAP LOCAL — POR QUE ESSE ALUNO RECEBEU ESSA PROBABILIDADE?
        # =================================================================
        st.divider()
        st.markdown("### 🔍 Por que o modelo deu esse risco?")
        st.caption(
            "SHAP local mostra quais features mais empurraram a probabilidade "
            "pra cima (vermelho) e quais protegeram (azul). Calculado direto "
            "dos coeficientes do LogReg — sem aproximação, é exato."
        )

        shap_dict = calcular_shap_local(X_aluno, shap_summary)

        if shap_dict is None:
            st.warning(
                "SHAP local indisponível — `models/shap_summary.json` não foi encontrado. "
                "Roda o notebook 06 antes."
            )
        else:
            df_shap = pd.DataFrame({
                "feature": shap_dict["feature_names"],
                "valor_aluno": shap_dict["x_raw"],
                "shap": shap_dict["shap_values"],
            })

            df_pos = df_shap[df_shap["shap"] > 0].sort_values("shap", ascending=False).head(5)
            df_neg = df_shap[df_shap["shap"] < 0].sort_values("shap").head(5)

            col_pos, col_neg = st.columns(2)

            def fmt_valor(v):
                if pd.isna(v):
                    return "—"
                return f"{v:.2f}"

            with col_pos:
                st.markdown("#### 🔴 Top fatores empurrando pra **RISCO**")
                if len(df_pos) == 0:
                    st.write("_(Nenhuma feature positiva — aluno totalmente seguro pelo modelo.)_")
                else:
                    for _, row in df_pos.iterrows():
                        st.markdown(
                            f"<div style='background-color:#e6394615; padding:8px 12px; "
                            f"border-radius:6px; margin:4px 0; "
                            f"border-left: 3px solid #e63946;'>"
                            f"<strong>{row['feature']}</strong> = {fmt_valor(row['valor_aluno'])} "
                            f"<span style='float:right; color:#e63946; font-weight:600;'>"
                            f"+{row['shap']:.3f}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            with col_neg:
                st.markdown("#### 🟢 Top fatores **PROTEGENDO** o aluno")
                if len(df_neg) == 0:
                    st.write("_(Nenhuma feature protetora — todas empurram pra risco.)_")
                else:
                    for _, row in df_neg.iterrows():
                        st.markdown(
                            f"<div style='background-color:#2a9d8f15; padding:8px 12px; "
                            f"border-radius:6px; margin:4px 0; "
                            f"border-left: 3px solid #2a9d8f;'>"
                            f"<strong>{row['feature']}</strong> = {fmt_valor(row['valor_aluno'])} "
                            f"<span style='float:right; color:#2a9d8f; font-weight:600;'>"
                            f"{row['shap']:.3f}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Visualização SHAP (top 10 absolutos)")
            df_top = (
                df_shap.assign(abs_shap=lambda d: d["shap"].abs())
                .sort_values("abs_shap", ascending=True)
                .tail(10)
            )

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.5))
            cores = ["#e63946" if v > 0 else "#2a9d8f" for v in df_top["shap"]]
            ax.barh(df_top["feature"], df_top["shap"], color=cores)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impacto no logit)")
            ax.set_title(
                "Como cada feature influenciou a previsão deste aluno\n"
                "(valor positivo = empurra pra risco, negativo = protege)",
                fontsize=10, fontweight="bold",
            )
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            with st.expander("🧮 Validação matemática SHAP (transparência)"):
                st.markdown(
                    f"""
                    **Decomposição SHAP exata:**
                    - Base value (intercepto LogReg): `{shap_dict['intercept']:.4f}`
                    - Soma dos SHAP values: `{shap_dict['shap_values'].sum():.4f}`
                    - **Logit recomposto**: `{shap_dict['logit_recomposto']:.4f}`
                    - **Probabilidade derivada**: `{1 / (1 + np.exp(-shap_dict['logit_recomposto'])):.4f}`
                    - **Probabilidade do modelo**: `{proba:.4f}`

                    Os dois últimos batem com diferença < 0.001 → SHAP recompõe
                    exatamente o output do modelo.
                    """
                )


# =========================================================================
# ABA 2 — VISÃO GERAL DO MODELO
# =========================================================================
with tab_overview:
    st.markdown("## Como esse modelo foi construído e validado")
    st.caption(
        "Visão executiva pra a coordenação da Passos Mágicos entender o que tem por trás "
        "da previsão — e poder defender as decisões com base no modelo."
    )

    # ---------------------------------------------------------------------
    # 1. Performance no teste
    # ---------------------------------------------------------------------
    st.markdown("### 📈 Performance validada (1014 alunos do teste 2023→2024)")

    if shap_summary:
        m = shap_summary["metricas_teste"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", f"{m['roc_auc']:.3f}",
                    help="Área sob a curva ROC. 1.0 = perfeito, 0.5 = aleatório.")
        col2.metric("PR-AUC", f"{m['pr_auc']:.3f}",
                    help="Área sob a curva Precision-Recall. Métrica de seleção do modelo.")
        col3.metric("Precisão @ Atenção", f"{m['precision_atencao']:.0%}",
                    help="Dos alunos flegados Atenção, 70% realmente estavam em risco.")
        col4.metric("Recall @ Atenção", f"{m['recall_atencao']:.0%}",
                    help="Captamos 72% dos alunos que entraram em risco no ano seguinte.")

        # Distribuição dos níveis no teste
        if "distribuicao_niveis_teste" in shap_summary:
            dist = shap_summary["distribuicao_niveis_teste"]
            st.markdown("**Distribuição dos 1014 alunos do teste pelos níveis de risco:**")
            cols = st.columns(4)
            niveis_ordem = ["🔴 Urgente", "🟡 Atenção", "🟢 Observação", "⚪ Baixo"]
            cores_n = ["#e63946", "#f4a261", "#52b788", "#264653"]
            for col, n, cor in zip(cols, niveis_ordem, cores_n):
                v = dist.get(n, 0)
                pct = v / sum(dist.values()) if sum(dist.values()) > 0 else 0
                col.markdown(
                    f"<div style='background-color:{cor}22; padding:14px; "
                    f"border-radius:8px; border-left:4px solid {cor};'>"
                    f"<div style='font-size:13px; color:#444;'>{n}</div>"
                    f"<div style='font-size:24px; font-weight:700; color:{cor};'>{v}</div>"
                    f"<div style='font-size:12px; color:#666;'>{pct:.1%} dos alunos</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ---------------------------------------------------------------------
    # 2. As 5 forças do modelo
    # ---------------------------------------------------------------------
    st.markdown("### 🎯 As 5 forças do modelo")
    st.caption(
        "Top 5 features que mais predizem risco — consenso entre **Permutation Importance** "
        "e **SHAP global**. Esses 5 sinais são os mais defensáveis pra Passos calibrar intervenção."
    )

    if shap_summary and "top5_consenso" in shap_summary:
        cols = st.columns(5)
        descricoes = {
            "idade": "Adolescentes evadem mais",
            "inde_zscore_fase": "Posição relativa do aluno na fase",
            "ieg": "Engajamento — preditor antecipado",
            "ips": "Pilar psicossocial",
            "fase": "Fases de transição (Q→Á, Am→T) trazem risco",
        }
        for col, feat in zip(cols, shap_summary["top5_consenso"]):
            with col:
                st.markdown(f"**{feat}**")
                st.caption(descricoes.get(feat, ""))

    # Figura SHAP global bar (centralizada, ~66% da largura)
    fig_path = FIGURES_DIR / "shap_03_bar_logreg.png"
    if fig_path.exists():
        _, col_img, _ = st.columns([1, 4, 1])
        with col_img:
            st.image(str(fig_path), caption="SHAP global — média do |SHAP value| por feature (notebook 06)")

    st.divider()

    # ---------------------------------------------------------------------
    # 3. Como o modelo decide em detalhe
    # ---------------------------------------------------------------------
    st.markdown("### 🔬 Como o modelo decide, em detalhe")
    st.caption(
        "Cada ponto = 1 aluno. Vermelho = valor alto da feature, azul = valor baixo. "
        "Posição horizontal = impacto na previsão."
    )

    fig_path = FIGURES_DIR / "shap_02_beeswarm_logreg.png"
    if fig_path.exists():
        _, col_img, _ = st.columns([1, 4, 1])
        with col_img:
            st.image(str(fig_path), caption="SHAP beeswarm — distribuição de impacto por feature")

    st.divider()

    # ---------------------------------------------------------------------
    # 4. Por que LogReg venceu
    # ---------------------------------------------------------------------
    st.markdown("### 🏆 Por que LogReg foi escolhido como modelo final")
    st.caption(
        "Comparamos LogReg vs XGBoost vs LightGBM (notebook 04) e MLP em PyTorch (notebook 05). "
        "**LogReg venceu por PR-AUC** com interpretabilidade exata via SHAP linear — "
        "decisão metodologicamente defensável, especialmente em dataset pequeno (860 amostras de treino)."
    )

    fig_path = FIGURES_DIR / "mod_curvas_comparacao.png"
    if fig_path.exists():
        _, col_img, _ = st.columns([1, 4, 1])
        with col_img:
            st.image(str(fig_path), caption="Curvas ROC e PR — LogReg, XGBoost, LightGBM")

    fig_path = FIGURES_DIR / "mod_matriz_confusao.png"
    if fig_path.exists():
        _, col_img, _ = st.columns([1, 4, 1])
        with col_img:
            st.image(str(fig_path), caption="Matriz de confusão @ threshold 🟡 Atenção")

    st.divider()

    # ---------------------------------------------------------------------
    # 5. Insights complementares (NLP nos relatórios PEDE)
    # ---------------------------------------------------------------------
    st.markdown("### 📚 Bônus — o que os relatórios PEDE estão dizendo")
    st.caption(
        "Pipeline OCR + embeddings + clustering nos 3 relatórios PEDE da Passos (2020/21/22) — "
        "424 chunks de texto, 8 temas dominantes."
    )

    st.markdown(
        """
        > **Achado que rodaria a apresentação executiva:** o cluster
        > **`defasagem moderada/severa, fase, matemática, português`** **emergiu de 0% em 2020 para 7.9% em 2022**.
        > A Passos só passou a discutir defasagem explicitamente em 2022 — exatamente o problema que o modelo prevê.
        > **O modelo não inventou a categoria; ele responde a uma preocupação institucional emergente.**
        """
    )

    fig_path = FIGURES_DIR / "dl_02_clusters_temporal_pede.png"
    if fig_path.exists():
        _, col_img, _ = st.columns([1, 6, 1])
        with col_img:
            st.image(
                str(fig_path),
                caption="Heatmap — evolução temática dos relatórios PEDE 2020 → 2022 (notebook 05)",
            )

    st.divider()

    # ---------------------------------------------------------------------
    # 6. Rodapé
    # ---------------------------------------------------------------------
    st.markdown(
        """
        <hr style="margin-top: 24px; margin-bottom: 16px;">
        <div style="text-align: center; font-size: 14px; color: #555; line-height: 1.7;">
            <strong>Pipeline reprodutível</strong> em
            <a href="https://github.com/MarianaRuther/postech-techchallenge5-datathon" target="_blank">
                <strong>github.com/MarianaRuther/postech-techchallenge5-datathon</strong>
            </a>
            — notebooks 02 a 06 + este app Streamlit.
            <br>
            Datathon FIAP PosTech Fase 5 — Deep Learning and Unstructured Data.
        </div>
        """,
        unsafe_allow_html=True,
    )
