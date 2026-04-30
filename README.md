# Datathon FIAP PosTech — Fase 5

**Modelo preditivo de risco de defasagem escolar para a Associação Passos Mágicos**

[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B?logo=streamlit)](https://postech-techchallenge5-datathon.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)

Projeto desenvolvido para o Datathon da Fase 5 da PosTech FIAP (disciplina de *Deep Learning and Unstructured Data*), em parceria com a [Associação Passos Mágicos](https://passosmagicos.org.br/) — ONG fundada em 1992 por Michelle Flues e Dimetri Ivanoff que atende hoje cerca de 1.213 crianças e jovens em vulnerabilidade social em Embu-Guaçu (SP).

---

## 🎯 O problema

A Passos Mágicos já provou que funciona — 36% dos alunos chegam ao ensino superior, em universidades como Insper, FGV, ESPM. Mas e os outros 64%? Quando começam a se afastar, e como a ONG identifica antes da queda virar evasão?

Esse projeto entrega um modelo preditivo que sinaliza, a partir dos dados de um aluno no ano **t**, a probabilidade dele entrar em defasagem em **t+1** — antes da nota cair, antes da família se desorganizar, antes do aluno desistir.

---

## 🚀 Aplicação live

**App deployed:** [postech-techchallenge5-datathon.streamlit.app](https://postech-techchallenge5-datathon.streamlit.app)

O tutor da Passos preenche os 7 indicadores de um aluno e recebe:
- Probabilidade de defasagem em t+1
- Classificação em 3 níveis: 🟢 Observação, 🟡 Atenção, 🔴 Urgente
- Ação recomendada com prazo concreto
- SHAP local: top 5 fatores empurrando pra risco + top 5 protegendo

> **Aviso:** apps no plano gratuito do Streamlit Community Cloud entram em modo *sleep* após 7 dias sem uso. Pra acordar, basta abrir a URL — leva ~30s pra subir.

---

## 📊 O modelo

| métrica | valor | observação |
|---|---|---|
| **PR-AUC** | 0.79 | métrica principal — defasagem é minoria (~40%) |
| **ROC-AUC** | 0.71 | métrica secundária |
| **Precisão @ Atenção** | 0.70 | dos sinalizados como "atenção", 70% efetivamente entram em risco |
| **Recall @ Atenção** | 0.72 | de todos os alunos que entrarão em risco, captamos 72% |

**Modelo final:** Regressão Logística com pipeline `impute → scale → clf`. Comparei contra XGBoost e LightGBM — empate técnico em desempenho, então escolhi LogReg pela interpretabilidade exata via SHAP linear (em árvores, SHAP é aproximação).

**Validação temporal (anti-leakage):**
- Treino: features de 2022 → target de 2023
- Teste: features de 2023 → target de 2024 (out-of-time, dados que o modelo nunca viu)

**Os 5 sinais que mais importam** (consenso entre Permutation Importance e SHAP):
1. `idade` — aluno mais velho na fase = risco maior
2. `inde_zscore_fase` — INDE relativo aos pares (ranking interno)
3. `IEG` — engajamento (leading indicator)
4. `IPS` — pilar psicossocial
5. `fase do programa` — transições concentram pontos de virada

---

## 📂 Estrutura do projeto

```
DATATHON/
│
├── README.md                       # este arquivo
├── requirements.txt                # stack enxuta (deploy do app)
├── requirements-notebooks.txt      # stack completa (notebooks: torch, transformers, OCR)
├── .gitignore
│
├── data/
│   ├── raw/                        # dataset original (não versionado — privacidade)
│   ├── interim/                    # caches regeneráveis (parquets, OCR, embeddings)
│   └── processed/                  # dados prontos pra modelagem
│
├── docs/                           # materiais de referência (PDFs do datathon, PEDEs)
│
├── notebooks/
│   ├── 00_setup.ipynb              # validação de ambiente
│   ├── 01_eda_limpeza.ipynb        # EDA + tratamento de missing + outliers
│   ├── 02_perguntas_negocio.ipynb  # perguntas 1-6 (IAN, IDA, IEG, IAA, IPS, IPP)
│   ├── 02b_perguntas_negocio_parte2.ipynb  # perguntas 7-11 (IPV, multidim, evasão)
│   ├── 03_feature_engineering.ipynb        # 24 features + target forward-looking
│   ├── 04_modelagem_tabular.ipynb          # LogReg vs XGBoost vs LightGBM + 3 thresholds
│   ├── 05_modelagem_dl.ipynb               # MLP em PyTorch + NLP nos relatórios PEDE
│   └── 06_shap_interpretacao.ipynb         # SHAP global + local + waterfall plots
│
├── models/                         # artefatos do modelo (versionados — leves)
│   ├── modelo_risco_v1.pkl                 # pipeline LogReg final
│   ├── thresholds.json                     # 3 níveis operacionais
│   ├── feature_names.json                  # ordem das 24 features
│   ├── shap_summary.json                   # coeficientes + stats (usado pelo app)
│   └── feature_engineering_stats.json      # stats por fase (features derivadas)
│
├── app/                            # aplicação Streamlit
│   ├── README.md                           # instruções de deploy
│   ├── app.py                              # entrypoint
│   ├── requirements.txt                    # stack mínima pro Streamlit Cloud
│   └── .streamlit/config.toml              # tema (cores Passos)
│
└── reports/
    ├── apresentacao.html                   # apresentação interativa (14 slides)
    ├── apresentacao.pdf                    # versão estática (entregável formal)
    ├── roteiro_video.md                    # roteiro do vídeo de 5min
    ├── figures/                            # ~30 figuras geradas pelos notebooks
    └── preview/                            # PNGs individuais dos slides
```

---

## 🚀 Como rodar

### Pré-requisitos
- Python 3.11
- Git

### Setup local

```bash
# 1. Clonar
git clone https://github.com/MarianaRuther/postech-techchallenge5-datathon.git
cd postech-techchallenge5-datathon

# 2. Ambiente virtual
python3.11 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 3. Dependências (escolha uma)
pip install -r requirements.txt              # mínima — só pra rodar o app
pip install -r requirements-notebooks.txt    # completa — pra rodar notebooks (com torch, transformers)

# 4. Rodar o app local
streamlit run app/app.py
# abre em http://localhost:8501

# 5. Ou rodar os notebooks
# abrir notebooks/00_setup.ipynb no VS Code, selecionar kernel .venv
```

### Replicar o pipeline completo

```bash
# Os notebooks devem ser rodados em ordem:
# 00 → 01 → 02 → 02b → 03 → 04 → 05 → 06
# Cada um gera artefatos (parquets, .pkl, .json) consumidos pelos seguintes
```

---

## 🧠 Decisões técnicas relevantes

### Target forward-looking
"Risco de defasagem em t+1" é definido como qualquer um dos 4 critérios:
1. queda de INDE ≤ -0.5 desvio padrão
2. regressão de fase (ex: Ágata → Quartzo)
3. evasão do programa
4. INDE abaixo do percentil 25 da fase no ano seguinte

Isso evita o problema clássico de modelos descritivos disfarçados de preditivos.

### Por que LogReg em vez de XGBoost?
Empate técnico em PR-AUC (0.79 vs 0.78). Em caso de empate, escolhi interpretabilidade — a Passos vai operar o modelo em decisões reais (coordenadora pedagógica precisa entender *por quê* antes de ligar pra família). SHAP em LogReg é exato; em árvores, é aproximação.

### 3 thresholds operacionais (não binário)
Em vez de "risco sim/não", calibrei thresholds que mapeiam direto pra ações:
- 🟢 **Observação** (prob ≥ 0.139): radar mensal, sem escalar pro tutor
- 🟡 **Atenção** (prob ≥ 0.231): tutor ativa contato em 30 dias — caso de uso mais valioso, antecipa antes do INDE cair
- 🔴 **Urgente** (prob ≥ 0.383): coordenação pedagógica em 7 dias

### Componente NLP (Deep Learning + Unstructured Data)
Análise dos relatórios PEDE 2020/21/22 via:
- OCR com Tesseract (pdfplumber + pdf2image + pytesseract `lang=por`) — 70% das páginas estavam em formato imagem
- Embeddings com sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2`
- KMeans (k=8) + TF-IDF labels com stopwords PT-BR

**Achado bônus:** o cluster de "defasagem moderada/severa" emergiu organicamente em 2022 (0% → 7.9% dos relatórios) — convergência narrativa entre dado estruturado e relato institucional. O modelo não inventou a categoria; responde a uma preocupação que a própria ONG estava articulando.

### Por que MLP perdeu pra LogReg
Treinei um MLP em PyTorch (3 camadas, ~1.300 parâmetros) nos mesmos splits — perdeu por 3 pontos de PR-AUC. Decisão metodológica defensável: com 860 amostras de treino, redes maiores overfittam. Mantive LogReg por (a) performance superior, (b) 27x menos parâmetros, (c) interpretabilidade exata.

---

## 📊 Perguntas de negócio respondidas

Notebook `02_perguntas_negocio.ipynb` (perguntas 1-6) e `02b_perguntas_negocio_parte2.ipynb` (perguntas 7-11):

1. **IAN** — perfil de defasagem e evolução: 33% dos alunos com defasagem moderada+severa, persistente em todas as fases
2. **IDA** — desempenho acadêmico: cresce +1.8 pontos de Quartzo a Topázio
3. **IEG** — engajamento: correlação positiva forte com IDA (0.55) e IPV (0.62)
4. **IAA** — autoavaliação: gap entre IAA e IDA > 1 ponto em 22% dos casos (alunos superestimam)
5. **IPS** — psicossocial: leading indicator — queda em IPS antecede queda em IDA em ~40% dos casos
6. **IPP** — psicopedagógico: confirma defasagem do IAN em 78% dos casos (não é ruído)
7. **IPV** — ponto de virada: IDA + IEG + IPS são os 3 maiores drivers (correlação 0.55+)
8. **Multidimensionalidade** — combinações de IDA alto + IEG alto + IPS alto elevam INDE em +1.5 pontos
9. **Previsão de risco** — modelo final com PR-AUC 0.79 em dados out-of-time
10. **Efetividade do programa** — IDA cresce consistentemente fase a fase (evidência de impacto)
11. **Insights livres** — convergência NLP (item bônus), análise de evasão por pedra, gap autoavaliação vs realidade

---

## 📦 Entregáveis

- ✅ Repositório GitHub com código completo e documentado
- ✅ 8 notebooks Jupyter cobrindo EDA → modelagem → interpretação
- ✅ Modelo preditivo serializado (`models/modelo_risco_v1.pkl`)
- ✅ App Streamlit deployado: [postech-techchallenge5-datathon.streamlit.app](https://postech-techchallenge5-datathon.streamlit.app)
- ✅ Apresentação executiva (`reports/apresentacao.pdf`) — 14 slides com storytelling do problema à solução
- ✅ Roteiro do vídeo de 5min (`reports/roteiro_video.md`)
- ✅ ~30 visualizações (`reports/figures/`)

---

## ⚠️ Limitações conhecidas

- **Tamanho do dataset:** 3.293 observações aluno-ano — limitado pra deep learning. Por isso LogReg vence MLP.
- **Generalização temporal:** modelo validado out-of-time em 2024, mas pode degradar em 2026+ se o perfil dos alunos da Passos mudar significativamente. Recomenda-se retreino anual.
- **Indicadores categóricos sutis:** o modelo captura padrões agregados — eventos pontuais (luto familiar, mudança de escola) não estão nas features e exigem julgamento humano.
- **Cobertura do NLP:** análise restrita aos relatórios PEDE 2020-2022 (PEDE 2023+ ainda não tinha PDF público no momento do projeto).

---

## 👤 Autora

**Mariana Ruther de Araújo**
Project Leader · Data &amp; AI Strategy
PosTech FIAP — Fase 5

---

## 📚 Referências

- [Associação Passos Mágicos](https://passosmagicos.org.br/) — site oficial
- [Quem Somos](https://passosmagicos.org.br/quem-somos/) — história e fundadores
- Relatórios PEDE 2020, 2021, 2022 — disponíveis em `docs/` (uso restrito ao projeto)
