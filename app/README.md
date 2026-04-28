# App Streamlit — Passos Mágicos

App de previsão de risco de defasagem da Associação Passos Mágicos.
Datathon FIAP PosTech Fase 5 — *Deep Learning and Unstructured Data*.

## O que o app faz

Tutor da Passos preenche os indicadores de um aluno (INDE, IAN, IDA, IEG,
IAA, IPS, IPV + metadados) e recebe:

1. **Probabilidade de defasagem no próximo ano**
2. **Classificação no semáforo** (🟢 Baixo / 🟢 Observação / 🟡 Atenção / 🔴 Urgente)
3. **Recomendação de ação** com prazo concreto
4. **SHAP local** mostrando os top 5 fatores que empurraram pra risco e os top 5 que protegeram

Inclui também uma aba **"Visão geral do modelo"** com performance, top features,
e um achado bônus de NLP nos relatórios PEDE 2020/21/22.

---

## Rodar localmente

A partir da raiz do repo:

```bash
# 1) Cria/ativa venv
python3.11 -m venv .venv
source .venv/bin/activate

# 2) Instala deps do app (enxuto — sem torch/shap/sentence-transformers)
pip install -r app/requirements.txt

# 3) Roda
streamlit run app/app.py
```

App abre em `http://localhost:8501`.

> **Observação:** o `requirements.txt` da raiz do projeto é mais completo e
> serve pra rodar os notebooks (com torch, sentence-transformers, etc).
> O `app/requirements.txt` é a stack mínima pro app rodar — mais leve pra
> deploy no Streamlit Cloud.

---

## Deploy no Streamlit Community Cloud

### Pré-requisitos

- Conta no GitHub (✅ você já tem)
- Repo público com o código (✅ `MarianaRuther/postech-techchallenge5-datathon`)
- Os artefatos do modelo versionados (✅ `models/*.pkl`, `*.json`, `feature_engineering_stats.json`)
- O `app/requirements.txt` enxuto (✅ pinado nesse repo)

### Passo a passo

**1.** Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com GitHub.

**2.** Clique em **"New app"** (botão azul, canto superior direito).

**3.** Configure:

| Campo | Valor |
|---|---|
| Repository | `MarianaRuther/postech-techchallenge5-datathon` |
| Branch | `main` |
| Main file path | `app/app.py` |
| App URL (custom) | `passos-magicos-risco` *(ou outro slug à sua escolha)* |

**4.** Clique em **"Advanced settings"** e:

- **Python version**: `3.11`
- **Requirements file**: `app/requirements.txt`

**5.** Clique em **"Deploy!"**

O primeiro deploy leva ~5 minutos (instala dependências, faz cache do modelo).
Deploys seguintes (após `git push`) são automáticos e levam 30-60s.

### URL final

Vai ficar algo como:
```
https://passos-magicos-risco.streamlit.app
```

Cole essa URL no README principal do projeto e no roteiro do vídeo.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"

O `app/requirements.txt` não foi encontrado. Em **Advanced settings** do Streamlit
Cloud, garanta que o campo "Requirements file" aponta pra `app/requirements.txt`.
Senão, ele pega o `requirements.txt` da raiz (que é mais pesado e pode demorar muito mais).

### "FileNotFoundError: models/modelo_risco_v1.pkl"

Esse arquivo deve estar versionado no Git. Confirme com:

```bash
git ls-files models/
```

Tem que listar pelo menos: `feature_names.json`, `thresholds.json`,
`modelo_risco_v1.pkl`, `shap_summary.json`, `feature_engineering_stats.json`.

Se faltar algum, faça `git add models/<arquivo>` e `git push`.

### App carrega mas dá "InconsistentVersionWarning" do scikit-learn

O modelo foi treinado em uma versão de sklearn diferente da que o Streamlit Cloud
está usando. Isso é só warning — o modelo continua funcionando. Pra eliminar,
ajuste a versão pinada em `app/requirements.txt`:

```
scikit-learn==1.5.2
```

(Use a mesma versão em que o `modelo_risco_v1.pkl` foi treinado — confere
com `joblib.load(...).__sklearn_version__` no notebook 04.)

### App fica sleeping após 7 dias sem uso

É um comportamento normal do plano gratuito do Streamlit Community Cloud.
Pra acordar, basta abrir a URL — leva ~30s pra subir de novo.

---

## Estrutura

```
app/
├── README.md                  # este arquivo
├── app.py                     # main entrypoint (Streamlit)
├── requirements.txt           # stack enxuta pra deploy
└── .streamlit/
    └── config.toml            # tema (cores Passos)
```

O app **lê** os seguintes arquivos da raiz do projeto (sem modificá-los):

```
models/
├── modelo_risco_v1.pkl              # pipeline LogReg (notebook 04)
├── thresholds.json                  # 3 níveis operacionais (notebook 04)
├── feature_names.json               # ordem das 24 features (notebook 03/04)
├── shap_summary.json                # coeficientes + stats SHAP (notebook 06)
└── feature_engineering_stats.json   # stats por fase pra calcular features derivadas

reports/figures/
├── shap_03_bar_logreg.png           # top features SHAP global
├── shap_02_beeswarm_logreg.png      # distribuição SHAP por feature
├── mod_curvas_comparacao.png        # ROC/PR comparando 3 modelos
├── mod_matriz_confusao.png          # confusão @ Atenção
└── dl_02_clusters_temporal_pede.png # heatmap NLP 2020-2022
```

Caminhos resolvidos via `Path(__file__).parent.parent` — funciona local e em deploy.
