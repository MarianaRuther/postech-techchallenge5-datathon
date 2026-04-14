# Datathon FIAP PosTech — Fase 5

**Modelo preditivo de risco de defasagem escolar para a Associação Passos Mágicos**

Projeto desenvolvido para o Datathon da Fase 5 da PosTech FIAP (disciplina de Deep Learning and Unstructured Data), em parceria com a [Associação Passos Mágicos](https://passosmagicos.org.br/) — ONG com 35 anos de atuação que usa educação para transformar a vida de crianças e jovens em vulnerabilidade social.

---

## 🎯 Objetivo

Desenvolver uma solução completa de **data analytics + modelo preditivo + aplicação web** que:

1. Responda a perguntas-chave de negócio da Passos Mágicos a partir dos indicadores do PEDE (INDE, IAN, IDA, IEG, IAA, IPS, IPP, IPV).
2. Construa um **modelo de machine learning** capaz de identificar precocemente alunos em risco de defasagem.
3. Entregue uma **aplicação Streamlit** para uso direto pela equipe pedagógica da ONG.
4. Comunique insights em formato executivo (storytelling + vídeo).

---

## 📂 Estrutura do projeto

```
DATATHON/
│
├── README.md                   # este arquivo
├── requirements.txt            # dependências Python
├── .gitignore
│
├── data/
│   ├── raw/                    # dataset original do datathon (versionado)
│   ├── interim/                # dados após limpeza (não versionado)
│   └── processed/              # dados prontos pra modelagem (não versionado)
│
├── docs/                       # materiais de referência
│   ├── POSTECH - Datathon - Fase 5.pdf
│   ├── Dicionário Dados Datathon.pdf
│   ├── PEDE_ Pontos importantes.docx
│   ├── Links adicionais da passos.docx
│   ├── desvendando_passos.pdf
│   ├── Relatório PEDE2020.pdf
│   ├── Relatório PEDE2021.pdf
│   └── Relatorio PEDE2022.pdf
│
├── notebooks/                  # análises e modelagem
│   ├── 00_setup.ipynb                   # validação de ambiente
│   ├── 01_eda_limpeza.ipynb             # EDA + tratamento de dados
│   ├── 02_perguntas_negocio.ipynb       # análise das 11 perguntas
│   ├── 03_feature_engineering.ipynb     # criação de features + target
│   ├── 04_modelagem_tabular.ipynb       # LR, XGBoost, LightGBM
│   ├── 05_modelagem_dl.ipynb            # componente Deep Learning + NLP
│   └── 06_shap_interpretacao.ipynb      # explicabilidade
│
├── models/                     # modelos treinados (.pkl)
│
├── app/                        # aplicação Streamlit
│   └── streamlit_app.py
│
└── reports/                    # apresentação executiva + roteiro do vídeo
    ├── figures/
    ├── apresentacao_executiva.pdf
    └── roteiro_video.md
```

---

## 🚀 Setup do ambiente

### Pré-requisitos
- Python 3.11 (recomendado)
- Git
- VS Code com extensões **Python** e **Jupyter** da Microsoft

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/MarianaRuther/postech-techchallenge5-datathon.git
cd postech-techchallenge5-datathon

# 2. Criar ambiente virtual
python3.11 -m venv .venv

# 3. Ativar ambiente virtual
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 4. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 5. Validar o ambiente
# Abrir notebooks/00_setup.ipynb no VS Code
# Selecionar o kernel .venv e executar todas as células
```

---

## 🧠 Abordagem técnica

### Target do modelo: risco *forward-looking*
A variável de "risco de defasagem" é definida de forma **preditiva**, não descritiva: o modelo estima, a partir dos dados de um aluno no ano **t**, a probabilidade de ele estar em risco no ano **t+1** (queda significativa no INDE, regressão de fase, ou posicionamento no quartil inferior).

### Validação temporal (anti-leakage)
- **Treino:** features de 2022 → target de 2023
- **Teste:** features de 2023 → target de 2024

### Modelos comparados
1. **Regressão Logística** — baseline interpretável
2. **XGBoost** — benchmark tabular
3. **LightGBM** — alternativa rápida e robusta
4. **Componente Deep Learning + NLP** — usando embeddings para enriquecer features tabulares com contexto não-estruturado (detalhes em `notebooks/05_modelagem_dl.ipynb`)

### Explicabilidade
SHAP values para tornar as decisões do modelo auditáveis pela equipe pedagógica da Passos Mágicos — critério inegociável para uma ferramenta de apoio à decisão em contexto educacional.

---

## 📊 Perguntas de negócio respondidas

1. **IAN** — Perfil de defasagem e evolução ao longo do ano
2. **IDA** — Desempenho acadêmico: melhora, estagna ou cai?
3. **IEG** — Relação entre engajamento, IDA e IPV
4. **IAA** — Autoavaliação vs desempenho real (gap analysis)
5. **IPS** — Padrões psicossociais que antecedem quedas
6. **IPP** — Confirmação/contradição do IAN pelo psicopedagógico
7. **IPV** — Comportamentos que mais influenciam o Ponto de Virada
8. **Multidimensionalidade** — Combinações que elevam o INDE
9. **Previsão de risco** — Modelo preditivo (coração técnico)
10. **Efetividade do programa** — Melhora ao longo das fases Quartzo → Topázio?
11. **Insights livres**

---

## 🗺️ Roadmap de desenvolvimento

- [x] **Etapa 1** — Estruturação do projeto
- [ ] **Etapa 2** — EDA e limpeza
- [ ] **Etapa 3** — Análise das 11 perguntas de negócio
- [ ] **Etapa 4** — Feature engineering e modelagem preditiva
- [ ] **Etapa 5** — App Streamlit
- [ ] **Etapa 6** — Storytelling executivo e roteiro do vídeo

---

## 👤 Autora

**Mariana Ruther** — PosTech FIAP Fase 5

---

## 📚 Referências

- [Associação Passos Mágicos](https://passosmagicos.org.br/)
- [Relatórios de atividades](https://passosmagicos.org.br/impacto-e-transparencia/)
- Relatórios PEDE 2020, 2021 e 2022 (disponíveis em `docs/`)
