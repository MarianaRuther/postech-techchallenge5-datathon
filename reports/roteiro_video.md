# Roteiro do vídeo — 5 minutos

> Datathon FIAP PosTech Fase 5 · Passos Mágicos
> Mariana Ruther de Araújo

**Como usar:** falar olhando pra câmera nos blocos de "fala"; o que está em `[tela: ...]` é o que aparece no compartilhamento de tela / corte. Tom informal, primeira pessoa, sem ler o roteiro literalmente — usar como guia.

---

## bloco 1 · o problema (0:00 → 0:30)

**[tela: slide 1 — capa]**

Oi. Sou a Mari, project leader de Data &amp; AI, e hoje vou te mostrar como construí um modelo preditivo pra Associação Passos Mágicos, uma ONG que há 35 anos transforma vidas pela educação em Embu-Guaçu.

**[tela: slide 2 — stats da Passos]**

A Passos já tem evidência sólida de impacto: hoje atende 1.213 alunos, eleva renda familiar em 45% e leva 36% deles pro ensino superior — Insper, FGV, ESPM. Mas a pergunta que ficou é: **e os outros 64%? quando começam a se afastar?** É essa parte que o modelo resolve.

---

## bloco 2 · os dados (0:30 → 1:30)

**[tela: slide 3 — pergunta + dados]**

A ONG acompanha cada aluno através de 7 indicadores no relatório PEDE: INDE como índice global, e mais 6 sub-indicadores cobrindo desempenho acadêmico, engajamento, autoavaliação, psicossocial, psicopedagógico e o ponto de virada. Os alunos progridem em 4 fases: Quartzo, Ágata, Ametista, Topázio.

Tive 3 anos de dados — 2022, 23 e 24 — totalizando 3.293 observações aluno-ano. Decidi cedo que faria validação **out-of-time**: treino em 2022→2023 e teste em 2024. Isso elimina leakage temporal e simula honestamente o uso real do modelo: em janeiro, a Passos quer prever quem vai entrar em risco até dezembro.

**[tela: slide 5 — IAN persistência]**

O primeiro achado importante: 1 em cada 3 alunos chega defasado e *continua* defasado, mesmo nas fases mais avançadas. Não é falha da Passos — é dívida acumulada da escola pública. Mas o programa principal não absorve sozinho. Esses são os candidatos naturais a evasão.

---

## bloco 3 · o modelo (1:30 → 3:00)

**[tela: slide 8 — comparação 3 modelos]**

Defini o target como "risco de defasagem em t+1" — combinação de queda de INDE acima de 0.5 desvio, regressão de pedra, evasão ou ficar abaixo do percentil 25 da fase. Comparei 3 modelos: regressão logística, XGBoost e LightGBM.

A LogReg ganhou com PR-AUC de 0.79. E aqui vai a parte que eu defendo metodologicamente: a diferença pro XGBoost foi 1 ponto — empate técnico. Em caso de empate, escolho **interpretabilidade**. A coordenadora pedagógica vai usar isso pra decidir se liga pra família de um aluno — ela precisa entender *por quê* antes de agir. SHAP em LogReg é exato; em árvores, é aproximação.

**[tela: slide 9 — top 5 SHAP]**

Pra confiar no modelo, validei interpretabilidade com duas técnicas independentes: permutation importance e SHAP. As duas concordam no top 5: idade, INDE relativo à fase, IEG (engajamento), IPS (psicossocial) e a fase do programa. Quando duas técnicas diferentes apontam pros mesmos sinais, é porque o sinal é real.

**[tela: slide 7 — NLP / convergência]**

Como bônus, fiz uma análise não-estruturada nos relatórios PEDE de 2020 a 2022 — 95 mil palavras extraídas via OCR (Tesseract), embeddings com sentence-transformers multilingual e clustering com KMeans. Achei o seguinte: o tema *"defasagem moderada e severa"* emergiu do nada em 2022 — saiu de 0% pra quase 8% dos relatórios. Ou seja, o modelo não inventou uma categoria — responde a uma preocupação que a própria ONG estava começando a articular institucionalmente.

---

## bloco 4 · a operação (3:00 → 4:00)

**[tela: slide 10 — 3 níveis operacionais]**

Em vez de uma decisão binária "risco sim/não", calibrei 3 thresholds que mapeiam direto pra ações da ONG: Observação acompanha no radar mensal; Atenção dispara contato do tutor em 30 dias; Urgente vai pra coordenação pedagógica em 7 dias. O nível mais valioso é o Atenção — antecipa risco *antes* do INDE cair. É aí que a intervenção faz mais diferença.

**[tela: app Streamlit em produção, demonstrando preenchimento]**

E pra tirar o modelo do código e colocar na operação, fiz o app Streamlit. Tutor preenche os 7 indicadores, recebe a probabilidade, o nível, a ação recomendada e o SHAP local — top 5 fatores empurrando pra risco e top 5 protegendo.

**[tela: SHAP local mostrando explicação de um aluno]**

Esse aluno aqui, por exemplo: na superfície parece ótimo — Topázio, INDE 8.10. Mas o modelo classificou Atenção. Por quê? Idade alta pra fase, ranking interno baixo, não tem nota de inglês. Sinais sutis que o tutor consegue agir antes da nota cair. É exatamente o caso de uso mais valioso.

---

## bloco 5 · impacto + fechamento (4:00 → 5:00)

**[tela: slide 13 — recomendações]**

Pra fechar, 4 recomendações pra Passos, em ordem de impacto vs esforço. **Um:** rodar o app semanalmente nas reuniões pedagógicas, focando no Atenção — custo zero, ganho de 4 a 12 semanas de antecipação. **Dois:** protocolo especial nas transições de fase, onde a evasão concentra. **Três:** trilha paralela de reforço pra defasagem severa em matemática e português — temas que emergiram organicamente no NLP. **Quatro:** virar "defasagem moderada+severa" em métrica institucional com meta anual.

**[tela: slide 14 — encerramento]**

O modelo tem PR-AUC 0.79, opera em 3 níveis com ações concretas, e é 100% explicável. Toda predição pode ser auditada feature por feature — a Passos não vai operar uma caixa-preta.

O código, os 8 notebooks, o app deployado e essa apresentação estão todos no GitHub. Obrigada — e fico aberta pra próximos passos com a equipe da Passos.

---

## notas pra gravação

- **timing**: pratica 1-2 vezes pra calibrar; fala natural geralmente fica 10-15% mais longa que leitura mental
- **se passar de 5min**: corta o bloco do NLP (slide 7) — é bônus, não core
- **se sobrar tempo**: amplia recomendações com casos concretos (ex: "se a Passos rodar isso semanalmente em 2026, vejo 30% de redução de evasão por antecipação")
- **olhar pra câmera** nos blocos com fala, **olhar pra tela** ao mostrar slide/app
- **tom**: peer-to-peer, como se estivesse explicando pra uma colega de outro time. Sem ler script literal
- **ferramenta de gravação sugerida**: Loom, OBS ou QuickTime + screen recording. Edição mínima — corte de pausas longas, opcional adicionar legenda
