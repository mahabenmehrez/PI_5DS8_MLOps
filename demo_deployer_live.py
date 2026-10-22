"""
Demo du Deployer (P5) branchee sur le VRAI DeployerAgent.
Lancer depuis la RACINE du depot :   streamlit run demo_deployer_live.py
(deployer_live.py doit etre dans le meme dossier.)
"""
import pandas as pd
import streamlit as st

from agents.deployer.llm_decision import make_llm
from agents.deployer.replay import simulated_replay
from deployer_live import ACTIVE, SCENARIOS, LiveHarness

st.set_page_config(page_title="Deployer (P5)", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #F5F8FF 0%, #FFFFFF 40%); }
.block-container { padding-top: 2rem; max-width: 1250px; }
h1 { color: #1E3A8A; font-weight: 800; letter-spacing: -0.5px; }
h2, h3 { color: #1E40AF; }
/* cartes de metriques */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #DBE5FF; border-left: 6px solid #2563EB;
    border-radius: 14px; padding: 14px 18px; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
}
[data-testid="stMetricLabel"] p { color: #64748B; font-weight: 600; }
[data-testid="stMetricValue"] { color: #1E3A8A; font-weight: 800; }
/* boutons */
.stButton > button { border-radius: 12px; font-weight: 700; padding: 0.6rem 1rem; border: 1px solid #BFD0FF; }
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2563EB, #4F46E5); color: #FFFFFF; border: none;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.30);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }
/* bandeaux et tableaux */
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #DBE5FF; border-radius: 14px; }
[data-testid="stDataFrame"] { border: 1px solid #DBE5FF; border-radius: 12px; }
/* barre de progression */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #22C55E, #2563EB); }
</style>
""", unsafe_allow_html=True)

EXPLICATION = {
    "Candidat sain": "Le nouveau modèle est aussi bon que l'ancien : il doit arriver jusqu'à la décision humaine.",
    "Candidat lent (scenario G)": "Le nouveau modèle répond trop lentement : un rollback automatique est attendu.",
    "Candidat en erreur (10 %)": "Le nouveau modèle échoue 1 fois sur 10 : un rollback automatique est attendu.",
    "Candidat moins precis": "Le nouveau modèle se trompe plus que l'ancien : il doit être bloqué dès le shadow.",
}
TRADUCTION = [
    ("shadow predictions", "prédictions shadow"), ("labelled", "avec labels"),
    ("requests", "requêtes"), ("hold requested:", "hold demandé :"),
    ("absolute errors significantly worse than champion", "erreurs de prédiction nettement pires que le champion"),
    ("error rate", "taux d'erreurs"), ("above champion", "au-dessus du champion"),
    ("no rollout in progress", "aucun déploiement en cours"),
    ("no analysis yet", "pas encore d'analyse"),
]
NOMS = {"champion": "Champion", "candidate": "Candidat", "shadow": "Candidat (shadow)"}


def fr(text: str) -> str:
    for en, fra in TRADUCTION:
        text = text.replace(en, fra)
    return text


PROVIDERS = {"Règles seulement (sans LLM)": "off",
             "LLM local (Ollama)": "ollama",
             "LLM via API (Claude)": "api"}


def nouvelle_demo():
    label = st.session_state.get("provider_label", list(PROVIDERS)[0])
    st.session_state.h = LiveHarness(st.session_state.scenario, 42,
                                     llm=make_llm(PROVIDERS[label]),
                                     replay=simulated_replay())


if "h" not in st.session_state:
    st.session_state.h = LiveHarness(list(SCENARIOS)[0], 42)
h: LiveHarness = st.session_state.h

# ------------------------------------------------------------------ en-tête
st.title("Deployer : déploiement progressif d'un nouveau modèle")
st.caption("Agent réel (agents/deployer) ; trafic et autres agents simulés. "
           "Un clic fait avancer le temps simulé.")
sc1, sc2 = st.columns(2)
sc1.selectbox("Scénario", list(SCENARIOS), key="scenario", on_change=nouvelle_demo)
sc2.selectbox("Décision dans la zone grise", list(PROVIDERS), key="provider_label",
              on_change=nouvelle_demo)
st.caption(EXPLICATION[h.scenario])
if h.agent.llm is None:
    st.caption("Mode règles : aucun LLM n'est appelé. Choisis un LLM pour voir l'agent raisonner "
               "dans la zone grise (scénario « Candidat moins précis » par exemple).")

rs = h.release_state()
stage = rs["stage"]
history = h.agent.history
fini = stage == "none" and bool(history) and history[-1]["to_stage"] in ("rolled_back", "100%")

# ------------------------------------------------------------------ une seule action à la fois
col_action, col_reset = st.columns([5, 1])
with col_action:
    if h.agent.pending:
        a, r = st.columns(2)
        if a.button("Approuver le passage à 100 %", type="primary", width="stretch"):
            h.human_decision(True)
            st.rerun()
        if r.button("Refuser", width="stretch"):
            h.human_decision(False)
            st.rerun()
    elif stage in ACTIVE:
        if st.button("Avancer de 30 minutes", type="primary", width="stretch"):
            for _ in range(30):
                if h.release_state()["stage"] not in ACTIVE or h.agent.pending:
                    break
                h.tick()
            h.settle()                      # verification du rollback (2 minutes)
            st.rerun()
    elif not fini:
        if st.button("Démarrer : le Trainer envoie un nouveau modèle", type="primary", width="stretch"):
            h.send_candidate()
            st.rerun()
with col_reset:
    if st.button("Recommencer", type="primary" if fini else "secondary", width="stretch"):
        nouvelle_demo()
        st.rerun()

# ------------------------------------------------------------------ état
niveau, texte = h.status()
{"error": st.error, "success": st.success, "warning": st.warning, "info": st.info}[niveau](fr(texte))

if h.agent.pending:
    etape = "Attente de l'humain"
elif fini:
    etape = "100 % (promu)" if history[-1]["to_stage"] == "100%" else "Rollback"
else:
    etape = {"none": "Aucun candidat", "shadow": "Shadow", "10%": "10 %", "50%": "50 %"}[stage]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Étape", etape)
m2.metric("Part du nouveau modèle", f"{rs['candidate_share'] * 100:.0f} %")
m3.metric("Temps simulé", f"{h.minute} min")
m4.metric("Ancien → nouveau", f"v{rs['champion_version']} → v{rs['candidate_version']}"
          if rs["candidate_version"] else f"v{rs['champion_version']}")

fait = {"shadow": 1, "10%": 2, "50%": 3}.get(stage, 0)
if h.agent.pending:
    fait = 4
elif fini and history[-1]["to_stage"] == "100%":
    fait = 5
st.progress(fait / 5, text="Shadow  →  10 %  →  50 %  →  approbation humaine  →  100 %")

# ------------------------------------------------------------------ pourquoi l'agent agit ainsi
g = h.gate()
if g:
    with st.expander("Pourquoi l'agent agit ainsi", expanded=True):
        st.write(f"Temps passé dans l'étape : **{g['minutes_in_stage']:.0f} min**")
        for w in g["gate"]["waiting"]:
            st.write("⏳ En attente :", fr(w))
        for b in g["gate"]["bad"]:
            st.write("⛔ Problème :", fr(b))
        if g["breach"]:
            st.error(f"Limite dure dépassée : {g['breach']['metric']} = {g['breach']['value']:.3g} "
                     f"(limite {g['breach']['limit']}) → rollback d'urgence, sans LLM ni approbation")
        if g["gate"]["ok"]:
            st.success("Toutes les portes sont passées : l'agent peut avancer.")
        rr = h.agent.replay_result
        if rr:
            st.write("🧪 Dry run traffic_replay 3× (estimation simulée) :", rr.get("detail", ""))

# ------------------------------------------------------------------ comparaison
gauche, droite = st.columns(2)
with gauche:
    st.subheader("Ancien modèle contre nouveau modèle")
    stats = h.window_stats()
    if stats.empty:
        st.write("Pas encore de données.")
    else:
        stats = stats.drop(columns=["n labels"]).replace({"variante": NOMS})
        stats = stats.rename(columns={"variante": "Modèle", "requetes": "Requêtes",
                                      "taux d'erreurs": "Erreurs", "MAE (labels recus)": "Erreur de prédiction"})
        st.dataframe(stats, hide_index=True, width="stretch")

    st.subheader("Latence p95 par minute (ms)")
    courbe = h.latency_chart()
    if courbe.empty:
        st.write("Clique sur « Avancer » pour voir la courbe.")
    else:
        courbe = courbe.rename(columns={"champion": "Champion", "candidat": "Candidat",
                                        "limite": "Limite (300 ms)"})
        couleurs = {"Champion": "#2563EB", "Candidat": "#F97316", "Limite (300 ms)": "#DC2626"}
        st.line_chart(courbe, color=[couleurs[c] for c in courbe.columns])

with droite:
    st.subheader("Messages échangés avec les autres agents")
    msgs = [m for m in h.log if m["kind"] != "action.hold"]
    if msgs:
        df = pd.DataFrame(msgs)
        df["De → à"] = df["sender"] + " → " + df["to"]
        df = df.rename(columns={"minute": "Min", "kind": "Message", "summary": "Résumé"})
        st.dataframe(df[["Min", "Message", "De → à", "Résumé"]].iloc[::-1],
                     hide_index=True, width="stretch", height=480)
    else:
        st.write("Aucun message pour l'instant.")

# ------------------------------------------------------------------ mémoire
with st.expander("Cahier de souvenirs de l'agent (experience log)"):
    mem = h.memory_rows()
    if mem.empty:
        st.write("Rien de mémorisé pour l'instant : l'agent écrit après chaque action (avance, rollback, demande 100 %).")
    else:
        st.dataframe(mem.rename(columns={"time": "Heure", "action": "Action",
                                         "outcome": "Résultat", "problem_text": "Raison"}),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ LLM
with st.expander("Décisions du LLM (prompt, réponse, validation)"):
    calls = getattr(h.agent.llm, "calls", [])
    if h.agent.llm is None:
        st.write("Mode règles : le LLM est désactivé.")
    elif not calls:
        st.write("Le LLM n'a pas encore été appelé : il n'intervient que dans la zone grise "
                 "(une porte franchement négative, mais aucune limite dure dépassée).")
    else:
        st.write(f"Modèle : **{h.agent.llm.provider} / {h.agent.llm.model}**. "
                 f"{len(calls)} appel(s). Dernière décision retenue : "
                 f"**{(h.agent.last_llm or {}).get('action', 'aucune (repli sur les règles)')}**")
        st.dataframe(pd.DataFrame([{
            "Essai": c["attempt"], "Valide": "oui" if c["valid"] else "non",
            "Erreur": c["error"] or "", "Latence (s)": c["latency_s"], "Jetons": c["tokens"],
            "Réponse": (c["raw"] or "")[:200]} for c in calls[::-1]]),
            hide_index=True, width="stretch")
        with st.expander("Dernier prompt envoyé (situation en chiffres)"):
            st.code(calls[-1]["prompt_user"], language="json")
