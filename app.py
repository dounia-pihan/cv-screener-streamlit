import os, re, io, zipfile, unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import streamlit as st
import pandas as pd
import pdfplumber
import altair as alt

# Theme
PRIMARY = "#F97316"
ACCENT  = "#EA580C"
RED     = "#FFD6D6"

st.set_page_config(page_title="Tri & Scoring de CV", layout="wide")
PRIMARY = "#F97316"
ACCENT  = "#EA580C"
MUTED   = "#6B7280"
BG_SOFT = "#FFF7ED"
RED     = "#FECACA"
GREEN = "#DCFCE7"

st.markdown(f"""
<style>
:root {{
  --primary: {PRIMARY};
  --accent: {ACCENT};
  --muted: {MUTED};
  --bgsoft: {BG_SOFT};
}}

h1 {{ text-align:center; letter-spacing:.2px; margin-bottom:.25rem; }}

.small-subtitle {{
  display:block; text-align:center; color:var(--muted);
  margin-bottom:1.1rem; font-size:0.95rem;
}}

.section-title {{
  font-weight:700; border-left:6px solid var(--primary);
  padding-left:.6rem; margin:.25rem 0 .6rem;
}}

.param-box {{
  background-color: var(--bgsoft);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 0 12px rgba(0,0,0,0.05);
  border: 1px solid rgba(0,0,0,0.05);
}}

.stButton>button {{
  background:var(--primary); border:0; color:white; font-weight:600;
  border-radius:10px; padding:.5rem 1rem;
}}
.stButton>button:hover {{ background:var(--accent); }}
input, textarea, select {{ accent-color:var(--primary); }}
.stDataFrame, .css-1v0mbdj {{ border-radius:12px; overflow:hidden; }}

/* Espace entre colonnes */
[data-testid="column"] {{
  padding-right: 4rem;
}}

/* ▼ Réduction de la taille de police à l’intérieur des widgets ▼ */
input[type="text"], textarea, .stTextInput input, .stNumberInput input {{
    font-size: 0.9rem !important;
    height: 3rem !important;
}}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] span {{
    font-size: 0.9rem !important;
}}

div[data-baseweb="tag"] {{
    font-size: 0.9rem !important;
    padding: 0.2rem 0.8rem !important;
}}

[data-testid="stSliderLabel"], [data-testid="stThumbValue"], .stSlider {{
    font-size: 0.8rem !important;
}}

label, .stMarkdown p {{
    font-size: 0.9rem !important;
}}

/* Widgets */
.stTextInput input, .stNumberInput input {{ width: 100% !important; height: 2.2rem !important; }}
div[data-baseweb="select"]                 {{ width: 100% !important; }}
div[data-baseweb="select"] > div           {{ min-height: 2.4rem !important; }}
div[data-baseweb="select"] input           {{ height: 2.0rem !important; }}

/* (multiselect) */
div[data-baseweb="tag"] {{ font-size: 0.85rem !important; padding: .15rem .45rem !important; }}

ul[role="listbox"] li {{ padding-top: .25rem !important; padding-bottom: .25rem !important; }}

[data-testid="column"] {{ padding-right: 1.1rem; }}

/* Style colonne Paramètres */
[data-testid="column"]:first-child > div {{
  background-color: #FFF7ED; /* Utilise la couleur BG_SOFT */
  padding: 1rem; /* Ajoute un peu de padding pour l'esthétique */
  border-radius: 12px; /* Arrondit les coins */
  box-shadow: 0 0 12px rgba(0,0,0,0.05); /* Ajoute une légère ombre */
  border: 1px solid rgba(0,0,0,0.05); /* Ajoute une bordure légère */
}}

/* widgets à l'intérieur */
[data-testid="column"]:first-child input[type="text"],
[data-testid="column"]:first-child textarea,
[data-testid="column"]:first-child .stTextInput input,
[data-testid="column"]:first-child .stNumberInput input,
[data-testid="column"]:first-child div[data-baseweb="select"] > div,
[data-testid="column"]:first-child div[data-baseweb="tag"] {{
    background-color: white !important;
}}


</style>
""", unsafe_allow_html=True)



def strip_accents(s: str) -> str:
    if not isinstance(s, str): return ""
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_text(s: str) -> str:
    s = (s or "").replace("\x00", " ")
    s = strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_csv(text: Optional[str]) -> List[str]:
    if not text: return []
    items = [normalize_text(x) for x in text.split(",")]
    return [x for x in items if x]

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def read_pdf_text(path: str) -> str:
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try: chunks.append(page.extract_text() or "")
            except Exception: continue
    return "\n".join(chunks)


def present(txt_norm: str, term: str) -> bool:
    t = normalize_text(term)
    pattern = re.escape(t)
    pattern = pattern.replace(r"\ ", r"[\s\-_]*").replace(r"\-", r"[\s\-_]*").replace("_", r"[\s\-_]*")
    return re.search(pattern, txt_norm) is not None

def any_present(txt_norm: str, words: List[str]) -> List[str]:
    return [w for w in words if present(txt_norm, w)]

def all_present(txt_norm: str, words: List[str]) -> List[str]:
    return [w for w in words if present(txt_norm, w)]

# Extractors
CONTRACT_VARIANTS = ["cdi","cdd","alternance","apprentissage","stage","freelance","independant","indépendant","mission"]
WORKTIME_VARIANTS = ["Temps plein","Temps partiel","Mi-temps","mi temps","80%","50%"]

LANG_VARIANTS = ["francais","français","anglais","bilingue","courant","intermediaire","toeic","ielts","toefl"]
SKILLS_DEFAULTS = ["power bi", "POWER BI", "PowerBI", "dax","power query","excel","python","pandas","numpy","sql","postgresql","tableau",
                   "scikit-learn","git","github","oracle","powerpoint"]

CITY_DEFAULTS = ["paris","ile de france","lyon","marseille","bordeaux","lille","nantes","rennes","toulouse",
                 "montpellier","nice","strasbourg","bruxelles","genève","lausanne","luxembourg"]

EDU_LEVELS = [
    ("Doctorat", ["doctorat","phd","docteur"]),
    ("Master/Mastère", ["master","mastère","bac+5","msc","ms"]),
    ("Licence/Bachelor", ["licence","bachelor","bac+3","but3"]),
    ("BTS/DUT", ["bts","dut","but"]),
    ("Baccalauréat", ["bac"])
]

EXP_PATTERNS = [
    r"(\d+)\s*ans?\s+d[’'\s]*experience",
    r"experience\s+de\s+(\d+)\s*ans?",
    r"(\d+)\s*ans?\s+d[’'\s]*exp",
]

@dataclass
class TriState:
    status: str
    evidence: Optional[str]
    confidence: float

def detect_contract(txt: str, preferences: List[str]) -> Tuple[Optional[str], str]:
    found = None
    for var in CONTRACT_VARIANTS:
        if present(txt, var): found = var; break
    if not preferences:
        return found, ("match" if found else "unknown")
    if found is None: return None, "unknown"
    return found, ("match" if found in preferences else "mismatch")

def detect_worktime(txt: str, preferences: List[str]) -> TriState:
    hits = [v for v in WORKTIME_VARIANTS if present(txt, v)]
    if not hits: return TriState("unknown", None, 0.0)
    status = "match" if (not preferences or any(h in preferences for h in hits)) else "mismatch"
    return TriState(status, hits[0], 1.0)

def detect_languages(txt: str, expected: List[str]) -> List[str]:
    variants = list(set([normalize_text(x) for x in (expected or [])] + LANG_VARIANTS))
    return sorted({v for v in variants if present(txt, v)})

def detect_skills(txt: str, expected: List[str]) -> List[str]:
    variants = list(set([normalize_text(x) for x in (expected or [])] + SKILLS_DEFAULTS))
    return sorted({v for v in variants if present(txt, v)})

def estimate_experience_years(txt: str) -> Optional[int]:
    for pat in EXP_PATTERNS:
        m = re.search(pat, txt)
        if m:
            try:
                y = int(m.group(1))
                if 0 <= y < 50: return y
            except Exception: pass
    return None

def detect_city(txt: str, preferred: List[str]) -> Tuple[Optional[str], str]:
    candidates = list(set(CITY_DEFAULTS + preferred))
    for city in candidates:
        if not city: continue
        if present(txt, city):
            return city, ("match" if (preferred and normalize_text(city) in preferred) else "unknown")
    return None, "unknown"

def detect_education(txt: str) -> Optional[str]:
    for level, keys in EDU_LEVELS:
        for k in keys:
            if present(txt, k):
                return level
    return None

# Scoring
@dataclass
class ScoringConfig:
    obligatoires: List[str]
    must_have_specific: List[str]
    cat_redhibitoire: Optional[str]
    type_contrat_pref: List[str]
    temps_pref: List[str]
    exp_min: int
    edu_pref: List[str]
    villes_pref: List[str]
    langues: List[str]
    competences: List[str]
    permis: List[str]
    seuil: int
    top_n: int
    optionnels: List[str]

def compute_score(txt_norm: str, cfg: ScoringConfig) -> Tuple[int, bool, List[str], Dict[str, object]]:
    motifs, score = [], 0
    rejected = False
    details = {}

    #Mots-clés
    missing_required = [w for w in cfg.obligatoires if not present(txt_norm, w)]
    if missing_required:
        motifs.append("Mots-clés obligatoires manquants : " + ", ".join(missing_required))
        rejected = True # Reject if obligatory keywords are missing

    #mots-clés rédhibitoires
    missing_must = [w for w in (cfg.must_have_specific or []) if not present(txt_norm, w)]
    if missing_must:
        motifs.append("Mots-clés rédhibitoires absents : " + ", ".join(missing_must))
        rejected = True # Reject if must-have specific keywords are missing

    # Catégorie rédhibitoire
    if cfg.cat_redhibitoire:
        if cfg.cat_redhibitoire == 'ville':
            ville_detectee, ville_status = detect_city(txt_norm, cfg.villes_pref)
            details["ville_detectee"] = ville_detectee
            details["ville_status"] = ville_status
            if ville_status != "match":
                motifs.append(f"Ville non préférée : {ville_detectee or 'non détectée'}")
                rejected = True
        elif cfg.cat_redhibitoire == 'formation':
            edu = detect_education(txt_norm)
            edu_status = "unknown"
            if cfg.edu_pref:
                if edu and edu in cfg.edu_pref:
                    edu_status = "match"
                elif edu:
                    edu_status = "mismatch"
            details["formation_detectee"] = edu
            details["formation_status"] = edu_status
            if edu_status != "match":
                 motifs.append(f"Formation non préférée : {edu or 'non détectée'}")
                 rejected = True
        elif cfg.cat_redhibitoire == 'type de contrat':
             contrat_detecte, contrat_status = detect_contract(txt_norm, cfg.type_contrat_pref)
             details["type_contrat_detecte"] = contrat_detecte
             details["type_contrat_status"] = contrat_status
             if contrat_status != "match":
                 motifs.append(f"Type de contrat non préféré : {contrat_detecte or 'non détecté'}")
                 rejected = True
        elif cfg.cat_redhibitoire == 'temps de travail':
            temps_res = detect_worktime(txt_norm, cfg.temps_pref)
            details["temps_travail_status"] = temps_res.status
            if temps_res.status != "match":
                motifs.append(f"Temps de travail non préféré : {temps_res.evidence or 'non détecté'}")
                rejected = True
        elif cfg.cat_redhibitoire == 'expérience min':
            exp_years = estimate_experience_years(txt_norm)
            details["experience_estimee_ans"] = exp_years
            if exp_years is None or exp_years < cfg.exp_min:
                 motifs.append(f"Expérience inférieure au minimum requis ({cfg.exp_min} ans): {exp_years or 'non détectée'} ans")
                 rejected = True
        elif cfg.cat_redhibitoire == 'langues':
             langues_trouvees = detect_languages(txt_norm, cfg.langues)
             details["langues_detectees"] = ", ".join(langues_trouvees)
             if not any(l in langues_trouvees for l in cfg.langues): # Check if at least one preferred language is found
                 motifs.append(f"Aucune des langues préférées trouvée ({', '.join(cfg.langues)})")
                 rejected = True
        elif cfg.cat_redhibitoire == 'compétences':
             skills_trouves = detect_skills(txt_norm, cfg.competences)
             details["competences_detectees"] = ", ".join(skills_trouves)
             if not any(s in skills_trouves for s in cfg.competences): # Check if at least one preferred skill is found
                 motifs.append(f"Aucune des compétences préférées trouvée ({', '.join(cfg.competences)})")
                 rejected = True


    if rejected:
        return 0, False, motifs, details

    #Bonus
    optionnels_list = cfg.optionnels if cfg.optionnels is not None else [] # Ensure it's a list
    optionnels_trouves = [w for w in optionnels_list if present(txt_norm, w)]
    score += min(len(optionnels_trouves), 10)
    details["mots_optionnels_trouves"] = ", ".join(optionnels_trouves)

    if cfg.cat_redhibitoire != 'type de contrat':
        contrat_detecte, contrat_status = detect_contract(txt_norm, cfg.type_contrat_pref)
        details["type_contrat_detecte"] = contrat_detecte
        details["type_contrat_status"] = contrat_status
        if cfg.type_contrat_pref:
            score += 1 if contrat_status == "match" else (-1 if contrat_status == "mismatch" else 0)

    if cfg.cat_redhibitoire != 'temps de travail':
        temps_res = detect_worktime(txt_norm, cfg.temps_pref)
        details["temps_travail_status"] = temps_res.status
        score += 1 if temps_res.status == "match" else (-1 if temps_res.status == "mismatch" else 0)


    if cfg.cat_redhibitoire != 'expérience min':
        exp_years = estimate_experience_years(txt_norm)
        details["experience_estimee_ans"] = exp_years
        if exp_years is not None and exp_years >= cfg.exp_min: score += 1

    if not cfg.cat_redhibitoire == 'formation':
        edu = detect_education(txt_norm)
        edu_status = "unknown"
        if cfg.edu_pref:
            if edu and edu in cfg.edu_pref:
                score += 1; edu_status = "match"
            elif edu:
                edu_status = "mismatch"
        details["formation_detectee"] = edu
        details["formation_status"] = edu_status


    if not cfg.cat_redhibitoire == 'ville':
        ville_detectee, ville_status = detect_city(txt_norm, cfg.villes_pref)
        details["ville_detectee"] = ville_detectee
        details["ville_status"] = ville_status
        if ville_status == "match": score += 1

    if cfg.cat_redhibitoire != 'langues':
        langues_trouvees = detect_languages(txt_norm, cfg.langues)
        details["langues_detectees"] = ", ".join(langues_trouvees)
        score += min(len(langues_trouvees), 2)

    if cfg.cat_redhibitoire != 'compétences':
        skills_trouves = detect_skills(txt_norm, cfg.competences)
        details["competences_detectees"] = ", ".join(skills_trouves)
        score += min(len(skills_trouves), 5)


    retenu = (not rejected) and (score >= cfg.seuil)

    return score, retenu, motifs, details

def process_pdfs(files: List[str], cfg: ScoringConfig) -> Dict[str, object]:
    results = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            raw = read_pdf_text(fpath)
            txt = normalize_text(raw)
            score, retenu, motifs, details = compute_score(txt, cfg)
            row = {"fichier": fname, "score_total": score, "retenu": "Oui" if retenu else "Non",
                   "motifs_rejet": "; ".join(motifs) if motifs else ""}
            row.update(details)
            results.append(row)
        except Exception as e:
            results.append({"fichier": fname, "score_total": -999, "retenu": "Non", "motifs_rejet": f"Erreur: {e}"})
    results.sort(key=lambda r: r.get("score_total", 0), reverse=True)
    for i, r in enumerate(results):
        r["top_rank"] = i + 1
    return {"df": pd.DataFrame(results), "results": results}

def build_zip_of_retained(results: List[Dict[str, object]], files: List[str], top_n: int) -> bytes:
    names = [r["fichier"] for r in results if r.get("retenu") == "Oui"]
    if top_n: names = names[:top_n]
    selected = [f for f in files if os.path.basename(f) in names]
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in selected: zf.write(f, arcname=os.path.basename(f))
    mem.seek(0)
    return mem.getvalue()

def df_to_xlsx_bytes(df: pd.DataFrame) -> Optional[bytes]:
    """Write df to XLSX in memory. Tries xlsxwriter first, then openpyxl."""
    try:
        import xlsxwriter  # noqa: F401
        b = io.BytesIO()
        with pd.ExcelWriter(b, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        return b.getvalue()
    except Exception:
        try:
            import openpyxl  # noqa: F401
            b = io.BytesIO()
            with pd.ExcelWriter(b, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            return b.getvalue()
        except Exception:
            return None

# UI
st.title("Tri & Scoring de CV")
st.markdown('<span class="small-subtitle">Analyse PDF → extraction → score → export.</span>', unsafe_allow_html=True)

left, right = st.columns([1.0, 1.3])

with left:
    st.markdown('<div class="section-title">| Paramètres</div>', unsafe_allow_html=True)

    mode = st.radio("Source des CV", ["Télécharger des PDF", "Charger depuis un dossier"])

    obligatoires = st.text_input("Mots-clés (obligatoires, séparés par des virgules)*", "power bi, python",
                                 help="Critères indispensables : si l’un des mots-clés manque, le CV est écarté (ex. python, power bi).")

    optionnels  = st.text_input("Compétences", "excel, sql, pandas, dax",
                                help="Must-have supplémentaire : si l'une des compétences est présente, le CV obtient des pts bonus (non éliminatoire).")

    # Contrats et temps de travail côte à côte
    c1, c2 = st.columns(2)
    with c1:
        type_contrat_pref = st.multiselect(
            "Contrats acceptés",
            ["CDI", "CDD", "Alternance", "Stage", "Freelance"],
            ["CDI", "Alternance"]
        )
    with c2:
        temps_pref = st.multiselect(
            "Temps de travail",
            ["Temps plein", "Temps partiel"],
            ["Temps plein"]
        )

    exp_min = st.slider("Exp. min (ans)", 0, 15, 0, 1, key="exp_slider")

    c4, c5 = st.columns(2)
    with c4:
        formation_pref = st.multiselect("Formation", ["Baccalauréat","BTS/DUT","Licence/Bachelor","Master/Mastère","Doctorat"], [])
    with c5:
        villes_pref = st.text_input("Villes en priorité", "Paris, Île de france")

    c6, c7 = st.columns(2)
    with c6:
        langues     = st.text_input("Langues", "Français, Anglais")
    with c7:
        competences = st.text_input("Technologies/Logiciels", "Power BI, Tableau, Snowflake")

    must_have_specific = st.text_input("Mots-clés rédhibitoires (absents ⇒ élimination)", "powerbi",
                                       help="Must-have supplémentaires : si l’un de ces mots-clés est absent du CV, celui-ci est rejeté.")

    redhibitory_categories = [
        None,
        'ville',
        'formation',
        'type de contrat',
        'temps de travail',
        'expérience min',
        'langues',
        'compétences'
    ]
    cat_redhibitoire = st.selectbox(
        "Catégorie rédhibitoire (non match ⇒ élimination)",
        redhibitory_categories,
        index=0,
        format_func=lambda x: "Aucune" if x is None else x.capitalize(),
        help="Sélectionnez une catégorie qui, si elle ne correspond pas aux préférences, entraînera le rejet du CV."
    )

    seuil = st.slider("Seuil (score ≥)*", 0, 20, 3, 1)
    st.caption("(*) Champs obligatoires pour lancer le tri. Les autres champs sont facultatifs.")

    uploaded_files, folder_files = [], []
    if mode == "Télécharger des PDF":
        ups = st.file_uploader("Déposez les CV (PDF, plusieurs)", type=["pdf"], accept_multiple_files=True)
        if ups:
            tmp = "uploaded_cvs"; ensure_dir(tmp)
            for uf in ups:
                p = os.path.join(tmp, uf.name)
                with open(p, "wb") as fh: fh.write(uf.read())
                uploaded_files.append(p)
    else:
        folder = st.text_input("Chemin du dossier de PDF", "./data/cvs")
        if folder and os.path.isdir(folder):
            folder_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".pdf")]
            folder_files.sort()
        else:
            st.info("Indiquez un dossier valide.")

    lancer = st.button("Lancer le tri")

with right:
    st.markdown('<div class="section-title">| Classement & graphiques</div>', unsafe_allow_html=True)
    kpis = st.empty()
    charts = st.container()
    table_area = st.empty()
    downloads = st.container()

df_ret = pd.DataFrame()

if lancer:
    files = uploaded_files if mode == "Télécharger des PDF" else folder_files
    if not files:
        st.warning("Aucun PDF à traiter.")
    else:
        cfg = ScoringConfig(
            obligatoires=split_csv(obligatoires),
            must_have_specific=split_csv(must_have_specific),
            cat_redhibitoire=cat_redhibitoire, # Pass the selected redhibitory category
            optionnels=split_csv(optionnels),
            type_contrat_pref=split_csv(",".join(type_contrat_pref)),
            temps_pref=split_csv(",".join(temps_pref)),
            exp_min=exp_min,
            edu_pref=formation_pref,
            villes_pref=split_csv(villes_pref),
            langues=split_csv(langues),
            competences=split_csv(competences),
            permis=[],  # champ retiré ici pour simplifier; on peut le remettre si besoin
            seuil=seuil,
            top_n=20,
        )
        out = process_pdfs(files, cfg)
        df_all = out["df"]

        # KPIs
        retenus = int((df_all["retenu"] == "Oui").sum()) if not df_all.empty else 0
        kpis.markdown(f"**{len(df_all)} CV analysés** • **{retenus} retenus (score ≥ {seuil})**")

        # Graphiques (palette harmonisée)
        with charts:
            # Pie chart (above)
            pie_df = pd.DataFrame({"Statut":["Retenus","Non retenus"], "Nombre":[retenus, max(0,len(df_all)-retenus)]})
            pie = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                theta="Nombre:Q",
                color=alt.Color("Statut:N", scale=alt.Scale(range=["#F472B6", "#D9B4CF"]), legend=None),
                tooltip=["Statut","Nombre"]
            ).properties(
                height=300 # Set height here
            )
            st.altair_chart(pie, use_container_width=True)

            # Bar chart (below)
            if not df_all.empty:
                hist = alt.Chart(df_all).mark_bar(color="#F472B6").encode(
                    x=alt.X("score_total:Q", bin=alt.Bin(maxbins=12), title="Score"),
                    y=alt.Y("count()", title="Nombre de CV"),
                    tooltip=[alt.Tooltip("count()", title="CV")]
                ).properties(
                    height=300 # Set height here
                )
                st.altair_chart(hist, use_container_width=True)


        df_display = df_all.copy()
        base_cols = ["top_rank","score_total", "retenu", "fichier","type_contrat_detecte","temps_travail_status",
                     "experience_estimee_ans","formation_detectee","ville_detectee",
                     "mots_optionnels_trouves","langues_detectees","competences_detectees"]
        show_cols = [c for c in base_cols if c in df_display.columns]
        df_display = df_display[show_cols]

        df_ret = df_all[df_all["retenu"] == "Oui"].copy()


        def highlight_row(row):
            styles = []
            for col in df_display.columns:
                style = ""
                if col == "retenu":
                    if row["retenu"] == "Oui":
                        style = f"background-color:{GREEN};"
                    elif row["retenu"] == "Non":
                        style = f"background-color:{RED};"
                elif cfg.cat_redhibitoire == 'ville' and col == "ville_detectee" and df_all.loc[row.name, "ville_status"] != "match":
                     style = f"background-color:{RED};"
                elif cfg.cat_redhibitoire == 'formation' and col == "formation_detectee" and df_all.loc[row.name, "formation_status"] != "match":
                     style = f"background-color:{RED};"
                elif col == "type_contrat_detecte" and "type_contrat_status" in df_all.columns:
                    if df_all.loc[row.name, "type_contrat_status"] == "mismatch":
                        style = f"background-color:{RED};"
                elif col == "temps_travail_status" and df_all.loc[row.name, col] == "mismatch": # Use df_all for original status
                    style = f"background-color:{RED};"
                elif col == "experience_estimee_ans":
                    exp = row[col]
                    if pd.isna(exp) or (isinstance(exp,(int,float)) and exp < cfg.exp_min):
                        style = f"background-color:{RED};"
                elif cfg.cat_redhibitoire != 'formation' and col == "formation_detectee" and df_all.loc[row.name, "formation_status"] == "mismatch":
                    style = f"background-color:{RED};"
                elif cfg.cat_redhibitoire != 'ville' and col == "ville_detectee" and df_all.loc[row.name, "ville_status"] != "match":
                    style = f"background-color:{RED};"

                styles.append(style)
            return styles


        if not df_display.empty:
            table_area.dataframe(df_display.style.apply(highlight_row, axis=1), use_container_width=True)
        else:
            table_area.info("Aucun CV retenu selon le seuil et les contraintes.")

        show_cards = st.toggle("Afficher la vue cartes (Top N)", value=True)

        if show_cards and not df_ret.empty:
            top_k = min(6, len(df_ret))  # nombre de cartes à afficher
            st.markdown("### Top profils retenus")
            for _, row in df_ret.head(top_k).iterrows():
                score = int(row["score_total"])
                title = row["fichier"]
                ville = row.get("ville_detectee") or "—"
                formation = row.get("formation_detectee") or "—"
                exp = row.get("experience_estimee_ans")
                exp_txt = f"{exp} ans" if pd.notna(exp) else "—"

                st.markdown(f"""
                <div style='background:white;border:1px solid #E5E7EB;border-radius:12px;padding:1rem;margin:.5rem 0;'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div style='font-weight:700;color:{PRIMARY}'>{title}</div>
                    <div class='badge' style='background:{PRIMARY};color:white;'>Score {score}</div>
                  </div>
                  <div style='margin-top:.4rem;color:#475569;'>
                    <b>Ville:</b> {ville} &nbsp;•&nbsp; <b>Formation:</b> {formation} &nbsp;•&nbsp; <b>Exp.:</b> {exp_txt}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with downloads:
            st.write("Téléchargements")
            st.download_button("📥 Tableau complet (CSV)", data=df_all.to_csv(index=False).encode("utf-8"),
                               file_name="resultats_tri_cv.csv", mime="text/csv")

            x_all = df_to_xlsx_bytes(df_all)
            if x_all:
                st.download_button("📥 Tableau complet (XLSX)", data=x_all,
                                   file_name="resultats_tri_cv.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Installez `xlsxwriter` ou `openpyxl` pour activer l’export Excel.")

            if not df_ret.empty:
                x_ret = df_to_xlsx_bytes(df_ret)
                if x_ret:
                    st.download_button("📊 CV retenus (XLSX)", data=x_ret,
                                       file_name="CV_retenus.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            try:
                zip_bytes = build_zip_of_retained(out["results"], files, top_n=20)
                st.download_button("📦 CV pertinents (ZIP)", data=zip_bytes,
                                   file_name="CV_pertinents.zip", mime="application/zip")
            except Exception:
                st.info("ZIP indisponible (aucun fichier retenu ou droits).")

st.markdown("""
---
<div style='text-align:center; color:#6B7280; font-size:0.9rem; margin-top:2rem;'>
    © 2025 — <b>Dounia Pihan</b> · Data Analyst
</div>
""", unsafe_allow_html=True)