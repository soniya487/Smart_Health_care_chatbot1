# app.py
# ============================================================
# Smart Healthcare Chatbot – A+ Engine with Voice & Medical UI
# ============================================================

import os
import io
import numpy as np
import pandas as pd
import torch

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Speech recognition (for voice input)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ------------------------------------------------------------
# 1. Streamlit Page Config & Top UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smart Healthcare Chatbot",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #BBBBBB;
    }
    .chat-bubble-user {
        background-color: #1f2933;
        padding: 10px 14px;
        border-radius: 12px;
        margin-bottom: 6px;
    }
    .chat-bubble-bot {
        background-color: #111827;
        padding: 10px 14px;
        border-radius: 12px;
        margin-bottom: 6px;
        border: 1px solid #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='big-title'>🩺 Smart Healthcare Chatbot (A+ Engine)</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Educational demo only – not a medical diagnosis. "
    "Always consult a doctor for real health decisions.</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 2. Paths & Dataset Loading
# ------------------------------------------------------------
DATASET_PATH = "merged_health_dataset_3500.csv"


@st.cache_data(show_spinner="📥 Loading dataset...")
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please place merged_health_dataset_3500.csv in the same folder as app.py."
        )

    df = pd.read_csv(path)

    # Normalize expected columns
    for col in ["disease", "subtype", "severity", "question", "answer", "followup"]:
        if col not in df.columns:
            df[col] = ""

    df["disease"] = df["disease"].astype(str).str.strip()
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].fillna("")
    df["followup"] = df["followup"].fillna("")

    df = df[["disease", "subtype", "severity", "question", "answer", "followup"]]
    return df


@st.cache_resource(show_spinner="🧠 Training TF-IDF + Logistic Regression...")
def train_classifier(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(df["question"].tolist())
    y = df["disease"].tolist()

    clf = LogisticRegression(max_iter=400, n_jobs=-1)
    clf.fit(X, y)

    return vectorizer, clf


@st.cache_resource(show_spinner="🔎 Loading MiniLM & computing embeddings...")
def load_embeddings(df: pd.DataFrame):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    questions = df["question"].tolist()
    embeddings = model.encode(
        questions,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True
    )
    return model, embeddings


# ------------------------------------------------------------
# 3. Rule Engine Config
# ------------------------------------------------------------
SYMPTOM_KEYWORDS = {
    "headache": ["headache", "head pain", "pressure in head", "behind eyes", "migraine"],
    "cold_flu": ["cold", "blocked nose", "runny nose", "sneezing", "flu", "sore throat", "cough"],
    "fever": ["fever", "feverish", "temperature", "high temperature", "chills", "hot body"],
    "stomach": ["stomach", "tummy", "belly", "bloat", "bloated", "gas", "acidity",
                "indigestion", "diarrhea", "loose motion", "constipation"],
    "metabolic_bp": ["bp", "blood pressure", "pressure high", "pressure low",
                     "hypertension", "sugar", "diabetes", "glucose"],
    "menstrual": ["period", "periods", "menstrual", "menstruation", "cycle",
                  "pms", "cramps", "cramping", "heavy flow"]
}

RED_FLAGS = [
    "chest pain", "heart pain", "pressure in chest",
    "difficulty breathing", "cannot breathe",
    "shortness of breath", "severe breathlessness",
    "fainted", "fainting", "confusion", "very high fever",
    "uncontrolled bleeding"
]

NON_CRITICAL_AREAS = ["stomach", "belly", "head", "back", "leg", "arm", "shoulder", "knee"]

SEVERITY_WORDS = {
    "severe": 2.0,
    "very strong": 2.0,
    "intense": 1.5,
    "vomit": 1.5,
    "vomiting": 1.5
}

EMOTION_WORDS = ["scared", "worried", "anxious", "afraid", "panic", "panicking"]


def extract_flags(text: str):
    t = text.lower()
    return {k: any(w in t for w in words) for k, words in SYMPTOM_KEYWORDS.items()}


def has_red_flag(text: str) -> bool:
    t = text.lower()
    return any(flag in t for flag in RED_FLAGS)


def mentions_severe_pain(text: str) -> bool:
    t = text.lower()
    return "severe pain" in t or "very bad pain" in t or "very strong pain" in t


def detect_contradiction(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["actually not", "no longer", "it's gone", "not anymore"])


def severity_score(flags: dict, text: str):
    t = text.lower()
    score = sum(flags.values())
    for w, val in SEVERITY_WORDS.items():
        if w in t:
            score += val
    if score <= 1:
        return "Mild", score
    if score <= 3:
        return "Moderate", score
    return "Elevated", score


DIFF_QUESTIONS = {
    "headache": [
        "Did this start after long screen time or studying?",
        "Do you have a cold or blocked nose?",
        "Did you sleep less than usual or skip meals?"
    ],
    "cold_flu": [
        "Do you have sore throat or sneezing?",
        "Did this start after cold weather or AC?",
        "Do you have body pain or tiredness?"
    ],
    "fever": [
        "Since when do you feel feverish?",
        "Do you also have chills or body pain?",
        "Did you take any medicine like paracetamol?"
    ],
    "stomach": [
        "Did this start after a particular meal?",
        "Do you feel bloated, gassy, or is it sharp pain?",
        "Do you have loose motion or constipation?"
    ],
    "metabolic_bp": [
        "Have you checked your BP or sugar recently?",
        "Do you feel dizziness or blurred vision?"
    ],
    "menstrual": [
        "Is your period ongoing or about to start?",
        "Is this stronger than your usual cramps?",
        "Do you also have back pain or heavy flow?"
    ]
}

TIPS = {
    "headache": "• Take short breaks from screens\n• Drink water\n• Rest in a dim, quiet room",
    "cold_flu": "• Sip warm fluids (soup, herbal tea)\n• Do gentle steam inhalation\n• Avoid very cold drinks",
    "stomach": "• Sip warm water slowly\n• Avoid oily/spicy or very heavy meals\n• Stay upright for a while after eating",
    "fever": "• Drink plenty of fluids\n• Rest more than usual\n• Avoid very heavy or greasy food",
    "metabolic_bp": "• Avoid very salty/sugary foods\n• Sit and rest quietly\n• Follow your doctor’s BP/sugar plan if you have one",
    "menstrual": "• Use a warm towel/heating pad on lower abdomen\n• Do light stretching or walking\n• Rest if cramps feel intense"
}

RECOVERY_HINTS = {
    "headache": "Many simple headache or sinus patterns settle in 1–3 days with rest, hydration and reduced screen time.",
    "cold_flu": "Mild cold/flu symptoms usually improve in 3–5 days with rest and fluids, but watch for high fever or breathing issues.",
    "stomach": "Simple indigestion or bloating commonly improves within a day as long as you avoid triggers and stay hydrated.",
    "fever": "Mild viral fevers often ease in 2–4 days, but persistent or very high fever needs medical attention.",
    "metabolic_bp": "Blood pressure or sugar concerns should always be taken seriously; regular follow-up with a doctor is important.",
    "menstrual": "Menstrual cramps often ease as the flow reduces, usually over 1–3 days."
}


# ------------------------------------------------------------
# 4. Classifier & Retrieval Helpers
# ------------------------------------------------------------
def map_label_to_bucket(label: str) -> str:
    l = label.lower()
    if "head" in l:
        return "headache"
    if "cold" in l or "flu" in l:
        return "cold_flu"
    if "fever" in l:
        return "fever"
    if "stomach" in l:
        return "stomach"
    if "menstru" in l or "period" in l:
        return "menstrual"
    return "metabolic_bp"


def classify_with_conf(text: str, vectorizer, clf):
    vec = vectorizer.transform([text])
    probs = clf.predict_proba(vec)[0]
    idx = np.argmax(probs)
    label = clf.classes_[idx]
    bucket = map_label_to_bucket(label)
    return label, bucket, float(probs[idx])


def choose_bucket(flags: dict, fallback: str, text: str) -> str:
    t = text.lower()
    if "bp" in t or "blood pressure" in t or "sugar" in t or "glucose" in t:
        return "metabolic_bp"

    if flags.get("stomach"):
        return "stomach"
    if flags.get("menstrual"):
        return "menstrual"
    if flags.get("cold_flu"):
        return "cold_flu"
    if flags.get("fever"):
        return "fever"
    if flags.get("headache"):
        return "headache"
    if flags.get("metabolic_bp"):
        return "metabolic_bp"

    return fallback


def retrieve_answer(text: str, bucket: str, df: pd.DataFrame, df_embeddings, model):
    subset = df[df["disease"].apply(map_label_to_bucket) == bucket]
    if len(subset) == 0:
        subset = df
        emb = df_embeddings
    else:
        emb = df_embeddings[subset.index]

    q_emb = model.encode([text], convert_to_tensor=True)
    sims = util.cos_sim(q_emb, emb)[0]
    best = torch.argmax(sims).item()
    sim_score = float(sims[best])
    ans = subset.iloc[best]["answer"]
    return ans, sim_score


# ------------------------------------------------------------
# 5. Bot State & Reasoning
# ------------------------------------------------------------
def new_engine_state():
    return {
        "awaiting_clarification": False,
        "awaiting_pain_location": False,
        "initial_text": "",
        "flags": {k: False for k in SYMPTOM_KEYWORDS.keys()},
        "reason": None,
    }


def format_explanation(reason: dict) -> str:
    if not reason:
        return "I do not have enough information yet to explain. Try describing your symptoms first."

    flags = ", ".join([k for k, v in reason["flags"].items() if v]) or "none"
    return f"""
🧠 **Model Reasoning (Educational View)**

- Final bucket: **{reason['final_bucket']}**
- Severity: **{reason['severity']}** (score {reason['sev_score']:.2f})
- Overall confidence: **{reason['overall_conf']:.2f}**
- ML label: **{reason['ml_label']}** (prob {reason['ml_conf']:.2f})
- Embedding similarity: **{reason['emb_sim']:.2f}**
- Detected symptom flags: **{flags}**

This is only to show how the model combined ML, embeddings, and rules.
It is not a medical diagnosis.
""".strip()


def chatbot_turn(
    user_text: str,
    state: dict,
    df,
    vectorizer,
    clf,
    model,
    df_embeddings,
):
    text = user_text.strip()

    # "why?" explanation
    if text.lower() in ["why", "why?", "explain", "how did you decide"]:
        return format_explanation(state.get("reason")), state

    # Red flag triage
    if has_red_flag(text):
        state = new_engine_state()
        msg = (
            "⚠️ Some of the symptoms you mentioned (like chest pain, breathing problems, fainting, "
            "or confusion) can be serious.\n\n"
            "For your safety, please contact a doctor, urgent care, or emergency services "
            "instead of relying only on this chatbot."
        )
        return msg, state

    # Greetings
    lower = text.lower()
    if lower in ["hi", "hello", "hey", "hai"]:
        state = new_engine_state()
        msg = (
            "Hello! Tell me what you're feeling. For example:\n\n"
            "- I have a headache\n"
            "- I feel feverish\n"
            "- My stomach is bloated"
        )
        return msg, state

    # Thanks
    if "thank" in lower:
        state = new_engine_state()
        return (
            "You're welcome. If anything changes or feels worse, you can tell me again and "
            "we'll go step by step."
        ), state

    # Conversation reset phrases
    if detect_contradiction(text):
        state = new_engine_state()
        return "Okay, I'll reset our conversation. Please tell me your current symptoms again.", state

    # If we were asking for pain location
    if state.get("awaiting_pain_location"):
        combined = (state["initial_text"] + " " + text).strip()
        flags = extract_flags(combined)
        ml_label, ml_bucket, ml_conf = classify_with_conf(combined, vectorizer, clf)
        bucket = choose_bucket(flags, ml_bucket, combined)

        low = text.lower()
        if any(w in low for w in ["chest", "heart", "lungs", "under chest", "around chest"]):
            state = new_engine_state()
            msg = (
                "⚠️ Because you mentioned **severe pain in the chest/heart/lung area**, "
                "this can be serious.\n\n"
                "Please seek urgent medical help instead of relying on this chatbot."
            )
            return msg, state

        state["awaiting_pain_location"] = False
        state["awaiting_clarification"] = True
        state["initial_text"] = combined
        state["flags"] = flags

        questions = DIFF_QUESTIONS.get(bucket, [])
        bullets = "\n".join(f"- {q}" for q in questions) or "- Can you tell a bit more about how it feels and when it started?"

        reply = (
            f"You mentioned: **{combined}**\n\n"
            "Let me understand this step-by-step:\n"
            f"{bullets}"
        )
        return reply, state

    # FIRST STAGE: no clarification yet
    if not state.get("awaiting_clarification", False):
        if mentions_severe_pain(text) and not any(area in text.lower() for area in NON_CRITICAL_AREAS):
            state["awaiting_pain_location"] = True
            state["initial_text"] = text
            return (
                "You mentioned **severe pain**. To guide you safely, where do you feel it the most?\n"
                "- Chest / heart / lungs\n"
                "- Stomach / abdomen\n"
                "- Head / sinus area\n"
                "- General body / muscles",
                state,
            )

        flags = extract_flags(text)
        ml_label, ml_bucket, ml_conf = classify_with_conf(text, vectorizer, clf)
        bucket = choose_bucket(flags, ml_bucket, text)

        state["awaiting_clarification"] = True
        state["initial_text"] = text
        state["flags"] = flags

        questions = DIFF_QUESTIONS.get(bucket, [])
        bullets = "\n".join(f"- {q}" for q in questions) or "- Can you tell a bit more about how it feels and when it started?"

        reply = (
            f"You mentioned: **{text}**\n\n"
            "Let me understand this step-by-step:\n"
            f"{bullets}"
        )
        return reply, state

    # SECOND STAGE: combine and give final answer
    combined = (state["initial_text"] + " " + text).strip()
    flags = extract_flags(combined)
    ml_label, ml_bucket, ml_conf = classify_with_conf(combined, vectorizer, clf)
    bucket = choose_bucket(flags, ml_bucket, combined)

    # Special tweak: headache + cold → sinus-type headache
    if flags.get("headache") and flags.get("cold_flu"):
        bucket = "headache"
        answer = (
            "Because you described both headache and cold-like symptoms, this behaves like a "
            "**sinus-type headache**, where blocked nose and sinus pressure cause pain around "
            "the forehead and eyes."
        )
        sim = 0.0
    else:
        answer, sim = retrieve_answer(combined, bucket, df, df_embeddings, model)
        if not answer:
            answer = (
                "From what you describe, it seems related to a routine pattern in this category. "
                "Focus on rest, fluids, and avoiding obvious triggers, and monitor how it changes."
            )

    severity, sev_score = severity_score(flags, combined)
    tips = TIPS.get(bucket, "• Rest, hydrate, and observe how your body responds.")
    recovery = RECOVERY_HINTS.get(
        bucket,
        "Recovery time depends on your overall health and whether symptoms are improving or worsening."
    )
    overall_conf = (ml_conf + max(sim, 0.0)) / 2.0

    reason = {
        "combined_text": combined,
        "flags": flags,
        "ml_label": ml_label,
        "ml_bucket": ml_bucket,
        "ml_conf": ml_conf,
        "final_bucket": bucket,
        "emb_sim": sim,
        "severity": severity,
        "sev_score": sev_score,
        "overall_conf": overall_conf,
    }
    state["reason"] = reason
    state["awaiting_clarification"] = False
    state["initial_text"] = ""

    reply = f"""
From what you described, this seems most related to **{bucket}** (**{severity} level**).

**In simple words:**  
{answer}

**What you can try now:**  
{tips}

**Rough recovery expectation:**  
{recovery}

This is not a medical diagnosis, but guidance based on patterns in my data.
If symptoms become very strong, feel very unusual, or last many days, it's safer to talk with a doctor or urgent care.

(You can type **"why?"** if you want to see how I reasoned this.)
""".strip()

    return reply, state


# ------------------------------------------------------------
# 6. Load Resources
# ------------------------------------------------------------
try:
    df = load_dataset(DATASET_PATH)
    vectorizer, clf = train_classifier(df)
    model, df_embeddings = load_embeddings(df)
    st.success(
        f"✅ Dataset loaded – {len(df)} rows | Diseases: {sorted(df['disease'].unique())}"
    )
except Exception as e:
    st.error(f"Error while loading resources: {e}")
    st.stop()

# ------------------------------------------------------------
# 7. Streamlit Session State
# ------------------------------------------------------------
if "engine_state" not in st.session_state:
    st.session_state.engine_state = new_engine_state()

if "chat_history" not in st.session_state:
    # list of {"role": "user"/"bot", "text": str}
    st.session_state.chat_history = []

# ------------------------------------------------------------
# 8. Layout: Chat (left) + Voice & Controls (right)
# ------------------------------------------------------------
left_col, right_col = st.columns([2.0, 1.0])

# ------------------ LEFT: CHAT ------------------------------
with left_col:
    st.subheader("💬 Chat")

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            # first bot message
            first_msg = "Hello! Tell me what you're feeling."
            st.session_state.chat_history.append({"role": "bot", "text": first_msg})

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-bubble-user'><b>You:</b> {msg['text']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-bubble-bot'><b>Bot:</b> {msg['text']}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("**Type your message:**")
    user_input = st.text_input("", key="input_box")

    col_send, col_reset = st.columns([1, 1])
    with col_send:
        send_clicked = st.button("Send")
    with col_reset:
        reset_clicked = st.button("Reset Conversation")

    if reset_clicked:
        st.session_state.engine_state = new_engine_state()
        st.session_state.chat_history = []
        st.success("Conversation reset. Start again, e.g., *I have a headache*.")

    if send_clicked and user_input.strip():
        text = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "text": text})

        bot_text, new_state = chatbot_turn(
            text,
            st.session_state.engine_state,
            df,
            vectorizer,
            clf,
            model,
            df_embeddings,
        )
        st.session_state.engine_state = new_state
        st.session_state.chat_history.append({"role": "bot", "text": bot_text})
        # Do NOT clear input_box programmatically to avoid Streamlit state errors


# ------------------ RIGHT: VOICE + CONTROLS -----------------
with right_col:
    st.subheader("🎙 Voice Input")

    st.caption(
        "Upload a short **WAV** voice recording (e.g., from your phone). "
        "The app will try to convert it to text and send it to the chatbot."
    )

    audio_file = st.file_uploader(
        "Upload voice message",
        type=["wav", "mp3", "m4a"],
        key="voice_file",
    )

    voice_text_placeholder = st.empty()

    if not SR_AVAILABLE:
        st.info(
            "Voice transcription library `SpeechRecognition` is not installed. "
            "Install it with `pip install SpeechRecognition` if you want real voice-to-text."
        )

    if st.button("Transcribe & Send Voice") and audio_file is not None:
        if not SR_AVAILABLE:
            voice_text_placeholder.warning(
                "SpeechRecognition is not available. Please install it or type manually."
            )
        else:
            try:
                recognizer = sr.Recognizer()
                audio_bytes = audio_file.read()
                audio_stream = io.BytesIO(audio_bytes)

                with sr.AudioFile(audio_stream) as source:
                    audio_data = recognizer.record(source)

                recognized_text = recognizer.recognize_google(audio_data)
                voice_text_placeholder.success(f"Recognized text: **{recognized_text}**")

                # Send recognized text into chat
                st.session_state.chat_history.append(
                    {"role": "user", "text": recognized_text}
                )
                bot_text, new_state = chatbot_turn(
                    recognized_text,
                    st.session_state.engine_state,
                    df,
                    vectorizer,
                    clf,
                    model,
                    df_embeddings,
                )
                st.session_state.engine_state = new_state
                st.session_state.chat_history.append(
                    {"role": "bot", "text": bot_text}
                )
            except Exception as e:
                voice_text_placeholder.error(
                    f"Could not transcribe audio. Use WAV if possible. Error: {e}"
                )

    st.markdown("---")
    st.subheader("📊 Session Snapshot")
    st.write(f"Chat history count: **{len(st.session_state.chat_history)}** messages")

    if st.session_state.engine_state.get("reason"):
        st.markdown(format_explanation(st.session_state.engine_state["reason"]))
    else:
        st.info("No reasoning yet. Try something like *I have a headache since morning*.")
