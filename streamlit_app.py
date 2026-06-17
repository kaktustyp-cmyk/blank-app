import streamlit as st
import sqlite3
import os
import io
import torch
from datetime import datetime
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Garbage Collection", layout="wide")

DB_NAME = "konten.db"

# =============================
# DATABASE
# =============================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS konto (
            nutzer TEXT PRIMARY KEY,
            passwort TEXT,
            punkte INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

init_db()

# =============================
# SESSION STATE
# =============================
if "user" not in st.session_state:
    st.session_state.user = None

if "punkte" not in st.session_state:
    st.session_state.punkte = 0

if "page" not in st.session_state:
    st.session_state.page = "Konto"


# =============================
# MODEL (CACHE GLOBAL)
# =============================
@st.cache_resource
def load_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

@st.cache_resource
def load_processor():
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# =============================
# LOGIN
# =============================
def login_page():
    st.title("Login / Registrierung")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    st.text("Mit dem Login stimmen sie unseren Nutzungsbedingungen automatisch zu")
    if st.button("Hier Klicken für die Datenschutzerklärung"):
        st.text("""Datenschutzerklärung

1. Verantwortlicher
Garbage Collection (WDG Freiday)

2. Zweck der App
Die App „Garbage Collection“ dient dazu, Nutzer*innen zu motivieren, Müll zu sammeln und korrekt zu entsorgen. Dafür können Punkte gesammelt werden, indem Müll entsorgt und dies per Foto nachgewiesen wird.

3. Erhobene Daten
Wir verarbeiten folgende Daten:
 Nutzerdaten

Benutzername 
Passwort

 Fotodaten

Fotos, die sie innerhalb der App aufnehmen oder hochladen
Diese Fotos zeigen Müll sowie dessen Entsorgung (z. B. vor einem Mülleimer) 
die Fotos werden nie gespeichert, sondern nur für dich gezeigt, sie haben trotzdem das Recht,
diese zu löschen.

  Nutzungsdaten

Gesammelte Punkte 
Aktivitäten innerhalb der App 

 4. Einsatz von Künstlicher Intelligenz
Die hochgeladenen Fotos werden automatisiert durch eine KI analysiert, um festzustellen, ob Müll korrekt entsorgt wurde.
Dabei gilt:
Die Analyse erfolgt automatisiert 
Es findet keine manuelle Überprüfung durch Personen statt (außer bei Betrug)
Die Bilder werden nur für diesen Zweck verwendet und werden gar nicht gespeichert.

5. Speicherung und Löschung
Die Fotos werden automatisch nach der Analyse gelöscht.

6.Du hast jederzeit das Recht auf:
Datenübertragbarkeit 

7.Sicherheit
Wir sammeln keine Daten, außer der Fotos (siehe Erhobene Daten).

8. Änderungen der Datenschutzerklärung
Wir behalten uns vor, diese Datenschutzerklärung anzupassen,
um sie an neue rechtliche Anforderungen oder Änderungen der App anzupassen.""")

    col1, col2 = st.columns(2)

    if col1.button("Login"):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        cur.execute(
            "SELECT nutzer, punkte FROM konto WHERE nutzer=? AND passwort=?",
            (username, password),
        )

        user = cur.fetchone()
        conn.close()

        if user:
            st.session_state.user = user[0]
            st.session_state.punkte = user[1]
            st.rerun()
        else:
            st.error("Falsche Daten")

    if col2.button("Registrieren"):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        try:
            cur.execute(
                "INSERT INTO konto (nutzer, passwort) VALUES (?, ?)",
                (username, password),
            )
            conn.commit()
            st.success("Registriert!")
        except:
            st.error("User existiert bereits")

        conn.close()


# =============================
# KAMERA
# =============================
def add_points(amount):
    st.session_state.punkte += amount

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        UPDATE konto
        SET punkte = ?
        WHERE nutzer = ?
    """, (st.session_state.punkte, st.session_state.user))

    conn.commit()
    conn.close()

def kamera():
    st.subheader("Kamera")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model().to(device)
    processor = load_processor()

    foto = st.camera_input("Foto aufnehmen")

    if not foto:
        st.info("Bitte Foto aufnehmen")
        return

    img_bytes = foto.getvalue()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.image(image, caption="Aufgenommen")

    labels = [
        "trash"
    ]

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=-1)[0]

    best_idx = probs.argmax().item()
    best_label = labels[best_idx]
    confidence = float(probs[best_idx])

    st.subheader("Ergebnis")

    if "garbage" in best_label or "trash" in best_label:
        st.error(f"Müll erkannt: {best_label} ({confidence:.2f})")
        add_points(2)
    elif "recycling" in best_label or "bin" in best_label:
        st.success(f"Ordnung: {best_label} ({confidence:.2f})")
        add_points(1)
    else:
        st.info(f"Neutral: {best_label} ({confidence:.2f}")

    st.markdown("---")
    for l, p in zip(labels, probs):
        st.write(f"{l}: {float(p):.3f}")


# =============================
# KONTOVERWALTUNG
# =============================
def konto():
    st.subheader("Konto")
    st.write("User:", st.session_state.user)
    st.write("Punkte:", st.session_state.punkte)


# =============================
# RANGLISTE
# =============================
def rangliste():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT nutzer, punkte FROM konto ORDER BY punkte DESC LIMIT 10")
    data = cur.fetchall()
    conn.close()

    st.subheader("Top 10")

    for i, (u, p) in enumerate(data):
        st.write(f"{i+1}. {u} — {p}")
def background():
    monat = datetime.now().month
    if monat in [3, 4, 5]:
        img = "Fruehling.png"
    elif monat in [6, 7, 8]:
        img = "Sommer.jpg"
    elif monat in [9, 10, 11]:
        img = "Herbst.png"
    else:
        img = "Winter.png"
    if os.path.exists(img):
        st.image(img, use_container_width=True)

# =============================
# ROUTER (WICHTIG!)
# =============================
def main_app():
    background()
    st.sidebar.title("Menü")

    choice = st.sidebar.radio(
        "Navigation",
        ["Konto", "Rangliste", "Kamera", "Logout"]
    )

    if choice == "Konto":
        konto()

    elif choice == "Rangliste":
        rangliste()

    elif choice == "Kamera":
        kamera()

    elif choice == "Logout":
        st.session_state.user = None
        st.rerun()


# =============================
# START
# =============================
if st.session_state.user is None:
    login_page()
else:
    main_app()
