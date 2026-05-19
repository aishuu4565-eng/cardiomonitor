from flask import Flask, render_template, request, redirect, session
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras

# ─────────────────────────────────────────────
# LOAD MODEL FILES
# ─────────────────────────────────────────────
model          = keras.models.load_model("stroke_prediction_model.keras", compile=False)
scaler         = joblib.load("models/scaler.pkl")
encoders       = joblib.load("models/encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
bmi_mean       = joblib.load("models/bmi_mean.pkl")

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "cardioguard_secret_2024"

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT,
            email    TEXT UNIQUE,
            password TEXT,
            age      INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT,
            name        TEXT,
            probability REAL,
            result      TEXT,
            date        TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
# HELPER: NORMALIZE FORM VALUES
# ─────────────────────────────────────────────
def normalize_form_values(gender, married, work, residence, smoking):
    gender    = str(gender).strip()
    married   = str(married).strip()
    work      = str(work).strip()
    residence = str(residence).strip()
    smoking   = str(smoking).strip()

    gender_map = {
        "male": "Male", "female": "Female", "other": "Male"  # fallback Other → Male
    }
    married_map  = {"yes": "Yes", "no": "No"}
    work_map     = {
        "govt job": "Govt_job", "govt_job": "Govt_job",
        "private": "Private",
        "self employed": "Self-employed", "self-employed": "Self-employed",
        "children": "children",
        "never worked": "Never_worked", "never_worked": "Never_worked"
    }
    residence_map = {"urban": "Urban", "rural": "Rural"}
    smoking_map   = {
        "formerly smoked": "formerly smoked",
        "never smoked": "never smoked",
        "smokes": "smokes",
        "unknown": "Unknown"
    }

    gender    = gender_map.get(gender.lower(), "Male")
    married   = married_map.get(married.lower(), "No")
    work      = work_map.get(work.lower(), "Private")
    residence = residence_map.get(residence.lower(), "Urban")
    smoking   = smoking_map.get(smoking.lower(), "Unknown")

    return gender, married, work, residence, smoking

# ─────────────────────────────────────────────
# HELPER: FEATURE ENGINEERING  (must match train_model.py)
# ─────────────────────────────────────────────
def add_features_from_input(row: pd.DataFrame) -> pd.DataFrame:
    data = row.copy()

    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 30, 50, 70, 120],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    data["high_glucose"] = (data["avg_glucose_level"] >= 140).astype(int)
    data["high_bmi"]     = (data["bmi"] >= 30).astype(int)

    data["risk_score"] = (
        data["hypertension"].astype(int)
        + data["heart_disease"].astype(int)
        + (data["avg_glucose_level"] >= 140).astype(int)
        + (data["bmi"] >= 30).astype(int)
        + (data["age"] >= 50).astype(int)
    )

    data["lifestyle_risk"] = (
        (data["avg_glucose_level"] >= 140).astype(int)
        + (data["bmi"] >= 30).astype(int)
        + (data["smoking_status"] == "smokes").astype(int)   # NOTE: checked BEFORE encoding
    )

    return data

# ─────────────────────────────────────────────
# HELPER: ENCODE CATEGORICAL COLUMN
# ─────────────────────────────────────────────
def encode_column(col_name, value):
    le    = encoders[col_name]
    value = str(value)
    if value not in le.classes_:
        value = le.classes_[0]           # fallback to first known class
    return int(le.transform([value])[0])

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    if "user" in session:
        return redirect("/dashboard")
    return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email    = request.form["email"].strip()
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?", (email, password)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = email
            return redirect("/dashboard")
        error = "Invalid email or password"

    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        name     = request.form["name"].strip()
        email    = request.form["email"].strip()
        password = request.form["password"]

        conn   = sqlite3.connect("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password)
            )
            conn.commit()
            conn.close()
            session["user"] = email
            return redirect("/profile")
        except sqlite3.IntegrityError:
            conn.close()
            error = "Email already registered. Please login."

    return render_template("signup.html", error=error)


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        name = request.form["name"].strip()
        age  = request.form["age"]

        conn   = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET name=?, age=? WHERE email=?",
            (name, age, session["user"])
        )
        conn.commit()
        conn.close()
        return redirect("/dashboard")

    return render_template("profile.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    conn   = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name, age FROM users WHERE email=?", (session["user"],))
    user = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM history WHERE email=?", (session["user"],))
    count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT date, probability FROM history
        WHERE email=? ORDER BY id ASC
    """, (session["user"],))
    history_rows = cursor.fetchall()
    conn.close()

    dates = [row[0] for row in history_rows]
    probs = [row[1] for row in history_rows]

    return render_template(
        "dashboard.html",
        name=user[0] if user else "User",
        age=user[1]  if user else "—",
        count=count,
        dates=dates,
        probs=probs
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    prob   = None
    result = None
    color  = None
    name   = None

    if request.method == "POST":
        try:
            # ── Read form ──────────────────────────
            name              = request.form.get("name", "").strip()
            gender            = request.form.get("gender", "Male").strip()
            age               = float(request.form.get("age", 0))
            hypertension      = int(request.form.get("hypertension", 0))
            heart_disease     = int(request.form.get("heart_disease", 0))
            married           = request.form.get("married", "No").strip()
            work              = request.form.get("work", "Private").strip()
            residence         = request.form.get("residence", "Urban").strip()
            avg_glucose_level = float(request.form.get("glucose", 100))
            bmi_raw           = request.form.get("bmi", "").strip()
            smoking           = request.form.get("smoking", "Unknown").strip()

            bmi = bmi_mean if bmi_raw in ("", "N/A") else float(bmi_raw)

            # ── Normalize text values ──────────────
            gender, married, work, residence, smoking = normalize_form_values(
                gender, married, work, residence, smoking
            )

            # ── Build raw DataFrame ────────────────
            raw_df = pd.DataFrame([{
                "gender":            gender,
                "age":               age,
                "hypertension":      hypertension,
                "heart_disease":     heart_disease,
                "ever_married":      married,
                "work_type":         work,
                "Residence_type":    residence,
                "avg_glucose_level": avg_glucose_level,
                "bmi":               bmi,
                "smoking_status":    smoking
            }])

            # ── Feature engineering (before encoding, so text checks work) ──
            raw_df = add_features_from_input(raw_df)

            # ── Encode categoricals ────────────────
            for col in ["gender", "ever_married", "work_type",
                        "Residence_type", "smoking_status"]:
                raw_df[col] = encode_column(col, raw_df.at[0, col])

            # ── Align to training feature order ────
            input_df        = raw_df.reindex(columns=feature_columns, fill_value=0)
            input_df        = input_df.astype(float)

            # ── Scale ─────────────────────────────
            input_scaled    = scaler.transform(input_df)

            # ── Predict ───────────────────────────
            raw_prob        = float(model.predict(input_scaled, verbose=0)[0][0])

            # ── Rule-based boost (same as before) ─
            risk_score      = float(input_df["risk_score"].values[0])
            rule_prob       = risk_score * 0.15
            final_prob      = float(np.clip(max(raw_prob, rule_prob), 0.01, 0.95))

            prob = round(final_prob * 100, 2)

            if final_prob < 0.1:
                result, color = "Very Low Risk", "green"
            elif final_prob < 0.3:
                result, color = "Low Risk",      "lightgreen"
            elif final_prob < 0.6:
                result, color = "Moderate Risk", "orange"
            else:
                result, color = "High Risk",     "red"

            # ── Save to history ───────────────────
            conn   = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO history (email, name, probability, result, date)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session["user"],
                name,
                prob,
                result,
                datetime.now().strftime("%d-%m-%Y %H:%M")
            ))
            conn.commit()
            conn.close()

        except Exception as e:
            return f"<h3>Prediction Error: {e}</h3><a href='/predict'>Try Again</a>"

    return render_template(
        "predict.html",
        prob=prob,
        result=result,
        color=color,
        name=name
    )


@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/")

    conn   = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, probability, result, date
        FROM history WHERE email=?
        ORDER BY id DESC
    """, (session["user"],))
    data = cursor.fetchall()
    conn.close()

    return render_template("history.html", data=data)


@app.route("/about")
def about():
    if "user" not in session:
        return redirect("/")
    return render_template("about.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)