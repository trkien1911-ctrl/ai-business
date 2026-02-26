from flask import Flask, render_template, request, redirect
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)
DB_NAME = "database.db"

# ===============================
# INIT DATABASE
# ===============================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS business (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month INTEGER,
            cost REAL,
            customers INTEGER,
            revenue REAL
        )
    """)
    conn.commit()
    conn.close()

# ===============================
# LOAD DATA
# ===============================
def get_data():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT month, cost, customers, revenue FROM business ORDER BY month")
    data = c.fetchall()
    conn.close()
    return data

# ===============================
# TRAIN + PREDICT
# ===============================
def train_and_predict():
    data = get_data()

    if len(data) < 5:
        return None, None, None

    X = np.array([[row[0], row[1], row[2]] for row in data])
    y = np.array([row[3] for row in data])

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = r2_score(y, predictions)

    next_month = max([row[0] for row in data]) + 1
    avg_cost = np.mean([row[1] for row in data])
    avg_customers = np.mean([row[2] for row in data])

    future_pred = model.predict([[next_month, avg_cost, avg_customers]])

    return round(future_pred[0], 2), next_month, round(accuracy, 3)

# ===============================
# ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        month = int(request.form["month"])
        cost = float(request.form["cost"])
        customers = int(request.form["customers"])
        revenue = float(request.form["revenue"])

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            "INSERT INTO business (month, cost, customers, revenue) VALUES (?, ?, ?, ?)",
            (month, cost, customers, revenue)
        )
        conn.commit()
        conn.close()

        return redirect("/")

    data = get_data()
    prediction, next_month, accuracy = train_and_predict()

    return render_template(
        "index.html",
        data=data,
        prediction=prediction,
        next_month=next_month,
        accuracy=accuracy
    )

if __name__ == "__main__":
    init_db()
    app.run()