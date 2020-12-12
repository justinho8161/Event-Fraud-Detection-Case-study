from flask import Flask, request, redirect, render_template
from pymongo import MongoClient
from flask_bootstrap import Bootstrap
import pickle
from mongo_db_process import update_db, pull_values

app = Flask(__name__)
Bootstrap(app)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/score", methods=["POST"])
def score():
    try:
        update_db(model)
    except:
        pass

    low, medium, high = pull_values()

    return render_template("score.html", low=low, medium=medium, high=high)


if __name__ == "__main__":
    mongo_client = MongoClient()
    db = mongo_client["Fraud_Detection"]
    collection = db["Events"]

    with open("model_gb.pkl", "rb") as f:
        model = pickle.load(f)

    app.run(host="0.0.0.0", port=8080, debug=True)
