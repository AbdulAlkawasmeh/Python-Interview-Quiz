from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

def load_data():
    try:
        df = pd.read_csv("sample.csv")
        return {"success": True, "message": "Data loaded successfully", "columns": df.columns.tolist()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route("/data", methods=["GET"])
def process_data():
    return jsonify(load_data())


@app.route("/")
def home():
    return jsonify({"message": "Welcome"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
