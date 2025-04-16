from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# 🔁 Charger le modèle MLflow depuis un run
# Remplace <run_id> par l’ID du run où tu as loggé ton modèle
model_path = "model/781887167403503729/8b59587d094044508ec10e06a24efb74/artifacts/model"
model = mlflow.sklearn.load_model(model_path)

@app.route("/")
def home():
    return "API Flask MLflow est en ligne ! 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "text" not in data:
            return jsonify({"error": "Le champ 'text' est requis"}), 400

        input_text = data["text"]

        # Prévoir une seule prédiction
        prediction = model.predict([input_text])[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
