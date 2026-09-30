import os
import re
import sys
import joblib
import pandas as pd

from urllib.parse import urlparse
from scipy.io import arff

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


MODEL_PATH = "model.joblib"

FEATURE_NAMES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State"
]


def extract_features(url):

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    domain = parsed.netloc.lower()

    features = []

    # 1. IP Address
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    features.append(1 if re.match(ip_pattern, hostname) else -1)

    # 2. URL Length
    length = len(url)

    if length < 54:
        features.append(-1)
    elif length <= 75:
        features.append(0)
    else:
        features.append(1)

    # 3. URL Shortening Service
    shortening_services = [
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "cutt.ly",
        "shorturl.at"
    ]

    features.append(
        1 if any(service in domain for service in shortening_services)
        else -1
    )

    # 4. @ Symbol
    features.append(1 if "@" in url else -1)

    # 5. Double Slash Redirecting
    remaining_url = url[8:]
    features.append(1 if "//" in remaining_url else -1)

    # 6. Prefix/Suffix (-)
    features.append(1 if "-" in hostname else -1)

    # 7. Sub Domain
    parts = hostname.split(".")

    if len(parts) <= 2:
        features.append(-1)
    elif len(parts) == 3:
        features.append(0)
    else:
        features.append(1)

    # 8. HTTPS
    features.append(1 if parsed.scheme == "https" else -1)

    return features


def find_dataset():

    for root, dirs, files in os.walk("."):

        for file in files:

            if file.lower() == "training dataset.arff":

                return os.path.join(root, file)

    return None


def train_model():

    print("\n========================================")
    print("      PHISHING URL DETECTION")
    print("========================================")

    dataset_file = find_dataset()

    if dataset_file is None:

        print("\nERROR: Training Dataset.arff not found.")

        print("\nMake sure the file is inside the dataset folder.")

        return

    print("\nDataset found:")
    print(dataset_file)

    data, meta = arff.loadarff(dataset_file)

    df = pd.DataFrame(data)

    for column in df.columns:

        if df[column].dtype == object:

            df[column] = df[column].apply(
                lambda x: x.decode("utf-8")
                if isinstance(x, bytes)
                else x
            )

    X = df[FEATURE_NAMES]

    y = df["Result"]

    y = pd.to_numeric(y)

    print("\nFeatures used:")

    for number, feature in enumerate(FEATURE_NAMES, start=1):

        print(f"{number}. {feature}")

    print("\nNumber of features:", len(FEATURE_NAMES))

    print("Number of samples:", len(df))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========================================")
    print("MODEL RESULTS")
    print("========================================")

    print(
        "\nAccuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\n========================================")

    print(
        "Model saved as:",
        MODEL_PATH
    )

    print("========================================\n")


def predict_url(url):

    if not os.path.exists(MODEL_PATH):

        print("\nModel not found.")

        print(
            "Run this first:"
        )

        print(
            "python phishing_detector.py train"
        )

        return

    model = joblib.load(MODEL_PATH)

    features = extract_features(url)

    print("\nURL:")
    print(url)

    print("\nExtracted features:")
    print(features)

    prediction = model.predict(
        [features]
    )[0]

    print("\nPrediction:")

    if prediction == -1:

        print(
            "⚠️ POSSIBLE PHISHING URL"
        )

    elif prediction == 1:

        print(
            "✅ LIKELY LEGITIMATE URL"
        )

    else:

        print(
            "Unknown prediction:",
            prediction
        )


def run_web():

    from flask import Flask, request, render_template_string

    app = Flask(__name__)

    html = """
    <!DOCTYPE html>

    <html>

    <head>

        <title>Phishing URL Detection System</title>

        <style>

            body {
                font-family: Arial;
                background: #f4f6f8;
                text-align: center;
                padding-top: 80px;
            }

            .container {
                width: 600px;
                max-width: 90%;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            input {
                width: 90%;
                padding: 12px;
                margin: 20px 0;
                font-size: 16px;
            }

            button {
                padding: 12px 25px;
                font-size: 16px;
                cursor: pointer;
            }

            .result {
                margin-top: 25px;
                font-size: 20px;
                font-weight: bold;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>Phishing URL Detection System</h1>

            <p>
                Enter a website URL to check whether it may be phishing.
            </p>

            <form method="POST">

                <input
                    type="text"
                    name="url"
                    placeholder="Enter URL here"
                    required
                >

                <br>

                <button type="submit">
                    Check URL
                </button>

            </form>

            {% if result %}

                <div class="result">
                    {{ result }}
                </div>

            {% endif %}

        </div>

    </body>

    </html>
    """

    @app.route("/", methods=["GET", "POST"])
    def home():

        result = None

        if request.method == "POST":

            url = request.form.get(
                "url",
                ""
            ).strip()

            if not os.path.exists(MODEL_PATH):

                result = "Model not found. Train the model first."

            elif not url:

                result = "Please enter a URL."

            else:

                model = joblib.load(
                    MODEL_PATH
                )

                features = extract_features(
                    url
                )

                prediction = model.predict(
                    [features]
                )[0]

                if prediction == -1:

                    result = "⚠️ This URL may be Phishing."

                else:

                    result = "✅ This URL appears to be Legitimate."

        return render_template_string(
            html,
            result=result
        )

    print("\n========================================")

    print(
        "Phishing URL Detection System"
    )

    print(
        "Open this in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "========================================\n"
    )

    app.run()


def show_help():

    print(
        """
Phishing URL Detection System

Commands:

Train:
python phishing_detector.py train

Predict:
python phishing_detector.py predict "https://www.google.com"

Web:
python phishing_detector.py web
"""
    )


def main():

    if len(sys.argv) < 2:

        show_help()

        return

    command = sys.argv[1].lower()

    if command == "train":

        train_model()

    elif command == "predict":

        if len(sys.argv) < 3:

            print(
                'Example: python phishing_detector.py predict "https://www.google.com"'
            )

            return

        predict_url(
            sys.argv[2]
        )

    elif command == "web":

        run_web()

    else:

        show_help()


if __name__ == "__main__":

    main()