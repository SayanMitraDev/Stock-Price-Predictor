from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from model import load_data, create_dataset, build_model, scaler

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    ticker = "AAPL"

    if request.method == "POST":
        ticker = request.form.get("ticker")

        data, scaled_data = load_data(ticker)

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]

        x_train, y_train = create_dataset(train_data)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = build_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

        test_data = scaled_data[train_size - 60:]
        x_test = []

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        pred = model.predict(x_test)
        pred = scaler.inverse_transform(pred)

        prediction = pred[-1][0]

    return render_template("index.html", prediction=prediction, ticker=ticker)

if __name__ == "__main__":
    app.run(debug=True)