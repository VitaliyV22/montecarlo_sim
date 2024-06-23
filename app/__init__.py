from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/monte-carlo', methods=['POST'])
def monte_carlo_simulation():
    data = request.json
    S0 = data.get('S0')  # Initial stock price
    K = data.get('K')    # Strike price
    T = data.get('T')    # Time to maturity
    r = data.get('r')    # Risk-free rate
    sigma = data.get('sigma')  # Volatility
    num_simulations = data.get('num_simulations', 10000)

    dt = 1/252
    S = np.zeros((num_simulations, int(T*252)))
    S[:, 0] = S0

    for t in range(1, int(T*252)):
        Z = np.random.standard_normal(num_simulations)
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    option_price = np.exp(-r * T) * np.mean(np.maximum(S[:, -1] - K, 0))

    return jsonify({'option_price': option_price})

if __name__ == '__main__':
    app.run(debug=True)
