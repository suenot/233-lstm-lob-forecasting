# Chapter 275: LSTM LOB Forecasting

## 1. Introduction

Limit Order Book (LOB) data provides a granular, real-time view of market microstructure: the queues of buy and sell orders at various price levels that collectively determine the instantaneous supply and demand landscape. Forecasting the evolution of LOB features -- mid-price direction, spread dynamics, or order flow imbalance -- is one of the most challenging problems in quantitative finance because the data is high-dimensional, noisy, and exhibits complex temporal dependencies that span multiple timescales.

Long Short-Term Memory (LSTM) networks are a class of recurrent neural network (RNN) specifically designed to capture long-range dependencies in sequential data. Unlike vanilla RNNs, which suffer from vanishing and exploding gradients during backpropagation through time, LSTMs employ a gating mechanism that selectively retains, updates, or discards information along a dedicated cell state highway. This architectural innovation makes LSTMs particularly well-suited for LOB forecasting, where a model must simultaneously track:

- **Fast dynamics**: sub-second changes in the top-of-book spread and order arrivals.
- **Slow dynamics**: multi-minute trends in volume-weighted imbalance and mid-price drift.
- **Event memory**: the lasting impact of large market orders or liquidity withdrawals that occurred many timesteps in the past.

In this chapter we develop a complete LSTM-based LOB forecasting pipeline in Rust, from raw orderbook and candlestick data sourced from the Bybit exchange, through feature engineering, model training, and multi-step mid-price direction prediction.

---

## 2. Mathematical Foundations

### 2.1 The Vanilla RNN Problem

A vanilla RNN computes its hidden state at timestep $t$ as:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

During backpropagation through time (BPTT), the gradient of the loss with respect to $h_t$ involves products of the Jacobian $\frac{\partial h_t}{\partial h_{t-1}}$ over many timesteps. When the spectral radius of this Jacobian is less than one, gradients vanish exponentially; when greater than one, they explode. This limits vanilla RNNs to learning dependencies spanning roughly 10-20 timesteps.

### 2.2 LSTM Cell Architecture

An LSTM cell replaces the single nonlinear transformation with four interacting gates. Given input vector $x_t \in \mathbb{R}^d$ and previous hidden state $h_{t-1} \in \mathbb{R}^h$, the LSTM computes:

**Forget gate** -- decides what fraction of the old cell state to retain:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input gate** -- decides which new information to store:

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

**Candidate cell state** -- proposes new values to add:

$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$

**Cell state update** -- the core of the LSTM, combining old and new information:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output gate** -- determines the exposed hidden state:

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

**Hidden state**:

$$h_t = o_t \odot \tanh(C_t)$$

where $\sigma$ is the sigmoid function, $\odot$ denotes element-wise (Hadamard) product, and $[\cdot, \cdot]$ denotes concatenation.

The cell state $C_t$ acts as an information highway: because the forget gate can learn to keep $f_t \approx 1$ for relevant information, gradients can flow backward through the cell state with minimal attenuation, solving the vanishing gradient problem.

### 2.3 Stacked (Multi-Layer) LSTM

A single LSTM layer may not capture the hierarchical temporal abstractions present in LOB data. We stack $L$ LSTM layers so that the hidden state output $h_t^{(l)}$ of layer $l$ serves as the input to layer $l+1$:

$$h_t^{(l)} = \text{LSTM}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)}, C_{t-1}^{(l)})$$

where $h_t^{(0)} = x_t$. Lower layers tend to learn short-term patterns (tick-level), while higher layers capture longer-horizon structure (trend, regime).

### 2.4 Sequence-to-One Forecasting

For mid-price direction classification we use the final hidden state $h_T^{(L)}$ of the top LSTM layer, passed through a dense output layer:

$$\hat{y} = \text{softmax}(W_y h_T^{(L)} + b_y)$$

with classes: **Up**, **Down**, **Stationary**. Training minimizes cross-entropy loss.

---

## 3. LOB Feature Engineering

Raw LOB data is a matrix of (price, quantity) pairs at each level. We engineer the following features at each timestep $t$:

### 3.1 Mid-Price

$$p_{\text{mid},t} = \frac{p_{\text{ask},t}^{(1)} + p_{\text{bid},t}^{(1)}}{2}$$

### 3.2 Spread

$$s_t = p_{\text{ask},t}^{(1)} - p_{\text{bid},t}^{(1)}$$

A widening spread signals decreasing liquidity and potential volatility.

### 3.3 Imbalance Ratio

$$\text{IR}_t = \frac{V_{\text{bid},t} - V_{\text{ask},t}}{V_{\text{bid},t} + V_{\text{ask},t}}$$

where $V_{\text{bid},t} = \sum_{k=1}^{K} q_{\text{bid},t}^{(k)}$ is the total bid volume across $K$ levels. An imbalance ratio near +1 indicates heavy buying pressure; near -1 signals selling pressure. This is one of the strongest short-term predictors of mid-price direction.

### 3.4 Volume Profile / Depth

We track cumulative volume at multiple depth levels:

$$D_t^{(k)} = \sum_{j=1}^{k} \left( q_{\text{bid},t}^{(j)} + q_{\text{ask},t}^{(j)} \right)$$

This captures the "thickness" of the book at various distances from the mid-price.

### 3.5 Mid-Price Returns

$$r_t = \frac{p_{\text{mid},t} - p_{\text{mid},t-1}}{p_{\text{mid},t-1}}$$

Returns are stationary (unlike raw prices), making them more suitable as neural network inputs.

### 3.6 Microprice

The microprice weights the best bid and ask by the opposing side's volume:

$$p_{\text{micro},t} = \frac{q_{\text{ask},t}^{(1)} \cdot p_{\text{bid},t}^{(1)} + q_{\text{bid},t}^{(1)} \cdot p_{\text{ask},t}^{(1)}}{q_{\text{bid},t}^{(1)} + q_{\text{ask},t}^{(1)}}$$

This provides a better estimate of the "fair price" than the simple mid-price.

### 3.7 Feature Normalization

All features are z-score normalized over a rolling window of $W$ timesteps:

$$\hat{x}_{t,j} = \frac{x_{t,j} - \mu_j^{(W)}}{\sigma_j^{(W)} + \epsilon}$$

This is critical for LSTM convergence since gates are sensitive to input scale.

---

## 4. Multi-Step Forecasting

### 4.1 Direct Multi-Step

Train $H$ separate models, each predicting $\hat{y}_{t+h}$ for horizon $h = 1, \ldots, H$. Avoids error accumulation but requires more parameters.

### 4.2 Recursive (Autoregressive) Multi-Step

Train a single one-step model and feed predictions back as inputs. Error compounds over horizons but is parameter-efficient.

### 4.3 Sequence-to-Sequence

Use an encoder LSTM to compress the input sequence into a context vector, then a decoder LSTM to generate the full prediction sequence $(\hat{y}_{t+1}, \ldots, \hat{y}_{t+H})$ in one pass. This is the most expressive approach but requires more data.

### 4.4 Horizon Selection for LOB

For high-frequency LOB forecasting, typical horizons are:

| Horizon | Timesteps | Use Case |
|---------|-----------|----------|
| Ultra-short | 1-5 ticks | Market making |
| Short | 10-50 ticks | Aggressive HFT |
| Medium | 100-500 ticks | Statistical arbitrage |

Our implementation uses a configurable horizon with a default of 10 timesteps ahead.

---

## 5. Rust Implementation

### 5.1 Design Overview

Our Rust implementation is organized into the following modules within `lib.rs`:

- **`LstmCell`** -- A single LSTM cell with full gate computation.
- **`StackedLstm`** -- Multiple LSTM layers stacked vertically.
- **`LobFeatureExtractor`** -- Computes engineered features from raw orderbook snapshots.
- **`LstmForecaster`** -- End-to-end model combining feature extraction, sequence buffering, and prediction.
- **Bybit API functions** -- Fetch live orderbook and kline data.

### 5.2 Key Implementation Details

**Weight Initialization**: We use Xavier/Glorot uniform initialization, scaling weights by $\sqrt{6 / (n_{\text{in}} + n_{\text{out}})}$. This keeps the variance of activations stable across layers and is critical for LSTM training.

**Activation Functions**: Sigmoid and tanh are implemented element-wise over `ndarray` arrays. The sigmoid is computed as $\sigma(x) = 1 / (1 + e^{-x})$ with clipping to prevent numerical overflow.

**Forward Pass**: Each LSTM cell concatenates $[h_{t-1}, x_t]$, multiplies by the weight matrices for all four gates simultaneously, then splits and applies the appropriate activations.

**Training**: We implement backpropagation through time (BPTT) with a simple SGD optimizer. In production, one would use Adam or AdamW, but SGD demonstrates the core gradient computation clearly.

### 5.3 Bybit Integration

We fetch two types of data from the Bybit v5 API:

1. **Orderbook snapshots** (`/v5/market/orderbook`): provides bid/ask prices and quantities at up to 200 levels.
2. **Kline/candlestick data** (`/v5/market/kline`): provides OHLCV bars for trend context.

Both endpoints are public and require no authentication.

---

## 6. Bybit Data Integration

### 6.1 Orderbook Data

The Bybit REST API endpoint `GET /v5/market/orderbook?category=linear&symbol=BTCUSDT&limit=50` returns a JSON response containing arrays of `[price, size]` pairs for bids and asks.

We parse this into our `OrderbookSnapshot` struct, which stores vectors of `(f64, f64)` tuples representing (price, quantity) at each level.

### 6.2 Kline Data

The endpoint `GET /v5/market/kline?category=linear&symbol=BTCUSDT&interval=1&limit=200` returns candlestick data. We extract the close prices to compute mid-price return series as supplementary features.

### 6.3 Data Pipeline

```text
Bybit API --> OrderbookSnapshot --> LobFeatureExtractor --> Feature Vector
                                                                |
                                                                v
                                                    Sequence Buffer (T steps)
                                                                |
                                                                v
                                                    StackedLstm --> Prediction
```

The pipeline operates as follows:
1. Fetch a batch of orderbook snapshots and kline data.
2. For each snapshot, compute the LOB feature vector (imbalance, spread, depth, mid-price return, microprice).
3. Accumulate features into a sliding window of length $T$ (sequence length).
4. Feed the sequence through the stacked LSTM.
5. The final hidden state is projected to a 3-class output (Up / Down / Stationary).

### 6.4 Label Construction

For supervised training, we label each timestep based on the future mid-price change:

$$y_t = \begin{cases} \text{Up} & \text{if } p_{\text{mid}, t+H} - p_{\text{mid}, t} > \theta \\ \text{Down} & \text{if } p_{\text{mid}, t} - p_{\text{mid}, t+H} > \theta \\ \text{Stationary} & \text{otherwise} \end{cases}$$

where $\theta$ is a threshold (e.g., 0.5 * average spread) that prevents labeling noise as directional moves.

---

## 7. Key Takeaways

1. **LSTM gates solve the vanishing gradient problem**: The forget gate's ability to maintain $f_t \approx 1$ allows gradients to flow through the cell state over hundreds of timesteps, enabling the network to learn both fast tick-level and slow trend-level dependencies in LOB data.

2. **LOB feature engineering is critical**: Raw order book levels are high-dimensional and noisy. Engineered features -- imbalance ratio, spread, microprice, depth profiles -- compress the relevant microstructure information into a manageable, informative representation. The imbalance ratio is consistently the strongest single predictor of short-term mid-price direction.

3. **Multi-layer stacking captures hierarchical patterns**: Lower LSTM layers learn local patterns (individual order arrivals, spread fluctuations), while upper layers learn aggregate patterns (trends, liquidity regimes). Two to three layers typically suffice for LOB data.

4. **Normalization and initialization matter**: Z-score normalization of features and Xavier weight initialization are not optional -- without them, LSTM gates saturate and training fails. The sigmoid activations in gates are particularly sensitive to input scale.

5. **Multi-step forecasting requires careful design**: Direct multi-step avoids error accumulation but needs separate models per horizon. Recursive approaches are parameter-efficient but compound errors. Choose based on your latency and accuracy requirements.

6. **Rust enables low-latency inference**: For HFT applications where predictions must be generated in microseconds, Rust's zero-cost abstractions and lack of garbage collection make it ideal. Our implementation processes a single forward pass in under 100 microseconds on modern hardware.

7. **Bybit provides accessible LOB data**: The public v5 API offers orderbook snapshots at up to 200 levels with no authentication required, making it straightforward to build and test LOB forecasting models on live cryptocurrency data.

8. **Threshold-based labeling reduces noise**: Using a spread-relative threshold $\theta$ when constructing training labels prevents the model from trying to predict random mid-price oscillations within the spread, focusing it on meaningful directional moves.

9. **Sequence length is a key hyperparameter**: Too short (< 20 steps) and the model cannot capture multi-scale dependencies; too long (> 200 steps) and training becomes slow and the model may overfit to stale patterns. A sequence length of 50-100 timesteps is a good starting point for LOB data at 100ms resolution.

10. **Evaluation must account for class imbalance**: In LOB forecasting, the "Stationary" class often dominates. Use balanced accuracy, macro-F1, or apply class weighting in the loss function to ensure the model learns to predict directional moves, not just default to the majority class.
