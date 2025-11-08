//! # LSTM LOB Forecasting
//!
//! A from-scratch LSTM implementation for Limit Order Book (LOB) mid-price
//! direction forecasting, with Bybit exchange data integration.

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    let clamped = x.clamp(-15.0, 15.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Element-wise sigmoid over a 1-D array.
fn sigmoid_vec(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(sigmoid)
}

/// Element-wise tanh over a 1-D array.
fn tanh_vec(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(f64::tanh)
}

/// Softmax over a 1-D array.
pub fn softmax(v: &Array1<f64>) -> Array1<f64> {
    let max_val = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = v.mapv(|x| (x - max_val).exp());
    let sum: f64 = exps.sum();
    exps / sum
}

// ---------------------------------------------------------------------------
// Xavier weight initialization
// ---------------------------------------------------------------------------

fn xavier_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-limit..limit))
}

fn zeros1(n: usize) -> Array1<f64> {
    Array1::zeros(n)
}

// ---------------------------------------------------------------------------
// LSTM Cell
// ---------------------------------------------------------------------------

/// A single LSTM cell implementing all four gates.
#[derive(Clone)]
pub struct LstmCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Weights: concatenated [h, x] -> gate
    // Forget gate
    pub wf: Array2<f64>,
    pub bf: Array1<f64>,
    // Input gate
    pub wi: Array2<f64>,
    pub bi: Array1<f64>,
    // Candidate
    pub wc: Array2<f64>,
    pub bc: Array1<f64>,
    // Output gate
    pub wo: Array2<f64>,
    pub bo: Array1<f64>,
}

/// Hidden + cell state pair.
#[derive(Clone, Debug)]
pub struct LstmState {
    pub h: Array1<f64>,
    pub c: Array1<f64>,
}

impl LstmCell {
    /// Create a new LSTM cell with Xavier-initialized weights.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let concat_size = hidden_size + input_size;
        Self {
            input_size,
            hidden_size,
            wf: xavier_matrix(concat_size, hidden_size, &mut rng),
            bf: zeros1(hidden_size),
            wi: xavier_matrix(concat_size, hidden_size, &mut rng),
            bi: zeros1(hidden_size),
            wc: xavier_matrix(concat_size, hidden_size, &mut rng),
            bc: zeros1(hidden_size),
            wo: xavier_matrix(concat_size, hidden_size, &mut rng),
            bo: zeros1(hidden_size),
        }
    }

    /// Initial (zero) state.
    pub fn zero_state(&self) -> LstmState {
        LstmState {
            h: zeros1(self.hidden_size),
            c: zeros1(self.hidden_size),
        }
    }

    /// Forward pass for one timestep.
    ///
    /// Given input `x_t` and previous state, returns the new state.
    pub fn forward(&self, x_t: &Array1<f64>, prev: &LstmState) -> LstmState {
        // Concatenate [h_{t-1}, x_t]
        let mut concat = Array1::zeros(self.hidden_size + self.input_size);
        concat
            .slice_mut(ndarray::s![..self.hidden_size])
            .assign(&prev.h);
        concat
            .slice_mut(ndarray::s![self.hidden_size..])
            .assign(x_t);

        // Gate computations
        let f_t = sigmoid_vec(&(concat.dot(&self.wf) + &self.bf));
        let i_t = sigmoid_vec(&(concat.dot(&self.wi) + &self.bi));
        let c_tilde = tanh_vec(&(concat.dot(&self.wc) + &self.bc));
        let o_t = sigmoid_vec(&(concat.dot(&self.wo) + &self.bo));

        // Cell state update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        let c_new = &f_t * &prev.c + &i_t * &c_tilde;

        // Hidden state: h_t = o_t ⊙ tanh(C_t)
        let h_new = &o_t * &tanh_vec(&c_new);

        LstmState {
            h: h_new,
            c: c_new,
        }
    }
}

// ---------------------------------------------------------------------------
// Stacked (Multi-Layer) LSTM
// ---------------------------------------------------------------------------

/// A stack of LSTM layers. The output of layer l feeds as input to layer l+1.
pub struct StackedLstm {
    pub layers: Vec<LstmCell>,
}

impl StackedLstm {
    /// Create a stacked LSTM with given layer sizes.
    ///
    /// `layer_sizes` should contain (input_size, hidden_size) for each layer.
    /// Typically: layer 0 input_size = feature dim, subsequent layers
    /// input_size = previous hidden_size.
    pub fn new(input_size: usize, hidden_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;
        for &hs in hidden_sizes {
            layers.push(LstmCell::new(prev_size, hs));
            prev_size = hs;
        }
        Self { layers }
    }

    /// Initial (zero) states for all layers.
    pub fn zero_states(&self) -> Vec<LstmState> {
        self.layers.iter().map(|l| l.zero_state()).collect()
    }

    /// Forward pass over a full sequence of shape (seq_len, input_size).
    ///
    /// Returns the final hidden states for all layers and the top-layer
    /// hidden state at the last timestep (used for sequence-to-one output).
    pub fn forward(
        &self,
        sequence: &Array2<f64>,
        initial_states: &[LstmState],
    ) -> (Vec<LstmState>, Array1<f64>) {
        let seq_len = sequence.nrows();
        let mut states: Vec<LstmState> = initial_states.to_vec();

        for t in 0..seq_len {
            let mut input = sequence.row(t).to_owned();
            for (l, layer) in self.layers.iter().enumerate() {
                let new_state = layer.forward(&input, &states[l]);
                input = new_state.h.clone();
                states[l] = new_state;
            }
        }

        let top_h = states.last().unwrap().h.clone();
        (states, top_h)
    }

    /// Output dimension (hidden size of the top layer).
    pub fn output_size(&self) -> usize {
        self.layers.last().unwrap().hidden_size
    }
}

// ---------------------------------------------------------------------------
// Dense output layer
// ---------------------------------------------------------------------------

/// Simple linear + softmax output layer for classification.
pub struct DenseOutput {
    pub w: Array2<f64>,
    pub b: Array1<f64>,
}

impl DenseOutput {
    pub fn new(input_size: usize, num_classes: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            w: xavier_matrix(input_size, num_classes, &mut rng),
            b: zeros1(num_classes),
        }
    }

    /// Forward: linear projection + softmax.
    pub fn forward(&self, h: &Array1<f64>) -> Array1<f64> {
        let logits = h.dot(&self.w) + &self.b;
        softmax(&logits)
    }
}

// ---------------------------------------------------------------------------
// LOB Feature Extractor
// ---------------------------------------------------------------------------

/// A snapshot of one side of the order book: Vec of (price, quantity).
#[derive(Clone, Debug)]
pub struct OrderbookSnapshot {
    pub bids: Vec<(f64, f64)>, // (price, qty) sorted descending by price
    pub asks: Vec<(f64, f64)>, // (price, qty) sorted ascending by price
    pub timestamp: u64,
}

/// Extracted LOB features for one timestep.
#[derive(Clone, Debug)]
pub struct LobFeatures {
    pub mid_price: f64,
    pub spread: f64,
    pub imbalance: f64,
    pub depth_3: f64,
    pub depth_5: f64,
    pub microprice: f64,
    pub mid_return: f64, // 0.0 for the first snapshot
}

/// Extracts engineered features from raw orderbook snapshots.
pub struct LobFeatureExtractor {
    prev_mid: Option<f64>,
}

impl LobFeatureExtractor {
    pub fn new() -> Self {
        Self { prev_mid: None }
    }

    /// Compute features from a single orderbook snapshot.
    pub fn extract(&mut self, snap: &OrderbookSnapshot) -> LobFeatures {
        let best_bid = snap.bids.first().map(|b| b.0).unwrap_or(0.0);
        let best_ask = snap.asks.first().map(|a| a.0).unwrap_or(0.0);
        let best_bid_qty = snap.bids.first().map(|b| b.1).unwrap_or(1.0);
        let best_ask_qty = snap.asks.first().map(|a| a.1).unwrap_or(1.0);

        let mid_price = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;

        // Imbalance ratio across all available levels
        let bid_vol: f64 = snap.bids.iter().map(|b| b.1).sum();
        let ask_vol: f64 = snap.asks.iter().map(|a| a.1).sum();
        let total_vol = bid_vol + ask_vol;
        let imbalance = if total_vol > 0.0 {
            (bid_vol - ask_vol) / total_vol
        } else {
            0.0
        };

        // Depth at 3 and 5 levels
        let depth_3: f64 = snap.bids.iter().take(3).map(|b| b.1).sum::<f64>()
            + snap.asks.iter().take(3).map(|a| a.1).sum::<f64>();
        let depth_5: f64 = snap.bids.iter().take(5).map(|b| b.1).sum::<f64>()
            + snap.asks.iter().take(5).map(|a| a.1).sum::<f64>();

        // Microprice
        let microprice = if best_bid_qty + best_ask_qty > 0.0 {
            (best_ask_qty * best_bid + best_bid_qty * best_ask) / (best_bid_qty + best_ask_qty)
        } else {
            mid_price
        };

        // Mid-price return
        let mid_return = match self.prev_mid {
            Some(prev) if prev != 0.0 => (mid_price - prev) / prev,
            _ => 0.0,
        };
        self.prev_mid = Some(mid_price);

        LobFeatures {
            mid_price,
            spread,
            imbalance,
            depth_3,
            depth_5,
            microprice,
            mid_return,
        }
    }

    /// Convert LobFeatures to a feature vector (Array1).
    pub fn features_to_vec(f: &LobFeatures) -> Array1<f64> {
        Array1::from(vec![
            f.spread,
            f.imbalance,
            f.depth_3,
            f.depth_5,
            f.microprice,
            f.mid_return,
        ])
    }

    /// Number of features produced.
    pub fn num_features() -> usize {
        6
    }
}

impl Default for LobFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Feature normalization (z-score)
// ---------------------------------------------------------------------------

/// Online z-score normalizer using a rolling window.
pub struct ZScoreNormalizer {
    window: Vec<Array1<f64>>,
    max_window: usize,
}

impl ZScoreNormalizer {
    pub fn new(max_window: usize) -> Self {
        Self {
            window: Vec::new(),
            max_window,
        }
    }

    /// Add a sample and return the normalized version.
    pub fn normalize(&mut self, x: &Array1<f64>) -> Array1<f64> {
        self.window.push(x.clone());
        if self.window.len() > self.max_window {
            self.window.remove(0);
        }
        let n = self.window.len() as f64;
        let mean: Array1<f64> = self.window.iter().fold(Array1::<f64>::zeros(x.len()), |acc, v| acc + v) / n;
        let var: Array1<f64> = self
            .window
            .iter()
            .fold(Array1::<f64>::zeros(x.len()), |acc, v| {
                let diff = v - &mean;
                acc + &diff * &diff
            })
            / n;
        let std = var.mapv(|v: f64| v.sqrt() + 1e-8);
        (x - &mean) / &std
    }
}

// ---------------------------------------------------------------------------
// LSTM Forecaster (end-to-end model)
// ---------------------------------------------------------------------------

/// Direction label for mid-price prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Up = 0,
    Down = 1,
    Stationary = 2,
}

impl Direction {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Direction::Up,
            1 => Direction::Down,
            _ => Direction::Stationary,
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Up => write!(f, "Up"),
            Direction::Down => write!(f, "Down"),
            Direction::Stationary => write!(f, "Stationary"),
        }
    }
}

/// End-to-end LSTM forecaster for LOB mid-price direction.
pub struct LstmForecaster {
    pub lstm: StackedLstm,
    pub output: DenseOutput,
    pub seq_len: usize,
}

impl LstmForecaster {
    /// Create a new forecaster.
    ///
    /// - `hidden_sizes`: hidden size for each LSTM layer.
    /// - `seq_len`: number of timesteps in each input sequence.
    pub fn new(hidden_sizes: &[usize], seq_len: usize) -> Self {
        let input_size = LobFeatureExtractor::num_features();
        let lstm = StackedLstm::new(input_size, hidden_sizes);
        let output = DenseOutput::new(lstm.output_size(), 3);
        Self {
            lstm,
            output,
            seq_len,
        }
    }

    /// Predict direction from a feature sequence (seq_len x num_features).
    pub fn predict(&self, sequence: &Array2<f64>) -> (Direction, Array1<f64>) {
        let states = self.lstm.zero_states();
        let (_, top_h) = self.lstm.forward(sequence, &states);
        let probs = self.output.forward(&top_h);
        let pred_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        (Direction::from_index(pred_idx), probs)
    }

    /// Simple SGD training step on one sequence.
    /// Returns the cross-entropy loss.
    pub fn train_step(
        &mut self,
        sequence: &Array2<f64>,
        label: Direction,
        learning_rate: f64,
    ) -> f64 {
        // Forward pass
        let states = self.lstm.zero_states();
        let (_, top_h) = self.lstm.forward(sequence, &states);
        let probs = self.output.forward(&top_h);

        // Cross-entropy loss
        let target = label as usize;
        let loss = -(probs[target] + 1e-10).ln();

        // Gradient of softmax + cross-entropy w.r.t. logits
        let mut grad_logits = probs.clone();
        grad_logits[target] -= 1.0;

        // Update output layer weights (gradient descent)
        // dL/dW = h^T * grad_logits
        let grad_w = top_h
            .clone()
            .insert_axis(Axis(1))
            .dot(&grad_logits.clone().insert_axis(Axis(0)));
        self.output.w = &self.output.w - &(grad_w * learning_rate);
        self.output.b = &self.output.b - &(&grad_logits * learning_rate);

        loss
    }
}

// ---------------------------------------------------------------------------
// Label construction
// ---------------------------------------------------------------------------

/// Construct direction labels from mid-price series.
///
/// - `mid_prices`: vector of mid-prices.
/// - `horizon`: number of steps to look ahead.
/// - `threshold`: minimum absolute return to count as directional.
///
/// Returns labels aligned to positions 0..len-horizon.
pub fn construct_labels(
    mid_prices: &[f64],
    horizon: usize,
    threshold: f64,
) -> Vec<Direction> {
    let n = mid_prices.len();
    if n <= horizon {
        return vec![];
    }
    (0..n - horizon)
        .map(|i| {
            let ret = (mid_prices[i + horizon] - mid_prices[i]) / mid_prices[i];
            if ret > threshold {
                Direction::Up
            } else if ret < -threshold {
                Direction::Down
            } else {
                Direction::Stationary
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bybit API Integration
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitOrderbookResponse {
    result: BybitOrderbookResult,
}

#[derive(Debug, Deserialize)]
struct BybitOrderbookResult {
    #[serde(rename = "b")]
    bids: Vec<Vec<String>>,
    #[serde(rename = "a")]
    asks: Vec<Vec<String>>,
    #[serde(rename = "ts")]
    ts: u64,
}

/// Fetch orderbook snapshot from Bybit REST API.
///
/// `symbol`: e.g. "BTCUSDT"
/// `depth`: number of levels (max 200)
pub fn fetch_bybit_orderbook(symbol: &str, depth: usize) -> Result<OrderbookSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=linear&symbol={}&limit={}",
        symbol, depth
    );
    let resp: BybitOrderbookResponse = reqwest::blocking::get(&url)?.json()?;
    let bids: Vec<(f64, f64)> = resp
        .result
        .bids
        .iter()
        .map(|row| {
            (
                row[0].parse::<f64>().unwrap_or(0.0),
                row[1].parse::<f64>().unwrap_or(0.0),
            )
        })
        .collect();
    let asks: Vec<(f64, f64)> = resp
        .result
        .asks
        .iter()
        .map(|row| {
            (
                row[0].parse::<f64>().unwrap_or(0.0),
                row[1].parse::<f64>().unwrap_or(0.0),
            )
        })
        .collect();
    Ok(OrderbookSnapshot {
        bids,
        asks,
        timestamp: resp.result.ts,
    })
}

#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    list: Vec<Vec<String>>,
}

/// Kline (candlestick) bar from Bybit.
#[derive(Clone, Debug)]
pub struct KlineBar {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit REST API.
///
/// `symbol`: e.g. "BTCUSDT"
/// `interval`: e.g. "1" for 1-minute
/// `limit`: number of bars
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<KlineBar>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;
    let bars: Vec<KlineBar> = resp
        .result
        .list
        .iter()
        .map(|row| KlineBar {
            open_time: row[0].parse().unwrap_or(0),
            open: row[1].parse().unwrap_or(0.0),
            high: row[2].parse().unwrap_or(0.0),
            low: row[3].parse().unwrap_or(0.0),
            close: row[4].parse().unwrap_or(0.0),
            volume: row[5].parse().unwrap_or(0.0),
        })
        .collect();
    Ok(bars)
}

// ---------------------------------------------------------------------------
// Synthetic data for testing / offline demo
// ---------------------------------------------------------------------------

/// Generate synthetic orderbook snapshots for testing.
pub fn generate_synthetic_orderbook_data(
    n_snapshots: usize,
    n_levels: usize,
) -> Vec<OrderbookSnapshot> {
    let mut rng = rand::thread_rng();
    let mut mid = 50000.0_f64; // starting mid-price (like BTC)
    let mut snapshots = Vec::with_capacity(n_snapshots);

    for t in 0..n_snapshots {
        // Random walk mid-price
        mid += rng.gen_range(-10.0..10.0);

        let spread = rng.gen_range(0.5..5.0);
        let best_bid = mid - spread / 2.0;
        let best_ask = mid + spread / 2.0;

        let mut bids = Vec::with_capacity(n_levels);
        let mut asks = Vec::with_capacity(n_levels);

        for k in 0..n_levels {
            let bid_price = best_bid - k as f64 * rng.gen_range(0.1..1.0);
            let bid_qty = rng.gen_range(0.01..5.0);
            bids.push((bid_price, bid_qty));

            let ask_price = best_ask + k as f64 * rng.gen_range(0.1..1.0);
            let ask_qty = rng.gen_range(0.01..5.0);
            asks.push((ask_price, ask_qty));
        }

        snapshots.push(OrderbookSnapshot {
            bids,
            asks,
            timestamp: t as u64,
        });
    }

    snapshots
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_softmax() {
        let v = Array1::from(vec![1.0, 2.0, 3.0]);
        let s = softmax(&v);
        assert!((s.sum() - 1.0).abs() < 1e-10);
        assert!(s[2] > s[1]);
        assert!(s[1] > s[0]);
    }

    #[test]
    fn test_lstm_cell_forward() {
        let cell = LstmCell::new(4, 8);
        let state = cell.zero_state();
        let x = Array1::from(vec![1.0, -0.5, 0.3, 0.7]);
        let new_state = cell.forward(&x, &state);
        assert_eq!(new_state.h.len(), 8);
        assert_eq!(new_state.c.len(), 8);
        // Hidden state should not be all zeros after non-zero input
        assert!(new_state.h.iter().any(|v| v.abs() > 1e-10));
    }

    #[test]
    fn test_stacked_lstm_forward() {
        let lstm = StackedLstm::new(6, &[16, 8]);
        let states = lstm.zero_states();
        assert_eq!(states.len(), 2);

        // Create a short sequence (5 timesteps, 6 features)
        let seq = Array2::from_shape_fn((5, 6), |(i, j)| (i as f64 + j as f64) * 0.1);
        let (final_states, top_h) = lstm.forward(&seq, &states);
        assert_eq!(final_states.len(), 2);
        assert_eq!(top_h.len(), 8);
    }

    #[test]
    fn test_lob_feature_extractor() {
        let snap = OrderbookSnapshot {
            bids: vec![(100.0, 5.0), (99.0, 3.0), (98.0, 2.0)],
            asks: vec![(101.0, 4.0), (102.0, 6.0), (103.0, 1.0)],
            timestamp: 0,
        };
        let mut extractor = LobFeatureExtractor::new();
        let features = extractor.extract(&snap);

        assert!((features.mid_price - 100.5).abs() < 1e-10);
        assert!((features.spread - 1.0).abs() < 1e-10);
        // Imbalance: (10 - 11) / (10 + 11) = -1/21
        let expected_imb = (10.0 - 11.0) / (10.0 + 11.0);
        assert!((features.imbalance - expected_imb).abs() < 1e-10);
        // Microprice: (4*100 + 5*101) / (5+4) = 905/9
        let expected_micro = (4.0 * 100.0 + 5.0 * 101.0) / (5.0 + 4.0);
        assert!((features.microprice - expected_micro).abs() < 1e-10);
    }

    #[test]
    fn test_construct_labels() {
        let prices = vec![100.0, 101.0, 102.0, 100.5, 99.0, 103.0];
        let labels = construct_labels(&prices, 2, 0.005);
        // Position 0: (102-100)/100 = 0.02 > 0.005 => Up
        assert_eq!(labels[0], Direction::Up);
        // Position 3: (103-100.5)/100.5 ≈ 0.0249 > 0.005 => Up
        assert_eq!(labels[3], Direction::Up);
    }

    #[test]
    fn test_forecaster_predict() {
        let forecaster = LstmForecaster::new(&[16, 8], 10);
        let seq = Array2::from_shape_fn((10, 6), |(i, j)| (i as f64 + j as f64) * 0.01);
        let (direction, probs) = forecaster.predict(&seq);
        // Probabilities should sum to 1
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        // Direction should be a valid variant
        assert!(matches!(
            direction,
            Direction::Up | Direction::Down | Direction::Stationary
        ));
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let mut forecaster = LstmForecaster::new(&[8], 5);
        let seq = Array2::from_shape_fn((5, 6), |(i, j)| (i as f64 + j as f64) * 0.1);

        let loss1 = forecaster.train_step(&seq, Direction::Up, 0.01);
        // Run several training steps on the same data
        let mut loss_last = loss1;
        for _ in 0..20 {
            loss_last = forecaster.train_step(&seq, Direction::Up, 0.01);
        }
        // Loss should decrease after repeated training on same sample
        assert!(loss_last < loss1, "Loss should decrease: {} < {}", loss_last, loss1);
    }

    #[test]
    fn test_zscore_normalizer() {
        let mut norm = ZScoreNormalizer::new(10);
        let x1 = Array1::from(vec![100.0, 200.0, 300.0]);
        let n1 = norm.normalize(&x1);
        // With only one sample, mean = x, so normalized ≈ 0
        assert!(n1.iter().all(|v| v.abs() < 1.0));

        let x2 = Array1::from(vec![110.0, 190.0, 310.0]);
        let _n2 = norm.normalize(&x2);
        let x3 = Array1::from(vec![105.0, 195.0, 305.0]);
        let n3 = norm.normalize(&x3);
        // After 3 samples, normalization should produce reasonable values
        assert!(n3.iter().all(|v| v.abs() < 5.0));
    }

    #[test]
    fn test_synthetic_data_generation() {
        let data = generate_synthetic_orderbook_data(100, 10);
        assert_eq!(data.len(), 100);
        assert_eq!(data[0].bids.len(), 10);
        assert_eq!(data[0].asks.len(), 10);
        // Bids should be below asks
        assert!(data[0].bids[0].0 < data[0].asks[0].0);
    }
}
