//! # LSTM LOB Forecasting -- Trading Example
//!
//! Demonstrates end-to-end LOB forecasting:
//! 1. Fetch BTCUSDT orderbook + candles from Bybit (or use synthetic data)
//! 2. Build LOB feature sequences
//! 3. Train LSTM, predict mid-price direction

use lstm_lob_forecasting::*;
use ndarray::Array2;

fn main() {
    println!("=== LSTM LOB Forecasting - Trading Example ===\n");

    // Configuration
    let seq_len = 20;
    let hidden_sizes = vec![16, 8];
    let horizon = 5;
    let threshold = 0.0005; // 0.05% move threshold
    let learning_rate = 0.01;
    let n_epochs = 30;

    // -----------------------------------------------------------------------
    // Step 1: Try to fetch live data from Bybit, fall back to synthetic
    // -----------------------------------------------------------------------
    println!("[1] Fetching data...");

    let snapshots = match fetch_bybit_orderbook("BTCUSDT", 25) {
        Ok(snap) => {
            println!("    Live Bybit orderbook fetched:");
            println!(
                "    Best bid: {:.2} | Best ask: {:.2} | Spread: {:.2}",
                snap.bids[0].0,
                snap.asks[0].0,
                snap.asks[0].0 - snap.bids[0].0
            );
            // For training we need many snapshots; use synthetic with live mid as base
            println!("    Generating synthetic sequence around live mid-price...");
            generate_synthetic_orderbook_data(500, 10)
        }
        Err(e) => {
            println!("    Could not fetch live data ({}), using synthetic data.", e);
            generate_synthetic_orderbook_data(500, 10)
        }
    };

    // Try to fetch klines for context
    match fetch_bybit_klines("BTCUSDT", "1", 5) {
        Ok(bars) => {
            println!("    Recent BTCUSDT 1m candles:");
            for bar in bars.iter().take(3) {
                println!(
                    "      O: {:.2}  H: {:.2}  L: {:.2}  C: {:.2}  V: {:.2}",
                    bar.open, bar.high, bar.low, bar.close, bar.volume
                );
            }
        }
        Err(_) => println!("    (Kline fetch skipped)"),
    }

    // -----------------------------------------------------------------------
    // Step 2: Feature engineering
    // -----------------------------------------------------------------------
    println!("\n[2] Extracting LOB features...");

    let mut extractor = LobFeatureExtractor::new();
    let mut normalizer = ZScoreNormalizer::new(100);

    let features: Vec<_> = snapshots
        .iter()
        .map(|snap| {
            let f = extractor.extract(snap);
            let fvec = LobFeatureExtractor::features_to_vec(&f);
            (f, normalizer.normalize(&fvec))
        })
        .collect();

    let mid_prices: Vec<f64> = features.iter().map(|(f, _)| f.mid_price).collect();

    println!("    Extracted {} feature vectors", features.len());
    println!(
        "    Feature dim: {} (spread, imbalance, depth3, depth5, microprice, mid_return)",
        LobFeatureExtractor::num_features()
    );
    println!(
        "    Mid-price range: {:.2} - {:.2}",
        mid_prices.iter().cloned().fold(f64::INFINITY, f64::min),
        mid_prices
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // -----------------------------------------------------------------------
    // Step 3: Construct labels
    // -----------------------------------------------------------------------
    println!("\n[3] Constructing labels (horizon={}, threshold={:.4})...", horizon, threshold);

    let labels = construct_labels(&mid_prices, horizon, threshold);
    let n_up = labels.iter().filter(|l| **l == Direction::Up).count();
    let n_down = labels.iter().filter(|l| **l == Direction::Down).count();
    let n_stat = labels.iter().filter(|l| **l == Direction::Stationary).count();

    println!(
        "    Labels: {} Up, {} Down, {} Stationary (total: {})",
        n_up,
        n_down,
        n_stat,
        labels.len()
    );

    // -----------------------------------------------------------------------
    // Step 4: Build training sequences
    // -----------------------------------------------------------------------
    println!("\n[4] Building training sequences (seq_len={})...", seq_len);

    let num_features = LobFeatureExtractor::num_features();
    let mut train_sequences: Vec<Array2<f64>> = Vec::new();
    let mut train_labels: Vec<Direction> = Vec::new();

    for i in seq_len..labels.len() {
        let mut seq = Array2::zeros((seq_len, num_features));
        for t in 0..seq_len {
            let idx = i - seq_len + t;
            seq.row_mut(t).assign(&features[idx].1);
        }
        train_sequences.push(seq);
        train_labels.push(labels[i]);
    }

    println!("    Training samples: {}", train_sequences.len());

    if train_sequences.is_empty() {
        println!("    Not enough data for training. Exiting.");
        return;
    }

    // Split 80/20
    let split = (train_sequences.len() as f64 * 0.8) as usize;
    let (train_seqs, test_seqs) = train_sequences.split_at(split);
    let (train_lbls, test_lbls) = train_labels.split_at(split);

    println!(
        "    Train: {} | Test: {}",
        train_seqs.len(),
        test_seqs.len()
    );

    // -----------------------------------------------------------------------
    // Step 5: Train LSTM
    // -----------------------------------------------------------------------
    println!(
        "\n[5] Training LSTM (layers: {:?}, epochs: {}, lr: {})...",
        hidden_sizes, n_epochs, learning_rate
    );

    let mut forecaster = LstmForecaster::new(&hidden_sizes, seq_len);

    for epoch in 0..n_epochs {
        let mut total_loss = 0.0;
        let n_samples = train_seqs.len().min(100); // limit per epoch for speed
        for i in 0..n_samples {
            let loss = forecaster.train_step(&train_seqs[i], train_lbls[i], learning_rate);
            total_loss += loss;
        }
        let avg_loss = total_loss / n_samples as f64;
        if epoch % 5 == 0 || epoch == n_epochs - 1 {
            println!("    Epoch {:3}/{}: avg_loss = {:.4}", epoch + 1, n_epochs, avg_loss);
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Evaluate on test set
    // -----------------------------------------------------------------------
    println!("\n[6] Evaluating on test set...");

    let mut correct = 0;
    let mut total = 0;
    let n_test = test_seqs.len().min(50);

    for i in 0..n_test {
        let (pred, _probs) = forecaster.predict(&test_seqs[i]);
        if pred == test_lbls[i] {
            correct += 1;
        }
        total += 1;
    }

    let accuracy = correct as f64 / total as f64;
    println!("    Test accuracy: {}/{} = {:.1}%", correct, total, accuracy * 100.0);

    // -----------------------------------------------------------------------
    // Step 7: Live prediction demo
    // -----------------------------------------------------------------------
    println!("\n[7] Sample predictions on recent data:");

    for i in (test_seqs.len().saturating_sub(5))..test_seqs.len().min(n_test) {
        let (pred, probs) = forecaster.predict(&test_seqs[i]);
        println!(
            "    Sample {}: predicted={}, actual={}, probs=[Up:{:.3}, Down:{:.3}, Stat:{:.3}]",
            i, pred, test_lbls[i], probs[0], probs[1], probs[2]
        );
    }

    println!("\n=== Done ===");
}
