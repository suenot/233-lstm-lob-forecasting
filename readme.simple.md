# Chapter 275: LSTM LOB Forecasting -- Simple Explanation

## What is this about?

Imagine a robot with a very good memory that remembers both recent and old market events. This robot watches a big screen showing all the buy and sell orders that people have placed -- like a scoreboard at an auction. By remembering patterns from the past, the robot tries to guess whether the price will go up or down next.

## The Order Book -- Like a Two-Sided Auction

Picture a marketplace where buyers line up on one side and sellers on the other:

- **Buyers (bids)**: "I want to buy at $99!" "I want to buy at $98!"
- **Sellers (asks)**: "I want to sell at $101!" "I want to sell at $102!"

The **order book** is just a list of all these offers. The space between the highest buyer ($99) and the lowest seller ($101) is called the **spread** -- like a gap in the middle of the marketplace.

## Why Regular Robots Forget

A regular robot (vanilla RNN) is like someone with short-term memory only. If you tell them 50 things, they forget the first 20 by the time they hear the last one. That is a problem because sometimes what happened 5 minutes ago in the market still matters right now.

## LSTM -- The Robot with a Notebook

LSTM stands for "Long Short-Term Memory." Think of it as a robot that carries a notebook:

- **Forget gate**: The robot looks at each old note and decides: "Is this still important?" Old news gets erased; important facts stay.
- **Input gate**: When something new happens, the robot decides: "Should I write this down?" Not everything is worth noting.
- **The notebook (cell state)**: This is the robot's long-term memory. It can keep notes for a very long time without the ink fading.
- **Output gate**: When asked for a prediction, the robot picks which notes to share. It does not dump everything -- just the relevant bits.

This is why LSTM is special: it can remember that a huge sell order happened 200 steps ago and still use that information today.

## What the Robot Watches

Instead of staring at thousands of numbers in the order book, our robot focuses on a few key clues:

1. **Imbalance**: Are there more buyers or sellers? If buyers outnumber sellers, the price might go up.
2. **Spread**: Is the gap between buyers and sellers wide or narrow? A wide gap means uncertainty.
3. **Depth**: How thick are the walls of orders? Thick walls are hard to push through.
4. **Price changes**: Has the price been going up or down recently?
5. **Microprice**: A smarter version of "the price in the middle" that accounts for how big the orders are.

## Making Predictions

The robot collects these clues at each moment in time -- like taking a photo every second. It lines up 50-100 photos in a row and looks at the whole filmstrip. Then it says: "Based on everything I have seen, I think the price will go **up** / **down** / **stay the same**."

## Stacking Robots

One robot is good, but stacking two or three on top of each other is even better:

- **Robot 1** (bottom): Watches tiny details -- individual order changes, small spread wiggles.
- **Robot 2** (middle): Spots medium patterns -- "the imbalance has been shifting for the last minute."
- **Robot 3** (top): Sees the big picture -- "we are in a downtrend."

Each robot passes its summary up to the next one, like a chain of analysts.

## Why Rust?

In trading, speed matters. Rust is a programming language that runs really fast -- like a sports car compared to a regular car. Our robot needs to make decisions in millionths of a second, and Rust helps it do that.

## Real Market Data from Bybit

Bybit is a cryptocurrency exchange where people trade Bitcoin and other coins. It has a free public feed that shows the full order book. Our robot connects to Bybit, watches the BTCUSDT order book in real time, and makes predictions about where the price is heading.

## Key Ideas to Remember

- **LSTM = robot with a great notebook** that remembers important old events and forgets unimportant ones.
- **Order book = auction scoreboard** showing all buyers and sellers.
- **Imbalance is the biggest clue**: if there are way more buyers than sellers, the price usually goes up.
- **Stacking multiple robots** helps catch patterns at different speeds (fast, medium, slow).
- **You need to normalize the data** (make all numbers a similar size) or the robot gets confused.
- **Not every price wiggle matters**: we only care about moves bigger than the spread.
