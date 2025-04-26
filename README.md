# Housing Price Prediction Neural Network 🏠

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.0-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8.0-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-2.9.0-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23.5-green.svg)](https://numpy.org/)

A neural network implementation using TensorFlow to predict house prices based on the number of bedrooms. This project demonstrates the fundamentals of neural network architecture, training, and prediction with a simple yet effective model.

![Housing Price Prediction Model](housing-model-visualization.png)

---

## Table of Contents 📋
- [Project Overview](#project-overview-🔎)
- [Model Architecture](#model-architecture-🧠)
- [Dataset](#dataset-📊)
- [Training Process](#training-process-🔄)
- [Results](#results-📈)
- [Installation & Usage](#installation--usage-🚀)
- [Key Learnings](#key-learnings-🔍)
- [Future Improvements](#future-improvements-🔮)

---

## Project Overview 🔎

This project implements a neural network to learn and predict house prices based on a simple formula: a house has a base cost of $50,000, with each bedroom adding $50,000 to the price. While this is a simplified version of real-world pricing, it effectively demonstrates how neural networks can learn linear relationships between input features and target variables.

**Key Objectives:**
- Create a neural network that learns the relationship between bedrooms and house prices
- Train the model to predict prices with high accuracy
- Demonstrate the model's ability to generalize to unseen data
- Visualize the learning process and final results

---

## Model Architecture 🧠

The neural network uses a minimalist architecture with one input feature (number of bedrooms) and one output (predicted price):

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mse')
