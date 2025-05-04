# Stochastic Process Analyzer 📊

A Streamlit-based web application for analyzing and simulating various stochastic processes including Markov Chains, Hidden Markov Models, and Queuing Systems.

## Features

- **Markov Chain Analysis**:
  - Transition matrix calculation
  - Steady-state probabilities
  - Mean first passage times
  - State classification (absorbing, transient, recurrent)
  - Visualization of state transitions

- **Hidden Markov Model Analysis**:
  - Transition and emission probability estimation
  - Forward algorithm for sequence probability
  - Viterbi algorithm for most likely path
  - Visualization of HMM structure

- **Queuing Theory Analysis**:
  - M/M/s queue analysis
  - Key metrics calculation (L, Lq, W, Wq)
  - System stability evaluation
  - Queue simulation

- **Simulation Tools**:
  - Markov chain simulation
  - HMM sequence generation
  - Queuing system simulation

## Installation

1. Clone the repository:
   git clone https://github.com/ibrahim3702/Stochastic-Process-Analyzer.git
   cd stochastic-process-analyzer
2. Install the required dependencies:
   pip install streamlit pandas numpy altair graphviz matplotlib seaborn plotly networkx scipy pillow

## Usage
  Run the application:
      streamlit run app.py
  The application will open in your default web browser at http://localhost:8501
  
  Use the sidebar to navigate between different analysis types:

  Home: Overview of the application

  Markov Chain Analysis: Analyze state transition data

  Hidden Markov Model Analysis: Analyze hidden state and observed event data

  Queuing Theory Analysis: Analyze arrival and service time data

  Simulation: Run simulations of various stochastic processes
