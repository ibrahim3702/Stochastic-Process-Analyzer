import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import StringIO
import time
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from PIL import Image
from math import factorial

# Set page config for better layout
st.set_page_config(
    page_title="Stochastic Process Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color schemes
primary_color = "#1f77b4"
secondary_color = "#ff7f0e"
accent_color = "#2ca02c"
dark_color = "#2c3e50"
light_color = "#ecf0f1"

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            padding: 1.5rem;
            background-color: #ffffff;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 2.3rem;
        }
        h2 {
            color: #2980b9;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.8rem;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
            font-size: 1.4rem;
        }
        .stAlert {
            border-radius: 10px;
        }
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background: #0E1117;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3498db;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 1rem;
            color: #7f8c8d;
        }
        .dashboard-card {
            background: #0E1117;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .dashboard-card h3 {
            margin-top: 0;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
        }
        .info-text {
            background-color: #0E1117;
            border-left: 4px solid #2196f3;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .stButton > button {
            border-radius: 5px;
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            background-color: #2980b9;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        .tab-content {
            padding: 20px;
            background-color: white;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        .simulation-controls {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .upload-section {
            background-color: #f1f9ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            height: auto;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3498db !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Function to calculate Markov Chain statistics
def markov_chain_analysis(df, state_col='current_state', next_state_col='next_state'):
    """
    Analyzes a Markov Chain from the given dataframe.
    """
    if df.empty:
        return None

    if state_col not in df.columns or next_state_col not in df.columns:
        return None

    # 1. Identify States
    states = sorted(list(set(df[state_col]) | set(df[next_state_col])))
    num_states = len(states)

    if num_states == 0:
        return None

    # 2. Create Transition Matrix
    transition_matrix = pd.DataFrame(0, index=states, columns=states)
    for _, row in df.iterrows():
        current_state = row[state_col]
        next_state = row[next_state_col]
        transition_matrix.loc[current_state, next_state] += 1

    # Convert counts to probabilities
    for state in states:
        row_sum = transition_matrix.loc[state].sum()
        if row_sum > 0:
            transition_matrix.loc[state] = transition_matrix.loc[state] / row_sum
        else:
             transition_matrix.loc[state] = 0  # handle absorbing states

    # 3. Steady-State Probabilities
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state_eigenvector = eigenvectors[:, closest_to_one_idx]
        steady_state_probs = np.abs(steady_state_eigenvector) / np.sum(np.abs(steady_state_eigenvector))
        steady_state_probs_df = pd.DataFrame(steady_state_probs, index=states, columns=['Probability'])
    except np.linalg.LinAlgError:
        steady_state_probs_df = pd.DataFrame(np.nan, index=states, columns=['Probability'])

    # 4. Average Passage Time (Mean First Passage Time)
    passage_times = pd.DataFrame(np.inf, index=states, columns=states)
    for i, origin_state in enumerate(states):
        for j, destination_state in enumerate(states):
            if origin_state == destination_state:
                passage_times.loc[origin_state, destination_state] = 0
            else:
                # solve system of equations
                A = transition_matrix.copy()
                A.loc[destination_state, :] = 0
                A.loc[destination_state, destination_state] = 1
                b = -np.ones(len(states))
                b[states.index(destination_state)] = 0
                try:
                    passage_times.loc[origin_state, destination_state] = np.linalg.solve(A,b)[states.index(origin_state)]
                except np.linalg.LinAlgError:
                     passage_times.loc[origin_state, destination_state] = np.inf

    # 5. Average Recurrence Time
    recurrence_times = pd.Series(np.inf, index=states)
    for state in states:
        if steady_state_probs_df.loc[state, 'Probability'] > 0:
            recurrence_times[state] = 1 / steady_state_probs_df.loc[state, 'Probability']
        else:
            recurrence_times[state] = np.inf
            
    # 6. Calculate Expected Number of Steps for Absorption
    # Only applicable if there are absorbing states
    absorbing_states = [state for state in states if transition_matrix.loc[state, state] == 1 and 
                        transition_matrix.loc[state].sum() == 1]
    
    absorption_steps = {}
    if absorbing_states:
        transient_states = [state for state in states if state not in absorbing_states]
        if transient_states:
            # Create Q matrix (transitions between transient states)
            Q = transition_matrix.loc[transient_states, transient_states]
            # Calculate N = (I-Q)^(-1)
            I = np.identity(len(transient_states))
            try:
                N = np.linalg.inv(I - Q.values)
                # Expected steps to absorption for each transient state
                absorption_steps = {state: N[i, :].sum() for i, state in enumerate(transient_states)}
            except np.linalg.LinAlgError:
                absorption_steps = {state: np.inf for state in transient_states}
    
    # 7. Calculate state transition frequencies
    state_counts = df[state_col].value_counts().to_dict()
    total_transitions = len(df)
    state_frequencies = {state: count/total_transitions for state, count in state_counts.items()}
    
    # 8. Calculate period of each state if possible
    periods = {}
    G = nx.DiGraph()
    for i, row in transition_matrix.iterrows():
        for j, prob in enumerate(row):
            if prob > 0:
                G.add_edge(i, states[j])
    
    for state in states:
        try:
            # Check if state is part of a cycle
            if nx.has_path(G, state, state):
                cycles = []
                for path in nx.all_simple_paths(G, state, state):
                    if len(path) > 1:  # Ignore self-loops
                        cycles.append(len(path))
                if cycles:
                    periods[state] = np.gcd.reduce(cycles)
                else:
                    periods[state] = 1 if transition_matrix.loc[state, state] > 0 else float('inf')
            else:
                periods[state] = float('inf')
        except nx.NetworkXNoPath:
            periods[state] = float('inf')

    return {
        'transition_matrix': transition_matrix,
        'steady_state_probabilities': steady_state_probs_df,
        'average_passage_times': passage_times,
        'average_recurrence_times': recurrence_times,
        'states': states,
        'absorption_steps': absorption_steps,
        'absorbing_states': absorbing_states,
        'state_frequencies': state_frequencies,
        'periods': periods
    }

# Function to perform Hidden Markov Model analysis
def hidden_markov_model_analysis(df, hidden_state_col='hidden_state', observed_event_col='observed_event'):
    """
    Analyzes a Hidden Markov Model from the given dataframe.
    """
    if df.empty:
        return None

    if hidden_state_col not in df.columns or observed_event_col not in df.columns:
        return None

    # 1. Identify Hidden States and Observed Events
    hidden_states = sorted(list(set(df[hidden_state_col])))
    observed_events = sorted(list(set(df[observed_event_col])))
    num_hidden_states = len(hidden_states)
    num_observed_events = len(observed_events)

    if num_hidden_states == 0 or num_observed_events == 0:
        return None

    # 2. Calculate Transition Probabilities (P(S_t+1 | S_t))
    transition_counts = pd.DataFrame(0, index=hidden_states, columns=hidden_states)
    for i in range(len(df) - 1):
        current_state = df.loc[i, hidden_state_col]
        next_state = df.loc[i + 1, hidden_state_col]
        transition_counts.loc[current_state, next_state] += 1
    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    # 3. Calculate Emission Probabilities (P(O_t | S_t))
    emission_counts = pd.DataFrame(0, index=hidden_states, columns=observed_events)
    for _, row in df.iterrows():
        state = row[hidden_state_col]
        event = row[observed_event_col]
        emission_counts.loc[state, event] += 1
    emission_probabilities = emission_counts.div(emission_counts.sum(axis=1), axis=0).fillna(0)

    # 4. Steady-State Probabilities of Hidden States
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_probabilities.T)
        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state_eigenvector = eigenvectors[:, closest_to_one_idx]
        steady_state_probs = np.abs(steady_state_eigenvector) / np.sum(np.abs(steady_state_eigenvector))
        steady_state_probs_df = pd.DataFrame(steady_state_probs, index=hidden_states, columns=['Probability'])
    except np.linalg.LinAlgError:
        steady_state_probs_df = pd.DataFrame(np.nan, index=hidden_states, columns=['Probability'])

    # 5. Prepare initial probabilities for the forward and viterbi algorithms
    initial_state_counts = df.groupby(hidden_state_col).size()
    initial_probs = initial_state_counts / initial_state_counts.sum()
    initial_probs = initial_probs.reindex(transition_probabilities.index, fill_value=0)

    # 6. Forward Algorithm
    def forward_algorithm(observations, initial_probs, transition_probs, emission_probs):
        num_states = len(transition_probs)
        T = len(observations)
        alpha = np.zeros((num_states, T))
        
        # For visualization: store all alphas
        all_alphas = []

        # Initialization
        for s_idx, state in enumerate(transition_probs.index):
            if observations[0] in emission_probs.columns:
                alpha[s_idx, 0] = initial_probs[state] * emission_probs.loc[state, observations[0]]
            else:
                alpha[s_idx, 0] = 0
        
        all_alphas.append(alpha.copy())

        # Induction
        for t in range(1, T):
            for s_idx, state in enumerate(transition_probs.index):
                if observations[t] in emission_probs.columns:
                    alpha[s_idx, t] = sum(alpha[prev_s_idx, t - 1] * transition_probs.iloc[prev_s_idx, s_idx] * emission_probs.loc[state, observations[t]]
                                            for prev_s_idx in range(num_states))
                else:
                    alpha[s_idx, t] = 0
            all_alphas.append(alpha.copy())

        # Termination
        probability_of_sequence = np.sum(alpha[:, T - 1])
        return probability_of_sequence, all_alphas

    # 7. Viterbi Algorithm
    def viterbi_algorithm(observations, initial_probs, transition_probs, emission_probs):
        num_states = len(transition_probs)
        T = len(observations)
        viterbi = np.zeros((num_states, T))
        backpointer = np.zeros((num_states, T), dtype=int)
        
        # For visualization
        all_viterbi_values = []
        all_backpointers = []

        # Initialization
        for s_idx, state in enumerate(transition_probs.index):
            if observations[0] in emission_probs.columns:
                viterbi[s_idx, 0] = initial_probs[state] * emission_probs.loc[state, observations[0]]
            else:
                viterbi[s_idx, 0] = 0
        
        all_viterbi_values.append(viterbi.copy())
        all_backpointers.append(backpointer[:, 0].copy())

        # Recursion
        for t in range(1, T):
            for s_idx, state in enumerate(transition_probs.index):
                if observations[t] in emission_probs.columns:
                    probs = [viterbi[prev_s_idx, t - 1] * transition_probs.iloc[prev_s_idx, s_idx] for prev_s_idx in range(num_states)]
                    max_val = max(probs) * emission_probs.loc[state, observations[t]] if max(probs) > 0 else 0
                    viterbi[s_idx, t] = max_val
                    backpointer[s_idx, t] = np.argmax(probs) if max(probs) > 0 else 0
                else:
                    viterbi[s_idx, t] = 0
                    backpointer[s_idx, t] = 0
            
            all_viterbi_values.append(viterbi.copy())
            all_backpointers.append(backpointer[:, t].copy())

        # Termination
        best_path_end = np.argmax(viterbi[:, T - 1])
        best_path = [hidden_states[best_path_end]]
        best_path_indices = [best_path_end]
        
        # Backtracking
        for t in range(T - 1, 0, -1):
            best_path_end = backpointer[best_path_end, t]
            best_path.insert(0, hidden_states[best_path_end])
            best_path_indices.insert(0, best_path_end)

        return best_path, all_viterbi_values, all_backpointers, best_path_indices

    # 8. Additional - Joint probability table
    joint_probability = pd.DataFrame(0, index=hidden_states, columns=observed_events)
    for state in hidden_states:
        state_prob = steady_state_probs_df.loc[state, 'Probability']
        for event in observed_events:
            if state in emission_probabilities.index and event in emission_probabilities.columns:
                joint_probability.loc[state, event] = state_prob * emission_probabilities.loc[state, event]
    
    # 9. Calculate observation probabilities
    observation_probabilities = {}
    for event in observed_events:
        prob = 0
        for state in hidden_states:
            if state in emission_probabilities.index and event in emission_probabilities.columns:
                prob += steady_state_probs_df.loc[state, 'Probability'] * emission_probabilities.loc[state, event]
        observation_probabilities[event] = prob
    
    return {
        'transition_probabilities': transition_probabilities,
        'emission_probabilities': emission_probabilities,
        'steady_state_probabilities': steady_state_probs_df,
        'forward_algorithm': forward_algorithm,
        'viterbi_algorithm': viterbi_algorithm,
        'hidden_states': hidden_states,
        'observed_events': observed_events,
        'initial_probs': initial_probs,
        'joint_probability': joint_probability,
        'observation_probabilities': observation_probabilities
    }

# Function to perform Queuing Theory analysis (M/M/s)
def queuing_theory_analysis(df, arrival_time_col='arrival_time_minutes', service_time_col='service_time_minutes', servers=1):
    """
    Analyzes a queuing system (M/M/s) from the given dataframe.
    """
    if df.empty:
        return None

    if arrival_time_col not in df.columns or service_time_col not in df.columns:
        return None

    # 1. Calculate Arrival Rate (lambda)
    arrival_times = df[arrival_time_col].sort_values().reset_index(drop=True)
    inter_arrival_times = arrival_times.diff().dropna()
    mean_inter_arrival_time = inter_arrival_times.mean()
    arrival_rate = 1 / mean_inter_arrival_time if mean_inter_arrival_time > 0 else 0
    
    # Calculate variance of inter-arrival times for coefficient of variation
    var_inter_arrival_time = inter_arrival_times.var() if len(inter_arrival_times) > 1 else 0
    cv_inter_arrival = np.sqrt(var_inter_arrival_time) / mean_inter_arrival_time if mean_inter_arrival_time > 0 else 0

    # 2. Calculate Service Rate (mu)
    mean_service_time = df[service_time_col].mean()
    service_rate = 1 / mean_service_time if mean_service_time > 0 else 0
    
    # Calculate variance of service times for coefficient of variation
    var_service_time = df[service_time_col].var() if len(df) > 1 else 0
    cv_service = np.sqrt(var_service_time) / mean_service_time if mean_service_time > 0 else 0

    # 3. Calculate System Metrics for M/M/s
    if service_rate <= 0 or arrival_rate <= 0:
        return None

    # Traffic intensity (rho)
    rho = arrival_rate / (servers * service_rate)
    
    if rho >= 1:  # System is unstable
        return {
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'utilization': rho,
            'servers': servers,
            'mean_inter_arrival_time': mean_inter_arrival_time,
            'mean_service_time': mean_service_time,
            'cv_inter_arrival': cv_inter_arrival,
            'cv_service': cv_service,
            'stable': False
        }
    
    # Calculate P0 (probability of zero customers)
    sum_term = sum([(servers * rho)**n / factorial(n) for n in range(servers)])
    last_term = (servers * rho)**servers / (factorial(servers) * (1 - rho))
    p0 = 1 / (sum_term + last_term)
    
    # Calculate Lq (average customers in queue)
    lq = p0 * (servers * rho)**servers * rho / (factorial(servers) * (1 - rho)**2)
    
    # Calculate L (average customers in system)
    l = lq + servers * rho
    
    # Calculate Wq (average time in queue)
    wq = lq / arrival_rate
    
    # Calculate W (average time in system)
    w = wq + 1/service_rate
    
    # Calculate probability of waiting
    p_wait = p0 * (servers * rho)**servers / (factorial(servers) * (1 - rho))
    
    # Calculate probability of n or more customers
    def p_n_or_more(n):
        if n < servers:
            return sum([p0 * (servers * rho)**k / factorial(k) for k in range(n, servers)]) + \
                   p0 * (servers * rho)**servers / (factorial(servers) * (1 - rho))
        else:
            return p0 * (servers * rho)**servers * rho**(n-servers) / (factorial(servers) * (1 - rho))
    
    # Calculate probabilities for different queue lengths
    queue_probs = {i: p_n_or_more(i) - p_n_or_more(i+1) for i in range(10)}
    
    # Calculate server utilization (different from traffic intensity for multi-server)
    utilization = rho
    
    return {
        'arrival_rate': arrival_rate,
        'service_rate': service_rate,
        'utilization': utilization,
        'servers': servers,
        'probability_of_zero_customers': p0,
        'average_customers_in_system': l,
        'average_customers_in_queue': lq,
        'average_time_in_system': w,
        'average_time_in_queue': wq,
        'probability_of_waiting': p_wait,
        'queue_length_probabilities': queue_probs,
        'mean_inter_arrival_time': mean_inter_arrival_time,
        'mean_service_time': mean_service_time,
        'cv_inter_arrival': cv_inter_arrival,
        'cv_service': cv_service,
        'stable': True
    }

def create_diagram(transition_matrix, highlight_state=None):
    """
    Creates a dictionary representation of the Markov chain for use with graphviz.
    Optionally highlights a specific state.
    """
    dot_graph = "digraph MarkovChain {\n"
    dot_graph += "  rankdir=LR;\n"
    dot_graph += "  bgcolor=\"transparent\";\n"
    dot_graph += "  node [shape=circle, style=filled, fontname=Helvetica];\n"
    dot_graph += "  edge [fontname=Helvetica, fontsize=10];\n"

    for state in transition_matrix.index:
        if highlight_state and state == highlight_state:
            dot_graph += f'  "{state}" [label="{state}", fillcolor="#ff7f0e", fontcolor="white"];\n'
        else:
            dot_graph += f'  "{state}" [label="{state}", fillcolor="lightblue"];\n'

    for from_state in transition_matrix.index:
        for to_state in transition_matrix.columns:
            probability = transition_matrix.loc[from_state, to_state]
            if probability > 0.01:  # Only show significant transitions
                edge_color = "red" if highlight_state and (from_state == highlight_state or to_state == highlight_state) else "black"
                dot_graph += f'  "{from_state}" -> "{to_state}" [label="{probability:.2f}", penwidth={probability*3:.1f}, color="{edge_color}"];\n'

    dot_graph += "}"
    return dot_graph

def create_hmm_diagram(transition_probs, emission_probs):
    """
    Creates a graphviz diagram for the HMM.
    """
    dot_graph = "digraph HMM {\n"
    dot_graph += "  rankdir=LR;\n"
    dot_graph += "  bgcolor=\"transparent\";\n"
    dot_graph += "  node [style=filled, fontname=Helvetica];\n"
    dot_graph += "  edge [fontname=Helvetica, fontsize=10];\n"
    
    # Create subgraph for hidden states
    dot_graph += "  subgraph cluster_hidden {\n"
    dot_graph += "    label=\"Hidden States\";\n"
    dot_graph += "    style=filled;\n"
    dot_graph += "    color=lightgrey;\n"
    dot_graph += "    node [shape=circle, fillcolor=\"#3498db\", fontcolor=white];\n"
    
    # Add hidden states
    for state in transition_probs.index:
        dot_graph += f'    "{state}" [label="{state}"];\n'
    
    # Add transitions between hidden states
    for from_state in transition_probs.index:
        for to_state in transition_probs.columns:
            probability = transition_probs.loc[from_state, to_state]
            if probability > 0.01:
                dot_graph += f'    "{from_state}" -> "{to_state}" [label="{probability:.2f}", penwidth={probability*2:.1f}];\n'
    
    dot_graph += "  }\n"
    
    # Create subgraph for observed events
    dot_graph += "  subgraph cluster_observed {\n"
    dot_graph += "    label=\"Observed Events\";\n"
    dot_graph += "    style=filled;\n"
    dot_graph += "    color=lightyellow;\n"
    dot_graph += "    node [shape=box, fillcolor=\"#2ecc71\", fontcolor=white];\n"
    
    # Add observed events
    for event in emission_probs.columns:
        dot_graph += f'    "obs_{event}" [label="{event}"];\n'
    
    dot_graph += "  }\n"
    
    # Add emission edges
    for state in emission_probs.index:
        for event in emission_probs.columns:
            probability = emission_probs.loc[state, event]
            if probability > 0.01:
                dot_graph += f'  "{state}" -> "obs_{event}" [label="{probability:.2f}", style=dashed, penwidth={probability*2:.1f}, color="#e74c3c"];\n'
    
    dot_graph += "}"
    return dot_graph

def plot_heatmap(data, title, cmap="YlGnBu", annot_fmt=".2f"):
    """
    Creates a heatmap visualization for matrices.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.isinf(data) if isinstance(data, pd.DataFrame) else None
    
    # Create a custom colormap for non-infinite values
    cmap_obj = plt.cm.get_cmap(cmap)
    
    # Plot the heatmap
    sns.heatmap(data, annot=True, fmt=annot_fmt, cmap=cmap_obj, mask=mask, 
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Value'})
    
    # Replace inf values with text
    if mask is not None:
        for i, j in zip(*np.where(mask)):
            ax.text(j + 0.5, i + 0.5, "∞", ha='center', va='center',
                    color='red', fontweight='bold', fontsize=12)
    
    plt.title(title, fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    return fig
def plot_network_graph(matrix, title):
    """
    Creates a network graph visualization for transition matrices.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    G = nx.DiGraph()
    
    # Add edges with weights
    for i in matrix.index:
        for j in matrix.columns:
            if matrix.loc[i, j] > 0.01:  # Only add significant transitions
                G.add_edge(i, j, weight=matrix.loc[i, j])
    
    # Calculate node sizes based on steady state probabilities if available
    try:
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state = np.abs(eigenvectors[:, closest_to_one_idx])
        steady_state = steady_state / steady_state.sum()
        node_sizes = {state: 300 + 1000 * prob for state, prob in zip(matrix.index, steady_state)}
    except:
        node_sizes = {state: 500 for state in matrix.index}
    
    # Set positions using spring layout
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="#1f77b4", alpha=0.9, 
                          node_size=[node_sizes[node] for node in G.nodes()])
    
    # Draw edges with varying thickness
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.7, 
                          edge_color="#ff7f0e", arrowsize=20, arrowstyle='-|>')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
    
    # Draw edge labels (probabilities)
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    return fig

def plot_queuing_metrics(queuing_data):
    """
    Creates visualizations for queuing theory metrics.
    """
    if not queuing_data['stable']:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "System is unstable (utilization ≥ 1)", 
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return fig
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Queue length distribution
    probs = queuing_data['queue_length_probabilities']
    keys = list(probs.keys())
    values = list(probs.values())
    
    axs[0, 0].bar(keys, values, color=primary_color)
    axs[0, 0].set_title('Queue Length Distribution')
    axs[0, 0].set_xlabel('Number of Customers')
    axs[0, 0].set_ylabel('Probability')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: System metrics
    metrics = ['probability_of_zero_customers', 'probability_of_waiting', 'utilization']
    metric_names = ['P(0)', 'P(wait)', 'Utilization']
    metric_values = [queuing_data[m] for m in metrics]
    
    axs[0, 1].bar(metric_names, metric_values, color=secondary_color)
    axs[0, 1].set_title('System Metrics')
    axs[0, 1].set_ylabel('Probability')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_ylim([0, 1])
    
    # Plot 3: Average number of customers
    l_metrics = ['average_customers_in_system', 'average_customers_in_queue']
    l_names = ['In System (L)', 'In Queue (Lq)']
    l_values = [queuing_data[m] for m in l_metrics]
    
    axs[1, 0].bar(l_names, l_values, color=accent_color)
    axs[1, 0].set_title('Average Number of Customers')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average waiting times
    w_metrics = ['average_time_in_system', 'average_time_in_queue']
    w_names = ['In System (W)', 'In Queue (Wq)']
    w_values = [queuing_data[m] for m in w_metrics]
    
    axs[1, 1].bar(w_names, w_values, color=dark_color)
    axs[1, 1].set_title('Average Time')
    axs[1, 1].set_ylabel('Time Units')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def animate_viterbi(observations, hidden_states, viterbi_values, backpointers, best_path_indices):
    """
    Creates an animation showing the Viterbi algorithm's steps.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a colormap for visualization
    cmap = plt.cm.viridis
    
    def update(frame):
        ax.clear()
        
        # Get the current step's data
        current_viterbi = viterbi_values[frame]
        
        # Plot the trellis diagram
        for i, state in enumerate(hidden_states):
            # Plot nodes
            for t in range(frame + 1):
                # Color based on whether this is part of the best path
                if frame == len(viterbi_values) - 1 and i == best_path_indices[t]:
                    color = 'red'
                    size = 100
                else:
                    color = cmap(current_viterbi[i, t] / np.max(current_viterbi) if np.max(current_viterbi) > 0 else 0)
                    size = 80
                
                ax.scatter(t, i, s=size, color=color, zorder=3)
                
                # Add value text
                if current_viterbi[i, t] > 0:
                    value_str = f"{current_viterbi[i, t]:.4f}"
                else:
                    value_str = "0"
                ax.text(t, i+0.2, value_str, fontsize=8, ha='center')
            
            # Plot edges from previous step
            if frame > 0:
                for i_prev in range(len(hidden_states)):
                    # Show all possible transitions
                    if current_viterbi[i_prev, frame-1] > 0:
                        # Make the chosen backpointer thicker
                        if i == backpointers[frame][i_prev]:
                            ax.plot([frame-1, frame], [i_prev, i], 'r-', linewidth=2, alpha=0.7, zorder=2)
                        else:
                            ax.plot([frame-1, frame], [i_prev, i], 'k-', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Add labels
        ax.set_title(f'Viterbi Algorithm Step {frame+1}/{len(viterbi_values)}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Hidden States')
        ax.set_yticks(range(len(hidden_states)))
        ax.set_yticklabels(hidden_states)
        ax.set_xticks(range(frame + 1))
        ax.set_xticklabels([f"{t+1}\n{observations[t]}" for t in range(frame + 1)])
        ax.grid(True, alpha=0.3)
    
    ani = FuncAnimation(fig, update, frames=len(viterbi_values), interval=1000, repeat=True)
    plt.close()  # Prevent the empty figure from displaying
    
    return ani

def plot_forward_algorithm(observations, hidden_states, alphas):
    """
    Creates a visualization of the forward algorithm's alpha values over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot alpha values for each state over time
    time_steps = range(1, len(observations) + 1)
    
    for i, state in enumerate(hidden_states):
        alpha_values = [step[i, t] for t, step in enumerate(alphas)]
        ax.plot(time_steps, alpha_values, 'o-', label=f'State: {state}')
    
    # Add labels and legend
    ax.set_title('Forward Algorithm: Alpha Values Over Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Alpha Value (Forward Probability)')
    ax.set_xticks(time_steps)
    ax.set_xticklabels([f"{t}\n{observations[t-1]}" for t in time_steps])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def simulate_markov_chain(transition_matrix, steps=50, start_state=None):
    """
    Simulates a Markov chain for the given number of steps.
    """
    states = transition_matrix.index.tolist()
    
    if start_state is None:
        # Start with a random state
        start_state = np.random.choice(states)
    
    if start_state not in states:
        return None
    
    # Initialize the simulation
    current_state = start_state
    state_sequence = [current_state]
    
    # Simulate the chain
    for _ in range(steps - 1):
        # Get transition probabilities for current state
        transition_probs = transition_matrix.loc[current_state].values
        
        # Choose next state based on transition probabilities
        next_state = np.random.choice(states, p=transition_probs)
        state_sequence.append(next_state)
        current_state = next_state
    
    return state_sequence

def simulate_hmm(transition_probs, emission_probs, steps=50, start_state=None):
    """
    Simulates a Hidden Markov Model for the given number of steps.
    """
    hidden_states = transition_probs.index.tolist()
    observed_events = emission_probs.columns.tolist()
    
    if start_state is None:
        # Start with a random state
        start_state = np.random.choice(hidden_states)
    
    if start_state not in hidden_states:
        return None
    
    # Initialize the simulation
    current_state = start_state
    hidden_sequence = [current_state]
    observed_sequence = []
    
    # Simulate the HMM
    for _ in range(steps):
        # Generate observation based on current hidden state
        observation_probs = emission_probs.loc[current_state].values
        observation = np.random.choice(observed_events, p=observation_probs)
        observed_sequence.append(observation)
        
        if _ < steps - 1:  # Don't transition after the last step
            # Choose next hidden state based on transition probabilities
            transition_probs_current = transition_probs.loc[current_state].values
            next_state = np.random.choice(hidden_states, p=transition_probs_current)
            hidden_sequence.append(next_state)
            current_state = next_state
    
    return hidden_sequence, observed_sequence

def simulate_queue(arrival_rate, service_rate, servers=1, duration=1000):
    """
    Simulates an M/M/s queuing system.
    """
    # Initialize simulation
    time = 0
    queue = []
    server_busy_until = [0] * servers
    
    events = []  # (time, event_type, customer_id)
    customer_id = 0
    
    # Generate first arrival
    next_arrival = np.random.exponential(1/arrival_rate)
    events.append((next_arrival, "arrival", customer_id))
    
    # Simulation statistics
    customer_stats = {}  # {id: (arrival_time, service_start, departure_time)}
    queue_length_over_time = [(0, 0)]  # (time, queue_length)
    customers_in_system_over_time = [(0, 0)]  # (time, customers_in_system)
    
    # Run simulation
    while time < duration:
        # Get next event
        events.sort()
        event_time, event_type, cust_id = events.pop(0)
        
        # Update time
        time = event_time
        
        if time > duration:
            break
        
        # Handle event
        if event_type == "arrival":
            # Schedule next arrival
            customer_id += 1
            next_arrival = time + np.random.exponential(1/arrival_rate)
            events.append((next_arrival, "arrival", customer_id))
            
            # Record arrival
            customer_stats[cust_id] = (time, None, None)
            
            # Find available server
            available_server = None
            for i in range(servers):
                if server_busy_until[i] <= time:
                    available_server = i
                    break
            
            if available_server is not None:
                # Server available, start service
                service_time = np.random.exponential(1/service_rate)
                server_busy_until[available_server] = time + service_time
                events.append((time + service_time, "departure", cust_id))
                
                # Update customer stats
                arrival_time, _, _ = customer_stats[cust_id]
                customer_stats[cust_id] = (arrival_time, time, time + service_time)
            else:
                # All servers busy, join queue
                queue.append(cust_id)
        
        elif event_type == "departure":
            # Customer leaves
            arrival_time, service_start, _ = customer_stats[cust_id]
            customer_stats[cust_id] = (arrival_time, service_start, time)
            
            # If queue not empty, start service for next customer
            if queue:
                next_cust = queue.pop(0)
                
                # Find server that just became available
                for i in range(servers):
                    if server_busy_until[i] <= time:
                        available_server = i
                        break
                
                service_time = np.random.exponential(1/service_rate)
                server_busy_until[available_server] = time + service_time
                events.append((time + service_time, "departure", next_cust))
                
                # Update customer stats
                arrival_time, _, _ = customer_stats[next_cust]
                customer_stats[next_cust] = (arrival_time, time, time + service_time)
        
        # Update statistics
        queue_length_over_time.append((time, len(queue)))
        customers_in_system = len(queue) + sum(1 for s in server_busy_until if s > time)
        customers_in_system_over_time.append((time, customers_in_system))
    
    # Calculate statistics
    waiting_times = []
    system_times = []
    
    for cust_id, (arrival, service_start, departure) in customer_stats.items():
        if service_start is not None and departure is not None:
            waiting_times.append(service_start - arrival)
            system_times.append(departure - arrival)
    
    stats = {
        'average_waiting_time': np.mean(waiting_times) if waiting_times else 0,
        'average_system_time': np.mean(system_times) if system_times else 0,
        'queue_length_over_time': queue_length_over_time,
        'customers_in_system_over_time': customers_in_system_over_time,
        'utilization': sum(server_busy_until) / (duration * servers),
        'customer_stats': customer_stats
    }
    
    return stats

# Main Streamlit App
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Analysis Type", 
                              ["Home", "Markov Chain Analysis", "Hidden Markov Model Analysis", 
                               "Queuing Theory Analysis", "Simulation"])

    if app_mode == "Home":
        st.title("Stochastic Process Analyzer 📊")
        
        st.markdown("""
        <div class="dashboard-card">
            <h2>Welcome to the Stochastic Process Analyzer!</h2>
            <p>This tool helps you analyze various stochastic processes:</p>
            <ul>
                <li><strong>Markov Chains</strong>: Analyze state transitions and steady-state probabilities</li>
                <li><strong>Hidden Markov Models</strong>: Understand hidden states from observed events</li>
                <li><strong>Queuing Theory</strong>: Analyze M/M/s queuing systems</li>
                <li><strong>Simulation</strong>: Run simulations of stochastic processes</li>
            </ul>
            <p>Select an analysis type from the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Markov Chains</h3>
                <p>Analyze discrete state transitions with the Markov property. Calculate steady-state probabilities, mean recurrence times, and more.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Hidden Markov Models</h3>
                <p>Analyze systems with hidden states that generate observable events. Apply forward and Viterbi algorithms.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Queuing Theory</h3>
                <p>Analyze M/M/s queuing systems. Calculate utilization, waiting times, queue lengths, and more.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div class="info-text">
            <h3>Getting Started</h3>
            <p>To analyze your data, you'll need to upload a CSV file with the appropriate columns for each analysis type. Example templates are provided on each analysis page.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example data formats
        with st.expander("Data Format Examples"):
            st.markdown("""
            ### Markov Chain Data Format
            Your CSV should contain pairs of current and next states:
            
            | current_state | next_state |
            |---------------|------------|
            | A             | B          |
            | B             | C          |
            | C             | A          |
            
            ### Hidden Markov Model Data Format
            Your CSV should contain pairs of hidden states and observed events:
            
            | hidden_state | observed_event |
            |--------------|----------------|
            | Sunny        | Dry            |
            | Sunny        | Dry            |
            | Rainy        | Wet            |
            
            ### Queuing Theory Data Format
            Your CSV should contain arrival times and service times:
            
            | arrival_time_minutes | service_time_minutes |
            |----------------------|----------------------|
            | 0.5                  | 2.3                  |
            | 1.2                  | 1.7                  |
            | 3.6                  | 3.1                  |
            """)

    elif app_mode == "Markov Chain Analysis":
        st.title("Markov Chain Analysis 🔄")
        
        st.markdown("""
        <div class="info-text">
            <p>A Markov chain is a stochastic model describing a sequence of possible events where the probability of each event depends only on the state of the previous event.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data upload section
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        st.header("Upload Data")
        
        # Options for data input
        data_input = st.radio("Choose data input method:", ["Upload CSV", "Use Example Data", "Manual Entry"])
        
        if data_input == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    df = None
            else:
                df = None
        
        elif data_input == "Use Example Data":
            st.info("Using example Markov Chain data")
            # Simple weather example
            example_data = {
                'current_state': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny'],
                'next_state': ['Sunny', 'Rainy', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Sunny']
            }
            df = pd.DataFrame(example_data)
            st.dataframe(df)
        
        elif data_input == "Manual Entry":
            st.subheader("Enter state transitions")
            
            # Dynamic inputs for states
            num_transitions = st.number_input("Number of transitions:", min_value=1, value=5)
            
            data = {'current_state': [], 'next_state': []}
            for i in range(num_transitions):
                cols = st.columns(2)
                with cols[0]:
                    current = st.text_input(f"Current State {i+1}", value=f"State {i%3 + 1}")
                with cols[1]:
                    next_state = st.text_input(f"Next State {i+1}", value=f"State {(i+1)%3 + 1}")
                
                data['current_state'].append(current)
                data['next_state'].append(next_state)
            
            df = pd.DataFrame(data)
            st.dataframe(df)
        
        # Column mapping
        if df is not None and not df.empty:
            st.subheader("Column Mapping")
            current_state_col = st.selectbox("Select current state column:", df.columns, index=df.columns.get_loc('current_state') if 'current_state' in df.columns else 0)
            next_state_col = st.selectbox("Select next state column:", df.columns, index=df.columns.get_loc('next_state') if 'next_state' in df.columns else min(1, len(df.columns) - 1))
            
            # Analyze button
            if st.button("Analyze Markov Chain"):
                with st.spinner("Analyzing..."):
                    results = markov_chain_analysis(df, state_col=current_state_col, next_state_col=next_state_col)
                
                if results:
                    # Display results in organized tabs
                    tabs = st.tabs(["Transition Matrix", "State Diagram", "Steady State", "Passage Times", "Properties"])
                    
                    with tabs[0]:
                        st.subheader("Transition Matrix")
                        st.write("The probability of transitioning from one state to another:")
                        st.dataframe(results['transition_matrix'])
                        
                        # Heatmap visualization
                        fig = plot_heatmap(results['transition_matrix'], "Transition Probability Heatmap")
                        st.pyplot(fig)
                    
                    with tabs[1]:
                        st.subheader("State Transition Diagram")
                        # Let user select a state to highlight
                        highlight_state = st.selectbox("Highlight state:", ["None"] + results['states'])
                        highlight = None if highlight_state == "None" else highlight_state
                        
                        # Create and display diagram
                        dot_graph = create_diagram(results['transition_matrix'], highlight)
                        st.graphviz_chart(dot_graph)
                        
                        # Network graph
                        st.subheader("Network Graph")
                        fig = plot_network_graph(results['transition_matrix'], "Markov Chain Network")
                        st.pyplot(fig)
                    
                    with tabs[2]:
                        st.subheader("Steady State Probabilities")
                        st.write("The long-run probability of being in each state:")
                        st.dataframe(results['steady_state_probabilities'])
                        
                        # Bar chart for steady state probabilities
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(results['steady_state_probabilities'].index, 
                              results['steady_state_probabilities']['Probability'], 
                              color=primary_color)
                        ax.set_title("Steady State Probabilities")
                        ax.set_ylabel("Probability")
                        ax.set_xlabel("State")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with tabs[3]:
                        st.subheader("Mean First Passage Times")
                        st.write("The expected number of steps to reach state j from state i:")
                        st.dataframe(results['average_passage_times'])
                        
                        # Heatmap for passage times
                        fig = plot_heatmap(results['average_passage_times'], "Mean First Passage Times")
                        st.pyplot(fig)
                        
                        st.subheader("Mean Recurrence Times")
                        st.write("The expected number of steps to return to the same state:")
                        recurrence_df = pd.DataFrame({'Recurrence Time': results['average_recurrence_times']})
                        st.dataframe(recurrence_df)
                    
                    with tabs[4]:
                        st.subheader("State Properties")
                        
                        # State frequencies
                        st.write("State Frequencies (Empirical):")
                        freq_df = pd.DataFrame({'Frequency': pd.Series(results['state_frequencies'])})
                        st.dataframe(freq_df)
                        
                        # State periods
                        st.write("State Periods:")
                        period_df = pd.DataFrame({'Period': pd.Series(results['periods'])})
                        st.dataframe(period_df)
                        
                        # Absorbing states
                        if results['absorbing_states']:
                            st.write(f"Absorbing States: {', '.join(results['absorbing_states'])}")
                            
                            if results['absorption_steps']:
                                st.write("Expected Steps to Absorption:")
                                abs_df = pd.DataFrame({'Steps to Absorption': pd.Series(results['absorption_steps'])})
                                st.dataframe(abs_df)
                        else:
                            st.write("No absorbing states found.")
                            
                        # Check if the chain is ergodic
                        is_irreducible = all(not np.isinf(results['average_passage_times'].loc[i, j]) 
                                            for i in results['states'] for j in results['states'])
                        is_aperiodic = all(results['periods'][state] == 1 for state in results['states'])
                        
                        st.write(f"The chain is {'irreducible' if is_irreducible else 'reducible'}.")
                        st.write(f"The chain is {'aperiodic' if is_aperiodic else 'periodic'}.")
                        st.write(f"The chain is {'ergodic' if (is_irreducible and is_aperiodic) else 'not ergodic'}.")
                        
                        # Classification of states
                        st.subheader("State Classification")
                        state_classify = {}
                        for state in results['states']:
                            if state in results['absorbing_states']:
                                state_classify[state] = "Absorbing"
                            elif np.isinf(results['average_recurrence_times'][state]):
                                state_classify[state] = "Transient"
                            else:
                                if results['periods'][state] == 1:
                                    state_classify[state] = "Positive Recurrent, Aperiodic"
                                else:
                                    state_classify[state] = f"Positive Recurrent, Periodic (p={results['periods'][state]})"
                        
                        state_class_df = pd.DataFrame({'Classification': pd.Series(state_classify)})
                        st.dataframe(state_class_df)
                else:
                    st.error("Could not analyze the Markov Chain. Please check your data.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif app_mode == "Hidden Markov Model Analysis":
        st.title("Hidden Markov Model Analysis 🔍")
        
        st.markdown("""
        <div class="info-text">
            <p>A Hidden Markov Model (HMM) is a statistical model where the system being modeled is assumed to be a Markov process with hidden states that generate observable events.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data upload section
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        st.header("Upload Data")
        
        # Options for data input
        data_input = st.radio("Choose data input method:", ["Upload CSV", "Use Example Data", "Manual Entry"], key="hmm_input")
        
        if data_input == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="hmm_file")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    df = None
            else:
                df = None
        
        elif data_input == "Use Example Data":
            st.info("Using example Hidden Markov Model data")
            # Weather-activity example
            example_data = {
                'hidden_state': ['Sunny', 'Sunny','Rainy', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny'],
                'observed_event': ['Dry', 'Dry', 'Wet', 'Wet', 'Dry', 'Wet', 'Dry', 'Dry', 'Wet', 'Dry']
            }
            df = pd.DataFrame(example_data)
            st.dataframe(df)
        
        elif data_input == "Manual Entry":
            st.subheader("Enter hidden states and observed events")
            
            # Dynamic inputs for HMM
            num_observations = st.number_input("Number of observations:", min_value=1, value=5, key="hmm_num")
            
            data = {'hidden_state': [], 'observed_event': []}
            for i in range(num_observations):
                cols = st.columns(2)
                with cols[0]:
                    hidden = st.text_input(f"Hidden State {i+1}", value="Sunny" if i%2 == 0 else "Rainy", key=f"hidden_{i}")
                with cols[1]:
                    observed = st.text_input(f"Observed Event {i+1}", value="Dry" if i%2 == 0 else "Wet", key=f"obs_{i}")
                
                data['hidden_state'].append(hidden)
                data['observed_event'].append(observed)
            
            df = pd.DataFrame(data)
            st.dataframe(df)
        
        # Column mapping
        if df is not None and not df.empty:
            st.subheader("Column Mapping")
            hidden_state_col = st.selectbox("Select hidden state column:", df.columns, 
                                          index=df.columns.get_loc('hidden_state') if 'hidden_state' in df.columns else 0)
            observed_event_col = st.selectbox("Select observed event column:", df.columns,
                                            index=df.columns.get_loc('observed_event') if 'observed_event' in df.columns else min(1, len(df.columns)-1))
            
            # Analyze button
            if st.button("Analyze Hidden Markov Model"):
                with st.spinner("Analyzing..."):
                    results = hidden_markov_model_analysis(df, hidden_state_col=hidden_state_col, observed_event_col=observed_event_col)
                
                if results:
                    # Display results in organized tabs
                    tabs = st.tabs(["Model Parameters", "State Diagram", "Forward Algorithm", "Viterbi Path", "Joint Probabilities"])
                    
                    with tabs[0]:
                        st.subheader("Transition Probabilities (Hidden States)")
                        st.write("Probability of transitioning between hidden states:")
                        st.dataframe(results['transition_probabilities'])
                        
                        st.subheader("Emission Probabilities")
                        st.write("Probability of observing events given hidden states:")
                        st.dataframe(results['emission_probabilities'])
                        
                        st.subheader("Steady State Probabilities")
                        st.write("Long-run probabilities of hidden states:")
                        st.dataframe(results['steady_state_probabilities'])
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = plot_heatmap(results['transition_probabilities'], "Transition Probabilities Heatmap")
                            st.pyplot(fig)
                        with col2:
                            fig = plot_heatmap(results['emission_probabilities'], "Emission Probabilities Heatmap")
                            st.pyplot(fig)
                    
                    with tabs[1]:
                        st.subheader("HMM State Diagram")
                        dot_graph = create_hmm_diagram(results['transition_probabilities'], results['emission_probabilities'])
                        st.graphviz_chart(dot_graph)
                    
                    with tabs[2]:
                        st.subheader("Forward Algorithm")
                        st.write("The forward algorithm computes the probability of the observed sequence given the model.")
                        
                        # Let user input an observation sequence
                        obs_sequence = st.text_input("Enter observation sequence (comma separated):", 
                                                    value=",".join(df[observed_event_col].astype(str).tolist()[:5]))
                        obs_sequence = [x.strip() for x in obs_sequence.split(",") if x.strip()]
                        
                        if obs_sequence:
                            # Run forward algorithm
                            prob, alphas = results['forward_algorithm'](obs_sequence, 
                                                                       results['initial_probs'],
                                                                       results['transition_probabilities'],
                                                                       results['emission_probabilities'])
                            
                            st.write(f"Probability of observation sequence: {prob:.6f}")
                            
                            # Plot alpha values
                            fig = plot_forward_algorithm(obs_sequence, results['hidden_states'], alphas)
                            st.pyplot(fig)
                    
                    with tabs[3]:
                        st.subheader("Viterbi Algorithm")
                        st.write("The Viterbi algorithm finds the most likely sequence of hidden states given observations.")
                        
                        # Use same observation sequence as forward algorithm
                        if 'obs_sequence' in locals() and obs_sequence:
                            best_path, viterbi_values, backpointers, path_indices = results['viterbi_algorithm'](
                                obs_sequence,
                                results['initial_probs'],
                                results['transition_probabilities'],
                                results['emission_probabilities']
                            )
                            
                            st.write(f"Most likely hidden state sequence: {', '.join(best_path)}")
                            
                            # Create animation
                            st.write("Viterbi Algorithm Visualization:")
                            ani = animate_viterbi(obs_sequence, results['hidden_states'], viterbi_values, backpointers, path_indices)
                            
                            # Display animation
                            with st.spinner("Rendering animation..."):
                                # Save animation as HTML
                                html = ani.to_jshtml()
                                st.components.v1.html(html, height=600)
                    
                    with tabs[4]:
                        st.subheader("Joint Probabilities")
                        st.write("Joint probability distribution of hidden states and observed events:")
                        st.dataframe(results['joint_probability'])
                        
                        st.subheader("Observation Probabilities")
                        st.write("Marginal probabilities of observed events:")
                        obs_probs = pd.DataFrame.from_dict(results['observation_probabilities'], 
                                                         orient='index', columns=['Probability'])
                        st.dataframe(obs_probs)
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        obs_probs.plot(kind='bar', ax=ax, color=secondary_color)
                        ax.set_title("Observation Probabilities")
                        ax.set_ylabel("Probability")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                else:
                    st.error("Could not analyze the Hidden Markov Model. Please check your data.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif app_mode == "Queuing Theory Analysis":
        st.title("Queuing Theory Analysis 🏦")
        
        st.markdown("""
        <div class="info-text">
            <p>Queuing theory is the mathematical study of waiting lines or queues. This analyzer focuses on M/M/s queues (Markovian arrival, Markovian service, s servers).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data upload section
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        st.header("Upload Data")
        
        # Options for data input
        data_input = st.radio("Choose data input method:", ["Upload CSV", "Use Example Data", "Manual Entry"], key="queue_input")
        
        if data_input == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="queue_file")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    df = None
            else:
                df = None
        
        elif data_input == "Use Example Data":
            st.info("Using example queuing data")
            # Generate example arrival and service times
            np.random.seed(42)
            arrival_times = np.cumsum(np.random.exponential(5, 50))
            service_times = np.random.exponential(4, 50)
            example_data = {
                'arrival_time_minutes': arrival_times,
                'service_time_minutes': service_times
            }
            df = pd.DataFrame(example_data)
            st.dataframe(df.head(10))
        
        elif data_input == "Manual Entry":
            st.subheader("Enter arrival and service times")
            
            # Dynamic inputs for queue data
            num_customers = st.number_input("Number of customers:", min_value=1, value=5, key="queue_num")
            
            data = {'arrival_time_minutes': [], 'service_time_minutes': []}
            for i in range(num_customers):
                cols = st.columns(2)
                with cols[0]:
                    arrival = st.number_input(f"Arrival time (min) {i+1}", min_value=0.0, value=float(i*5 + np.random.randint(1,4)), key=f"arrival_{i}")
                with cols[1]:
                    service = st.number_input(f"Service time (min) {i+1}", min_value=0.1, value=float(np.random.randint(2,6)), key=f"service_{i}")
                
                data['arrival_time_minutes'].append(arrival)
                data['service_time_minutes'].append(service)
            
            df = pd.DataFrame(data)
            st.dataframe(df)
        
        # Column mapping and parameters
        if df is not None and not df.empty:
            st.subheader("Column Mapping")
            arrival_col = st.selectbox("Select arrival time column:", df.columns,
                                     index=df.columns.get_loc('arrival_time_minutes') if 'arrival_time_minutes' in df.columns else 0)
            service_col = st.selectbox("Select service time column:", df.columns,
                                      index=df.columns.get_loc('service_time_minutes') if 'service_time_minutes' in df.columns else 1)
            
            st.subheader("System Parameters")
            servers = st.number_input("Number of servers (s):", min_value=1, value=1)
            
            # Analyze button
            if st.button("Analyze Queue"):
                with st.spinner("Analyzing..."):
                    results = queuing_theory_analysis(df, arrival_time_col=arrival_col, service_time_col=service_col, servers=servers)
                
                if results:
                    # Display results in organized tabs
                    tabs = st.tabs(["Metrics", "Visualizations", "Simulation", "System Properties"])
                    
                    with tabs[0]:
                        st.subheader("Queue Metrics")
                        
                        # Key metrics in cards
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Arrival Rate (λ)</div>
                                <div class="metric-value">{results['arrival_rate']:.2f}</div>
                                <div class="metric-label">customers per time unit</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Service Rate (μ)</div>
                                <div class="metric-value">{results['service_rate']:.2f}</div>
                                <div class="metric-label">customers per time unit</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Utilization (ρ)</div>
                                <div class="metric-value">{results['utilization']:.2f}</div>
                                <div class="metric-label">{'⚠ High (>0.7)' if results['utilization'] > 0.7 else '✓ Good'}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Customers in System (L)</div>
                                <div class="metric-value">{results['average_customers_in_system']:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Time in System (W)</div>
                                <div class="metric-value">{results['average_time_in_system']:.2f}</div>
                                <div class="metric-label">time units</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Probability of Waiting</div>
                                <div class="metric-value">{results['probability_of_waiting']:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed metrics
                        st.subheader("Detailed Metrics")
                        st.write(f"Average customers in queue (Lq): {results['average_customers_in_queue']:.2f}")
                        st.write(f"Average time in queue (Wq): {results['average_time_in_queue']:.2f}")
                        st.write(f"Probability system is empty (P0): {results['probability_of_zero_customers']:.2f}")
                        
                        # System stability
                        if not results['stable']:
                            st.error("⚠ Warning: System is unstable (utilization ≥ 1). The queue will grow infinitely long.")
                        elif results['utilization'] > 0.7:
                            st.warning("⚠ Warning: High utilization (>0.7). Consider adding more servers.")
                    
                    with tabs[1]:
                        st.subheader("Visualizations")
                        fig = plot_queuing_metrics(results)
                        st.pyplot(fig)
                        
                        # Additional distribution plots
                        st.subheader("Input Distributions")
                        col1, col2 = st.columns(2)
                        with col1:
                            fig, ax = plt.subplots()
                            sns.histplot(df[arrival_col].diff().dropna(), kde=True, ax=ax)
                            ax.set_title("Inter-arrival Time Distribution")
                            ax.set_xlabel("Time between arrivals")
                            st.pyplot(fig)
                            
                            st.write(f"Coefficient of variation: {results['cv_inter_arrival']:.2f}")
                        
                        with col2:
                            fig, ax = plt.subplots()
                            sns.histplot(df[service_col], kde=True, ax=ax)
                            ax.set_title("Service Time Distribution")
                            ax.set_xlabel("Service time")
                            st.pyplot(fig)
                            
                            st.write(f"Coefficient of variation: {results['cv_service']:.2f}")
                    
                    with tabs[2]:
                        st.subheader("Queue Simulation")
                        st.write("Run a discrete-event simulation of the queue based on the estimated parameters.")
                        
                        sim_duration = st.number_input("Simulation duration (time units):", min_value=10, value=100)
                        
                        if st.button("Run Simulation"):
                            with st.spinner("Simulating..."):
                                sim_results = simulate_queue(results['arrival_rate'], 
                                                           results['service_rate'], 
                                                           servers=servers,
                                                           duration=sim_duration)
                            
                            st.subheader("Simulation Results")
                            
                            # Compare theoretical vs simulated
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                <div class="dashboard-card">
                                    <h3>Theoretical</h3>
                                    <p>Customers in system: {:.2f}</p>
                                    <p>Time in system: {:.2f}</p>
                                    <p>Utilization: {:.2f}</p>
                                </div>
                                """.format(
                                    results['average_customers_in_system'],
                                    results['average_time_in_system'],
                                    results['utilization']
                                ), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class="dashboard-card">
                                    <h3>Simulated</h3>
                                    <p>Customers in system: {:.2f}</p>
                                    <p>Time in system: {:.2f}</p>
                                    <p>Utilization: {:.2f}</p>
                                </div>
                                """.format(
                                    np.mean([x[1] for x in sim_results['customers_in_system_over_time']]),
                                    sim_results['average_system_time'],
                                    sim_results['utilization']
                                ), unsafe_allow_html=True)
                            
                            # Plot queue length over time
                            fig, ax = plt.subplots(figsize=(10, 5))
                            times, lengths = zip(*sim_results['queue_length_over_time'])
                            ax.step(times, lengths, where='post')
                            ax.set_title("Queue Length Over Time")
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Queue Length")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    with tabs[3]:
                        st.subheader("System Properties")
                        
                        # Little's Law verification
                        st.write("### Little's Law Verification")
                        st.markdown("""
                        Little's Law states: L = λW
                        """)
                        st.write(f"Calculated L: {results['average_customers_in_system']:.2f}")
                        st.write(f"λ × W: {results['arrival_rate'] * results['average_time_in_system']:.2f}")
                        
                        # Traffic intensity interpretation
                        st.write("### Traffic Intensity Interpretation")
                        rho = results['utilization']
                        if rho < 0.5:
                            st.success(f"Low traffic intensity (ρ = {rho:.2f}). The system has ample capacity.")
                        elif rho < 0.7:
                            st.info(f"Moderate traffic intensity (ρ = {rho:.2f}). The system is stable but may experience occasional delays.")
                        elif rho < 1:
                            st.warning(f"High traffic intensity (ρ = {rho:.2f}). The system is stable but will experience significant delays.")
                        else:
                            st.error(f"Critical traffic intensity (ρ = {rho:.2f}). The system is unstable and the queue will grow without bound.")
                        
                        # Server recommendation
                        if rho >= 1 and st.button("Calculate Required Servers"):
                            min_servers = int(np.ceil(results['arrival_rate'] / results['service_rate'])) + 1
                            st.warning(f"Minimum {min_servers} servers needed for stability (ρ < 1).")
                else:
                    st.error("Could not analyze the queuing system. Please check your data.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif app_mode == "Simulation":
        st.title("Process Simulation 🎮")
        
        st.markdown("""
        <div class="info-text">
            <p>Simulate stochastic processes to understand their behavior under different parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation type selection
        sim_type = st.selectbox("Select simulation type:", 
                               ["Markov Chain", "Hidden Markov Model", "Queuing System"])
        
        if sim_type == "Markov Chain":
            st.header("Markov Chain Simulation")
            
            # Options for simulation input
            sim_input = st.radio("Choose input method:", ["Define Matrix", "Upload Matrix", "Use Example"], key="mc_sim_input")
            
            transition_matrix = None
            
            if sim_input == "Define Matrix":
                st.subheader("Define Transition Matrix")
                
                num_states = st.number_input("Number of states:", min_value=2, value=3)
                state_names = [st.text_input(f"Name for State {i+1}:", value=f"S{i+1}") for i in range(num_states)]
                
                st.write("Enter transition probabilities (rows must sum to 1):")
                
                # Create empty matrix
                data = {}
                for i, from_state in enumerate(state_names):
                    row = []
                    for j, to_state in enumerate(state_names):
                        prob = st.number_input(f"P({from_state}→{to_state})", 
                                             min_value=0.0, max_value=1.0, 
                                             value=0.5 if i == j else 0.5/(num_states-1),
                                             key=f"prob_{i}_{j}")
                        row.append(prob)
                    data[from_state] = row
                
                transition_matrix = pd.DataFrame(data, index=state_names, columns=state_names)
                
                # Check row sums
                row_sums = transition_matrix.sum(axis=1)
                if not all(np.isclose(row_sums, 1)):
                    st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                    transition_matrix = None
                else:
                    st.success("Valid transition matrix!")
                    st.dataframe(transition_matrix)
            
            elif sim_input == "Upload Matrix":
                uploaded_file = st.file_uploader("Upload transition matrix (CSV)", type="csv", key="mc_matrix_upload")
                if uploaded_file is not None:
                    try:
                        transition_matrix = pd.read_csv(uploaded_file, index_col=0)
                        # Validate
                        row_sums = transition_matrix.sum(axis=1)
                        if not all(np.isclose(row_sums, 1)):
                            st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                            transition_matrix = None
                        else:
                            st.success("Valid transition matrix!")
                            st.dataframe(transition_matrix)
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            elif sim_input == "Use Example":
                st.info("Using example transition matrix")
                example_matrix = pd.DataFrame({
                    'Sunny': [0.8, 0.2, 0.1],
                    'Rainy': [0.15, 0.6, 0.3],
                    'Cloudy': [0.05, 0.2, 0.6]
                }, index=['Sunny', 'Rainy', 'Cloudy'])
                transition_matrix = example_matrix
                st.dataframe(transition_matrix)
            
            # Run simulation if matrix is valid
            if transition_matrix is not None:
                st.subheader("Simulation Parameters")
                num_steps = st.number_input("Number of steps:", min_value=10, value=50)
                start_state = st.selectbox("Starting state:", transition_matrix.index)
                
                if st.button("Run Simulation"):
                    state_sequence = simulate_markov_chain(transition_matrix, steps=num_steps, start_state=start_state)
                    
                    st.subheader("Simulation Results")
                    
                    # Display sequence
                    st.write("State sequence:")
                    st.write(", ".join(state_sequence))
                    
                    # Plot sequence over time
                    fig, ax = plt.subplots(figsize=(10, 4))
                    state_indices = [transition_matrix.index.get_loc(state) for state in state_sequence]
                    ax.plot(range(num_steps), state_indices, 'o-')
                    ax.set_yticks(range(len(transition_matrix.index)))
                    ax.set_yticklabels(transition_matrix.index)
                    ax.set_title("Markov Chain Simulation")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("State")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Calculate empirical probabilities
                    st.subheader("Empirical State Frequencies")
                    state_counts = pd.Series(state_sequence).value_counts(normalize=True)
                    state_counts = state_counts.reindex(transition_matrix.index, fill_value=0)
                    st.bar_chart(state_counts)
                    
                    # Compare with theoretical steady state
                    try:
                        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
                        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
                        steady_state = np.abs(eigenvectors[:, closest_to_one_idx])
                        steady_state = steady_state / steady_state.sum()
                        steady_state = pd.Series(steady_state, index=transition_matrix.index)
                        
                        st.write("Comparison with theoretical steady state:")
                        compare_df = pd.DataFrame({
                            'Empirical': state_counts,
                            'Theoretical': steady_state
                        })
                        st.dataframe(compare_df)
                        
                        # Plot comparison
                        fig, ax = plt.subplots()
                        compare_df.plot(kind='bar', ax=ax)
                        ax.set_title("Empirical vs Theoretical State Frequencies")
                        ax.set_ylabel("Probability")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    except:
                        st.warning("Could not calculate steady state probabilities")
        
        elif sim_type == "Hidden Markov Model":
            st.header("Hidden Markov Model Simulation")
            
            # Options for simulation input
            sim_input = st.radio("Choose input method:", ["Define Parameters", "Upload Parameters", "Use Example"], key="hmm_sim_input")
            
            transition_probs = None
            emission_probs = None
            
            if sim_input == "Define Parameters":
                st.subheader("Define HMM Parameters")
                
                # Hidden states
                num_hidden = st.number_input("Number of hidden states:", min_value=2, value=2)
                hidden_names = [st.text_input(f"Name for Hidden State {i+1}:", value=f"HS{i+1}") for i in range(num_hidden)]
                
                # Observed events
                num_observed = st.number_input("Number of observed events:", min_value=2, value=2)
                observed_names = [st.text_input(f"Name for Observed Event {i+1}:", value=f"E{i+1}") for i in range(num_observed)]
                
                st.subheader("Transition Probabilities")
                st.write("Enter transition probabilities between hidden states (rows must sum to 1):")
                
                # Create empty transition matrix
                trans_data = {}
                for i, from_state in enumerate(hidden_names):
                    row = []
                    for j, to_state in enumerate(hidden_names):
                        prob = st.number_input(f"P({from_state}→{to_state})", 
                                             min_value=0.0, max_value=1.0, 
                                             value=0.5 if i == j else 0.5/(num_hidden-1),
                                             key=f"trans_{i}_{j}")
                        row.append(prob)
                    trans_data[from_state] = row
                
                transition_probs = pd.DataFrame(trans_data, index=hidden_names, columns=hidden_names)
                
                # Check row sums
                row_sums = transition_probs.sum(axis=1)
                if not all(np.isclose(row_sums, 1)):
                    st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                    transition_probs = None
                else:
                    st.success("Valid transition matrix!")
                    st.dataframe(transition_probs)
                
                st.subheader("Emission Probabilities")
                st.write("Enter emission probabilities (rows must sum to 1):")
                
                # Create empty emission matrix
                emit_data = {}
                for i, state in enumerate(hidden_names):
                    row = []
                    for j, event in enumerate(observed_names):
                        prob = st.number_input(f"P({event}|{state})", 
                                             min_value=0.0, max_value=1.0, 
                                             value=0.5 if i == j else 0.5/(num_observed-1),
                                             key=f"emit_{i}_{j}")
                        row.append(prob)
                    emit_data[state] = row
                
                emission_probs = pd.DataFrame(emit_data, index=hidden_names, columns=observed_names)
                
                # Check row sums
                row_sums = emission_probs.sum(axis=1)
                if not all(np.isclose(row_sums, 1)):
                    st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                    emission_probs = None
                else:
                    st.success("Valid emission matrix!")
                    st.dataframe(emission_probs)
            
            elif sim_input == "Upload Parameters":
                st.subheader("Upload Transition Matrix")
                trans_file = st.file_uploader("Upload transition matrix (CSV)", type="csv", key="hmm_trans_upload")
                if trans_file is not None:
                    try:
                        transition_probs = pd.read_csv(trans_file, index_col=0)
                        # Validate
                        row_sums = transition_probs.sum(axis=1)
                        if not all(np.isclose(row_sums, 1)):
                            st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                            transition_probs = None
                        else:
                            st.success("Valid transition matrix!")
                            st.dataframe(transition_probs)
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                st.subheader("Upload Emission Matrix")
                emit_file = st.file_uploader("Upload emission matrix (CSV)", type="csv", key="hmm_emit_upload")
                if emit_file is not None:
                    try:
                        emission_probs = pd.read_csv(emit_file, index_col=0)
                        # Validate
                        row_sums = emission_probs.sum(axis=1)
                        if not all(np.isclose(row_sums, 1)):
                            st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                            emission_probs = None
                        else:
                            st.success("Valid emission matrix!")
                            st.dataframe(emission_probs)
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            elif sim_input == "Use Example":
                st.info("Using example HMM parameters")
                
                # Weather example
                transition_probs = pd.DataFrame({
                    'Sunny': [0.8, 0.2],
                    'Rainy': [0.3, 0.7]
                }, index=['Sunny', 'Rainy'])
                
                emission_probs = pd.DataFrame({
                    'Dry': [0.9, 0.1],
                    'Wet': [0.1, 0.9]
                }, index=['Sunny', 'Rainy'])
                
                st.write("Transition Probabilities:")
                st.dataframe(transition_probs)
                st.write("Emission Probabilities:")
                st.dataframe(emission_probs)
            
            # Run simulation if both matrices are valid
            if transition_probs is not None and emission_probs is not None:
                st.subheader("Simulation Parameters")
                num_steps = st.number_input("Number of steps:", min_value=10, value=20, key="hmm_steps")
                start_state = st.selectbox("Starting state:", transition_probs.index, key="hmm_start")
                
                if st.button("Run HMM Simulation"):
                    hidden_seq, observed_seq = simulate_hmm(transition_probs, emission_probs, steps=num_steps, start_state=start_state)
                    
                    st.subheader("Simulation Results")
                    
                    # Display sequences
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Hidden state sequence:")
                        st.write(", ".join(hidden_seq))
                    with col2:
                        st.write("Observed event sequence:")
                        st.write(", ".join(observed_seq))
                    
                    # Plot sequences
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    
                    # Hidden states
                    hidden_indices = [transition_probs.index.get_loc(state) for state in hidden_seq]
                    ax1.plot(range(num_steps), hidden_indices, 'o-', color=primary_color)
                    ax1.set_yticks(range(len(transition_probs.index)))
                    ax1.set_yticklabels(transition_probs.index)
                    ax1.set_ylabel("Hidden State")
                    ax1.grid(True, alpha=0.3)
                    
                    # Observed events
                    observed_indices = [emission_probs.columns.get_loc(event) for event in observed_seq]
                    ax2.plot(range(num_steps), observed_indices, 'o-', color=secondary_color)
                    ax2.set_yticks(range(len(emission_probs.columns)))
                    ax2.set_yticklabels(emission_probs.columns)
                    ax2.set_xlabel("Step")
                    ax2.set_ylabel("Observed Event")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.suptitle("HMM Simulation")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Calculate empirical probabilities
                    st.subheader("Empirical Statistics")
                    
                    # Hidden state frequencies
                    st.write("Hidden State Frequencies:")
                    hidden_counts = pd.Series(hidden_seq).value_counts(normalize=True)
                    hidden_counts = hidden_counts.reindex(transition_probs.index, fill_value=0)
                    st.bar_chart(hidden_counts)
                    
                    # Observed event frequencies
                    st.write("Observed Event Frequencies:")
                    observed_counts = pd.Series(observed_seq).value_counts(normalize=True)
                    observed_counts = observed_counts.reindex(emission_probs.columns, fill_value=0)
                    st.bar_chart(observed_counts)
                    
                    # Compare with theoretical steady state
                    try:
                        eigenvalues, eigenvectors = np.linalg.eig(transition_probs.T)
                        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
                        steady_state = np.abs(eigenvectors[:, closest_to_one_idx])
                        steady_state = steady_state / steady_state.sum()
                        steady_state = pd.Series(steady_state, index=transition_probs.index)
                        
                        st.write("Hidden State Comparison:")
                        compare_df = pd.DataFrame({
                            'Empirical': hidden_counts,
                            'Theoretical': steady_state
                        })
                        st.dataframe(compare_df)
                    except:
                        st.warning("Could not calculate steady state probabilities")
        
        elif sim_type == "Queuing System":
            st.header("Queuing System Simulation")
            
            # Simulation parameters
            st.subheader("System Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                arrival_rate = st.number_input("Arrival rate (λ, customers/time unit):", 
                                             min_value=0.01, value=0.5)
            with col2:
                service_rate = st.number_input("Service rate (μ, customers/time unit):", 
                                             min_value=0.01, value=1.0)
            with col3:
                servers = st.number_input("Number of servers (s):", 
                                        min_value=1, value=1)
            
            st.subheader("Simulation Parameters")
            sim_duration = st.number_input("Simulation duration (time units):", 
                                         min_value=10, value=100)
            
            if st.button("Run Queuing Simulation"):
                with st.spinner("Simulating..."):
                    sim_results = simulate_queue(arrival_rate, service_rate, 
                                               servers=servers, duration=sim_duration)
                
                st.subheader("Simulation Results")
                
                # Key metrics
                st.write(f"Average waiting time: {sim_results['average_waiting_time']:.2f}")
                st.write(f"Average system time: {sim_results['average_system_time']:.2f}")
                st.write(f"Server utilization: {sim_results['utilization']:.2f}")
                
                # Plot queue length over time
                fig, ax = plt.subplots(figsize=(10, 5))
                times, lengths = zip(*sim_results['queue_length_over_time'])
                ax.step(times, lengths, where='post')
                ax.set_title("Queue Length Over Time")
                ax.set_xlabel("Time")
                ax.set_ylabel("Queue Length")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Plot customers in system over time
                fig, ax = plt.subplots(figsize=(10, 5))
                times, counts = zip(*sim_results['customers_in_system_over_time'])
                ax.step(times, counts, where='post')
                ax.set_title("Customers in System Over Time")
                ax.set_xlabel("Time")
                ax.set_ylabel("Customers")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Compare with theoretical results if possible
                try:
                    theoretical = queuing_theory_analysis(
                        pd.DataFrame({
                            'arrival_time_minutes': np.cumsum(np.random.exponential(1/arrival_rate, 100)),
                            'service_time_minutes': np.random.exponential(1/service_rate, 100)
                        }),
                        servers=servers
                    )
                    
                    if theoretical and theoretical['stable']:
                        st.subheader("Comparison with Theoretical Results")
                        
                        # Create comparison table
                        compare_data = {
                            'Metric': ['Avg. Customers in System', 'Avg. Time in System', 'Utilization'],
                            'Theoretical': [
                                theoretical['average_customers_in_system'],
                                theoretical['average_time_in_system'],
                                theoretical['utilization']
                            ],
                            'Simulated': [
                                np.mean([x[1] for x in sim_results['customers_in_system_over_time']]),
                                sim_results['average_system_time'],
                                sim_results['utilization']
                            ]
                        }
                        
                        st.dataframe(pd.DataFrame(compare_data))
                except:
                    pass

# Run the app
if __name__ == "__main__":
    main()