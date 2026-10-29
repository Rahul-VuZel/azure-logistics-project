# model.py
# This file contains ALL of our AI model definitions.
# Every model follows the same simple interface:
#   .fit(X_train, y_train)     - train the model on training data
#   .predict(X)                - make predictions on new data
#   .evaluate(X_test, y_test)  - score the model and print results
# Models:
#   Week 6: HealthcareClassifier - Logistic Regression, Random Forest, Naive Bayes
#   Week 4: QLearningAgent       - Reinforcement Learning (Q-Learning)
#   Week 2: FeedforwardNN        - Neural Network (PyTorch, same structure as MNIST)

import os
import sys
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB

from config import RANDOM_STATE

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# SKLEARN CLASSIFIERS
class HealthcareClassifier:
    """
    A wrapper around three different classifiers from our Week 6 assignment.

    We compare all three because accuracy alone is misleading - which was the
    KEY lesson from Week 6. Different models have different strengths:

        Logistic Regression : simple, fast, good baseline model
        Random Forest       : more powerful, handles complex patterns
        Naive Bayes         : very fast but assumes features are independent
    """

    # Only these three model names are allowed
    SUPPORTED_MODELS = ("logistic_regression", "random_forest", "naive_bayes")

    def __init__(self, model_type="logistic_regression"):
        # Check that the user passed a valid model name
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                "model_type must be one of: " + str(self.SUPPORTED_MODELS) + "\n"
                "You passed: '" + model_type + "'"
            )

        self.model_type = model_type

        # Build whichever model was requested
        if model_type == "logistic_regression":
            # max_iter=1000 because logistics data may need more steps to converge
            self.model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

        elif model_type == "random_forest":
            # 100 decision trees vote together - this reduces overfitting
            self.model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

        elif model_type == "naive_bayes":
            # GaussianNB assumes all features follow a normal distribution
            # Fast and simple, but that assumption does not always hold
            self.model = GaussianNB()

        print("[Model] Ready: " + model_type)

    def fit(self, X_train, y_train):
        """
        Train the model on the training data.
        This is where the actual 'learning' happens.
        """
        print("[Model] Training " + self.model_type + " on " + str(len(X_train)) + " samples")
        self.model.fit(X_train, y_train)
        print("[Model] Training complete!")
        return self

    def predict(self, X):
        """
        Make predictions on new data.
        Returns an array of 0s (On-Time) and 1s (Delayed).
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Score the model using MULTIPLE metrics.

        Why not just use accuracy?
        From Week 6: accuracy alone is misleading, especially on imbalanced datasets.

        The four metrics we report:
            Accuracy 
            Precision 
            Recall  
            F1 Score 
        In logistics (and healthcare), RECALL is most important.
        Missing a real delay (false negative) costs more than a false alarm.
        """
        y_pred = self.predict(X_test)

        # Calculate all four metrics
        results = {
            "model":            self.model_type,
            "accuracy":         round(accuracy_score(y_test, y_pred), 4),
            "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # Print a clear summary to the terminal
        print("\n  --- " + self.model_type + " ---")
        print("  Accuracy  : " + str(results["accuracy"]))
        print("  Precision : " + str(results["precision"]))
        print("  Recall    : " + str(results["recall"]))
        print("  F1 Score  : " + str(results["f1"]))

        return results


# Q-LEARNING AGENT  ( Reinforcement Learning)

class QLearningAgent:
    """
    A Q-Learning Reinforcement Learning agent.

    This is taken directly from our Week 4 assignment environment:
        - 6 positions on a number line (states 0 through 5)
        - Goal : reach state 5 → big positive reward (+10)
        - Pit  : fall into state 0 → big negative reward (-10)
        - Actions: move LEFT (0) or move RIGHT (1)

    The agent does NOT know the rules at the start.
    It learns through trial and error using the Q-Learning update formula:
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s')) - Q(s, a)]

    The Q-table stores what the agent has learned:
        Rows = states, Columns = actions
        Each cell = expected total reward for taking that action from that state
    """

    def __init__(self, n_states=6, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.99, epsilon_min=0.01):

        self.n_states = n_states

        # alpha = learning rate: how fast we update our Q-value estimates
        # Too high = unstable learning. Too low = very slow learning. 0.1 is a safe default.
        self.alpha = alpha

        # gamma = discount factor: how much we value future rewards vs immediate ones
        # 0.9 means we care about the future, but slightly prefer rewards now
        self.gamma = gamma

        # epsilon = exploration rate: probability of taking a RANDOM action
        # Starts at 1.0 (explore everything at first) and slowly decreases
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize the Q-table with all zeros
        # The agent starts with NO knowledge - it learns everything from scratch
        self.q_table = np.zeros((n_states, 2))

        print("[RL Agent] Q-Learning agent created.")
        print("[RL Agent] States: " + str(n_states) + " | Q-table shape: " + str(self.q_table.shape))

    def _step(self, state, action):
        """
        Simulate one action in the environment.

        This is the environment logic from Week 4.
        The agent calls this to find out what happens after each move.
        """
        # Move left or right, but stay within the valid state range
        if action == 0:
            next_state = state - 1   # move left
        else:
            next_state = state + 1   # move right

        # Clamp to valid range so agent can't go out of bounds
        next_state = max(0, min(next_state, self.n_states - 1))

        # Check if the episode is over
        if next_state == self.n_states - 1:
            return next_state, +10.0, True    # reached the goal!
        if next_state == 0:
            return next_state, -10.0, True    # fell into the pit

        # Small penalty per step - this pushes the agent to find the shortest path
        return next_state, -0.1, False

    def train(self, episodes=500, start_state=3):
        """
        Train the agent by running many episodes.

        Each episode:
            1. Start at start_state
            2. Pick an action (random OR best known - epsilon-greedy)
            3. See what happens (_step)
            4. Update Q-table using Bellman equation
            5. Repeat until goal or pit is reached
        """
        print("\n[RL Agent] Starting training for " + str(episodes) + " episodes...")

        rewards_per_episode = []

        for ep in range(episodes):
            state = start_state
            total_reward = 0.0

            # Run one episode (cap at 200 steps to prevent infinite loops)
            for step in range(200):

                # Epsilon-greedy decision:
                # With probability epsilon → take a RANDOM action (explore)
                # Otherwise              → take the BEST KNOWN action (exploit)
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(2)              # random: explore
                else:
                    action = int(np.argmax(self.q_table[state]))   # best known: exploit

                # Take the action and get feedback from the environment
                next_state, reward, done = self._step(state, action)
                total_reward += reward

                # Update Q-table using the Bellman equation (from Week 4):
                #   new_Q = old_Q + alpha * (reward + gamma * best_future_Q - old_Q)
                best_future_q = np.max(self.q_table[next_state])
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * best_future_q - self.q_table[state, action]
                )

                state = next_state

                # End the episode if we hit the goal or the pit
                if done:
                    break

            # Decay epsilon so the agent explores less as it learns more
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_per_episode.append(total_reward)

            # Print a progress update every 100 episodes
            if (ep + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(
                    "  Episode " + str(ep + 1) + "/" + str(episodes) +
                    " | Avg reward: " + str(round(avg_reward, 2)) +
                    " | Epsilon: " + str(round(self.epsilon, 3))
                )

        print("[RL Agent] Training complete!")
        return rewards_per_episode

    def run_episode(self, start_state=3):
        """
        Run one full episode using the LEARNED policy.
        This shows the path the agent takes AFTER training - it should go
        straight to the goal without falling into the pit.
        """
        state = start_state
        path = [state]

        for step in range(50):
            # Always pick the best known action (epsilon = 0, pure exploitation)
            action = int(np.argmax(self.q_table[state]))
            state, reward, done = self._step(state, action)
            path.append(state)
            if done:
                break

        # Tell us how it ended
        if path[-1] == self.n_states - 1:
            outcome = "GOAL reached!"
        else:
            outcome = "Fell into PIT"

        print("[RL Agent] Path from state " + str(start_state) + ": " + str(path) + " -> " + outcome)
        return path


# FEEDFORWARD NEURAL NETWORK  (Week 2 - MNIST architecture, adapted)

class FeedforwardNN:
    """
    A simple feedforward neural network built with PyTorch.

    Architecture is the same as the Week 2 MNIST project, just adapted
    for tabular (spreadsheet-style) data instead of image data:

        Input layer  → 128 neurons (ReLU activation)
        Hidden layer → 64 neurons  (ReLU activation)
        Output layer → 2 neurons   (one per class: Delayed / On-Time)

    Loss function : CrossEntropyLoss  (same as Week 2)
    Optimizer     : Adam              (same as Week 2)
    Training loop : same pattern as Week 2 (zero_grad → forward → loss → backward → step)
    """

    def __init__(self, input_dim=30, output_dim=2):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Run this first:\n"
                "    pip install torch"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Use GPU if one is available (Azure VMs often have NVIDIA GPUs)
        # Otherwise fall back to regular CPU - both work fine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the network - same 3-layer structure as the Week 2 MNIST network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),   # input layer:   features → 128 neurons
            nn.ReLU(),                    # ReLU activation (same as Week 2)
            nn.Linear(128, 64),           # hidden layer:  128 → 64 neurons
            nn.ReLU(),
            nn.Linear(64, output_dim),    # output layer:  64 → number of classes
        ).to(self.device)

        # CrossEntropyLoss is the standard loss for classification (same as Week 2)
        self.criterion = nn.CrossEntropyLoss()

        # Adam optimizer - works well out of the box with no manual tuning needed
        self.optimizer = optim.Adam(self.net.parameters())

        print("[Neural Net] FeedforwardNN built on: " + str(self.device))
        print("[Neural Net] Architecture: " + str(input_dim) + " → 128 → 64 → " + str(output_dim))

    def fit(self, X, y, epochs=20, batch_size=32):
        """
        Train the neural network.

        This uses the exact same 5-step PyTorch training loop from Week 2:
            1. Reset gradients 
            2. Forward pass    
            3. Calculate loss  
            4. Backpropagation 
            5. Update weights  
        """
        print("\n[Neural Net] Training for " + str(epochs) + " epochs...")

        # Convert numpy arrays to PyTorch tensors (the format PyTorch expects)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        # DataLoader splits data into batches automatically
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set model to training mode (enables things like dropout if we add it later)
        self.net.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0

            for X_batch, y_batch in loader:
                # The 5-step PyTorch training loop (same as Week 2 MNIST):
                self.optimizer.zero_grad()                        #Reset gradients
                predictions = self.net(X_batch)                   #Forward pass
                loss = self.criterion(predictions, y_batch)       #Calculate loss
                loss.backward()                                   #Backpropagation
                self.optimizer.step()                             #Update weights
                total_loss += loss.item()

            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                avg_loss = total_loss / len(loader)
                print("  Epoch " + str(epoch) + "/" + str(epochs) + " | Loss: " + str(round(avg_loss, 4)))

        print("[Neural Net] Training complete!")

    def predict(self, X):
        """Make predictions (no gradient tracking needed here - saves memory)."""
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(X_tensor)
            # argmax picks the class with the highest score (0 or 1)
            return logits.argmax(dim=1).cpu().numpy()

    def evaluate(self, X_test, y_test):
        """Score the neural network on the test set."""
        y_pred = self.predict(X_test)

        acc = round(accuracy_score(y_test, y_pred), 4)
        f1  = round(f1_score(y_test, y_pred, zero_division=0), 4)

        print("\n   Feedforward Neural Network ")
        print("  Accuracy  : " + str(acc))
        print("  F1 Score  : " + str(f1))

        return {"model": "feedforward_nn", "accuracy": acc, "f1": f1}
