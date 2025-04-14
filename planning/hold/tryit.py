# putting more info on tool used and prompt in commit message (before even trying to run it!)
#
# and now putting first error in next commit message (color me cynical, but my guess is that after debugging this
import numpy as np

# Simple predictive coding network
class PredictiveCodingNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.W = np.random.normal(0, 0.1, (hidden_size, input_size))  # Bottom-up weights
        self.V = np.random.normal(0, 0.1, (input_size, hidden_size))  # Top-down weights

        # Initialize activations
        self.hidden_state = np.zeros(hidden_size)
        self.prediction_error = np.zeros(input_size)

    def forward(self, input_data, iterations=10, learning_rate=0.01):
        for _ in range(iterations):
            # Compute prediction
            prediction = self.V @ self.hidden_state

            # Compute prediction error
            self.prediction_error = input_data - prediction

            # Update hidden state using prediction error and bottom-up weights
            self.hidden_state += learning_rate * (self.W @ self.prediction_error)

        return self.prediction_error, self.hidden_state

    def train(self, input_data, iterations=10, learning_rate=0.01):
        for _ in range(iterations):
            # Forward pass
            prediction_error, _ = self.forward(input_data, iterations=1, learning_rate=learning_rate)

            # Update weights based on error
            self.W += learning_rate * np.outer(self.hidden_state, prediction_error)
            self.V += learning_rate * np.outer(prediction_error, self.hidden_state)

# Example usage
if __name__ == "__main__":
    # Input data (e.g., a simple signal)
    input_data = np.array([1.0, 0.5, -0.5])

    # Create a predictive coding network
    pc_network = PredictiveCodingNetwork(input_size=3, hidden_size=2)

    # Train the network
    for epoch in range(100):
        prediction_error, hidden_state = pc_network.train(input_data)
        print(f"Epoch {epoch+1}, Prediction Error: {np.sum(prediction_error**2):.4f}")
