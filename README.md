# Graph Neural Networks - Learning Rate Optimization and Evaluation

This repository contains the implementation of a Graph Attention Network (GAT) model using PyTorch Geometric to classify nodes in the Cora dataset. The project includes functionality to optimize the learning rate using a validation set and evaluate the final performance on a test set.

---

## Project Structure

- **GraphNeuralNetworks.ipynb**: The main Jupyter notebook containing the implementation and experiments.
- **README.md**: This file, providing an overview of the project.
- **data/**: Folder containing the Cora dataset (automatically downloaded by PyTorch Geometric).

---

## Methodology

### 1. **Dataset**
The Cora dataset consists of:
- **Nodes**: 2708 scientific publications.
- **Edges**: 5429 citation relationships between publications.
- **Features**: Each node is represented by a 1433-dimensional word vector.
- **Labels**: Each node belongs to one of 7 classes.

### 2. **Model**
The model is a Graph Attention Network (GAT) implemented using PyTorch Geometric's `GATConv` layer. It includes:
- Three GAT layers:
  - First layer with 4 attention heads and 64 output dimensions.
  - Second layer with 4 attention heads and 128 output dimensions.
  - Third layer with 1 attention head and 7 output dimensions.

### 3. **Optimization**
- **Grid Search**: Learning rates from the range `[0.1, 0.01, 0.001]` were tested.
- **Training**: The training data was split into 80% training and 20% validation nodes.
- **Evaluation**: The test set was evaluated using the learning rate that yielded the lowest validation loss.

---

## Installation

### Prerequisites
Ensure you have Python 3.8 or later installed. The required dependencies are listed below:
- `torch`
- `torch-geometric`
- `numpy`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
    ```
