# Privacy-Preserving Federated Learning for Healthcare IoT

A privacy-preserving federated learning framework designed for secure and decentralized healthcare IoT environments.

The project explores how **Federated Learning (FL)** and **Differential Privacy (DP)** can be combined to train machine learning models across distributed clients while reducing the need to centrally collect sensitive healthcare data.

It also includes an interactive frontend for visualizing federated learning experiments, privacy parameters, training behaviour, and performance metrics.

---

## Overview

Healthcare IoT systems can generate large amounts of sensitive patient data across multiple devices and locations. Moving all of this information to a centralized server can introduce privacy and security concerns.

This project investigates a decentralized approach where participating clients train models locally and contribute model updates rather than directly sharing their underlying datasets.

Differential privacy is incorporated to further protect information contained within model updates.

### Core Idea

```text
                  ┌─────────────────────┐
                  │   Global FL Model   │
                  └──────────┬──────────┘
                             │
               Global Model Distribution
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     ┌─────────┐        ┌─────────┐        ┌─────────┐
     │ Client 1│        │ Client 2│   ...  │ Client N│
     └────┬────┘        └────┬────┘        └────┬────┘
          │                  │                  │
     Local Training     Local Training     Local Training
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                  Privacy-Protected Updates
                             │
                             ▼
                    Model Aggregation
                             │
                             ▼
                    Updated Global Model
```

---

## Key Features

- Decentralized federated learning simulation
- Multiple-client training experiments
- Differential privacy integration
- Configurable privacy parameters
- Global and client-level model analysis
- Accuracy monitoring across communication rounds
- Loss monitoring across communication rounds
- Privacy-versus-accuracy analysis
- Client-count comparison
- Training-mode comparison
- Experimental graph generation
- Interactive dashboard for visualizing FL behaviour and results

---

## Privacy-Preserving Learning

Traditional centralized machine learning typically requires data to be transferred to a central location.

Federated learning instead allows clients to perform training locally.

```text
Local Data
    │
    ▼
Local Model Training
    │
    ▼
Model Update
    │
    ▼
Differential Privacy
    │
    ▼
Federated Aggregation
    │
    ▼
Updated Global Model
```

This architecture is particularly relevant to healthcare environments where data confidentiality is important.

---

## Differential Privacy

Differential privacy is used to reduce the amount of information about individual training records that can potentially be inferred from model updates.

The project allows the relationship between **privacy and model performance** to be investigated experimentally.

This includes analysis of factors such as:

- Privacy budget
- Model accuracy
- Training rounds
- Number of participating clients
- Noise introduced during privacy-preserving training

The resulting experiments help demonstrate the trade-off between stronger privacy protection and model utility.

---

## Experimental Analysis

The project generates visual results for evaluating federated learning behaviour.

Examples include:

### Accuracy vs Communication Rounds

Tracks how global model accuracy changes as federated training progresses.

### Loss vs Communication Rounds

Shows convergence behaviour across multiple training rounds.

### Privacy vs Accuracy

Evaluates the impact of differential privacy on model performance.

### Privacy Budget vs Accuracy

Examines how different privacy configurations affect predictive performance.

### Client Scaling

Studies how changing the number of participating clients affects federated training.

### Training Mode Comparison

Compares different federated-learning configurations and experimental conditions.

Generated experiment graphs are stored within the project's graph directories.

---

## Technology Stack

### Machine Learning / Experimentation

- Python
- Federated Learning
- Differential Privacy
- Data analysis and experimental graph generation

### Frontend

- React
- JavaScript / JSX
- HTML
- CSS

### Development Tools

- Visual Studio Code
- Git
- GitHub
- npm / Node.js ecosystem

---

## Project Structure

```text
fl_project/
│
├── fedmed-project/
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── fedmed_tharun.jsx
│   │   ├── TrafficFLDashboard.jsx
│   │   └── ...
│   │
│   ├── package.json
│   └── README.md
│
├── graphs/
│
├── graphs_v2/
│   ├── accuracy-vs-rounds experiments
│   ├── loss-vs-rounds experiments
│   ├── privacy-vs-accuracy experiments
│   ├── client-scaling experiments
│   └── training-mode comparisons
│
├── generate_graphs.py
├── .gitignore
└── README.md
```

---

## Running the Frontend

### 1. Clone the repository

```bash
git clone https://github.com/Pratyush2024/Privacy-Preserving-Federated-Learning-IoT.git
```

### 2. Enter the project

```bash
cd Privacy-Preserving-Federated-Learning-IoT
```

### 3. Enter the React application

```bash
cd fedmed-project
```

### 4. Install dependencies

```bash
npm install
```

### 5. Start the application

```bash
npm start
```

The development server will then launch the application locally.

---

## Applications

The concepts demonstrated by this project can be relevant to distributed environments such as:

- Healthcare IoT systems
- Hospitals and healthcare facilities
- Remote healthcare infrastructure
- Medical IoT devices
- Privacy-sensitive distributed machine learning
- Edge computing environments

---

## Future Improvements

Potential extensions include:

- Real distributed edge-device deployment
- Secure aggregation
- Additional differential privacy mechanisms
- Larger healthcare datasets
- Comparison of multiple federated aggregation algorithms
- Communication-cost analysis
- Client dropout and fault-tolerance simulation
- Cloud/edge deployment
- Real-time IoT data integration

---

## Author

**Pratyush Dash**  
B.Tech Computer Science and Engineering  
Vellore Institute of Technology (VIT), Chennai

GitHub: [Pratyush2024](https://github.com/Pratyush2024)

---

## Disclaimer

This project is intended for academic and research purposes. It demonstrates privacy-preserving machine learning concepts and is not intended for clinical diagnosis or deployment in production healthcare systems.
