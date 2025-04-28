# 🚦 Applying Deep Learning for City-Wide Traffic Inference
<p align="center">
  <img src="https://github.com/harshakalluri1403/Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference/blob/d5db858965692e874766ee5a2164689d67297961/Reports/308c1fa7-f714-44f9-ac81-6d7546f62632.jpg?raw=true" width="45%" />
  <img src="https://github.com/harshakalluri1403/Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference/blob/d5db858965692e874766ee5a2164689d67297961/Reports/faa9283f-e295-4b41-b874-597cfb99d954.jpg?raw=true" width="45%" />
</p>

## Project Overview

This project applies deep learning techniques to infer city-wide traffic volumes, aiming to optimize traffic management, reduce congestion, and enhance urban mobility.  
We combine **Graph Attention Networks (GAT)** and **Long Short-Term Memory (LSTM)** models to capture **spatio-temporal traffic patterns**.

The system evaluates models using **PeMS 04**, **PeMS 08**, and **UTD-19** datasets, integrates real-time weather data, uses pathfinding algorithms for travel estimation, and provides an interactive **React** frontend for visualization.

---

## ✨ Features

- **Deep Learning Models**
  - **LSTM**: Captures temporal dependencies in traffic flow data.
  - **GAT**: Models spatial relationships using attention mechanisms.

- **Datasets**
  - **PeMS 04**: San Francisco Bay Area traffic (307 sensors, Jan-Feb 2018).
  - **PeMS 08**: San Bernardino and Riverside counties (170 sensors, Jul-Aug 2016).
  - **UTD-19**: Augsburg, Germany urban traffic (100 sensors).

- **Real-Time Weather Integration**
  - Fetches live weather using **WeatherAPI**.
  - Adjusts speed predictions dynamically (e.g., rain reduces speed by 20%).

- **Pathfinding**
  - **NetworkX** for shortest path search and Kruskal's MST for route optimization.

- **Visualization**
  - **Matplotlib** and **Seaborn** for plotting:
    - Loss curves
    - Prediction vs actual scatter plots
    - Traffic visualizations

- **User Interface**
  - **React.js** frontend for:
    - Location input
    - Model selection (GAT or LSTM)
    - Map-based traffic and ETA visualizations

---

## 📊 Results

| Model | Strength | Weakness |
| :---- | :------- | :-------- |
| **LSTM** | Excels in temporal pattern detection | Lacks spatial relationship modeling |
| **GAT** | Superior in spatial pattern detection | Computationally heavier |

### Dataset Insights

- **PeMS 04/08**: GAT outperforms LSTM for freeway traffic prediction.
- **UTD-19**: GAT achieves higher accuracy in urban traffic scenarios.

Metrics (RMSE, MAE, R²) are detailed in Table 8.1 of the report.

Visualizations of model outputs and UI screens are available in Figures 8.1–8.7 and Annexure 2.

---

## ⚠️ Limitations

- **Scalability**: Deep models are resource-intensive; hard to deploy on edge devices.
- **Data Privacy**: Centralized data poses risks; federated learning is a future direction.
- **Generalizability**: Retraining is needed when applying models across different cities.
- **Training Time**: GAT models are slower to train compared to LSTM.

---

## 🔮 Future Work

- Implement **federated learning** for privacy-preserving traffic prediction.
- Apply **model compression** for edge deployments.
- Use **transfer learning** for better model generalization across diverse urban settings.
- Improve model explainability to assist urban planners.

---

## 🛠️ Tech Stack

- **Python** (TensorFlow, PyTorch, NetworkX, Matplotlib, Seaborn)
- **React.js** (Frontend Interface)
- **WeatherAPI** (Weather data integration)
- **PeMS/UTD-19 Datasets** (Traffic data sources)

---

## 📬 Contact

For questions or collaborations, feel free to reach out!
## License 
This project is licensed under the [MIT License](https://github.com/harshakalluri1403/Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference/blob/10fe16b692252bc59c2c92f2ef213d88771b7165/LICENSE).
