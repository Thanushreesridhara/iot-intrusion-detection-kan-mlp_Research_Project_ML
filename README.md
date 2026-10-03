# IoT Intrusion Detection with KAN & MLP

### Empirical study of Kolmogorov-Arnold Networks and Multi-Layer Perceptrons for IoT cybersecurity

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=flat&logo=microsoftazure&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

---

## 🆕 Detecting Cyberattacks on IoT Devices: KAN vs. MLP vs. XGBoost

* Investigated **Kolmogorov-Arnold Networks (KAN)** for IoT cyberattack detection and compared them with **MLP and XGBoost** using the CICIoT2023 dataset.
* Prepared and evaluated the dataset by **removing duplicate records, separating training and test data**, and testing models across **5 random seeds** and training sizes from **500 to 446K samples**.
* Implemented and benchmarked **KAN, MLP, and XGBoost** models to study accuracy, training time, and performance across different training sizes.
* Found that KAN improved macro-F1 by **0.03 over a size-matched MLP**, while taking approximately **3× longer to train**; XGBoost achieved the highest macro-F1 (**0.837 vs. 0.651 for KAN**).
* Analyzed model performance across attack classes and found **brute-force and web-based attacks** to be the most difficult to detect, with **F1 ≤ 0.53**.
* Deployed the trained model as a cloud service on **Azure Kubernetes Service (AKS)** using **Docker, Terraform, and GitHub Actions**, including automated security checks.

Code, scripts and result tables for this study are in [`LLM_Cybersecurity/`](./LLM_Cybersecurity).

---

## 🔍 Overview

This project investigates **Kolmogorov-Arnold Networks (KANs)** as an alternative to conventional **Multi-Layer Perceptrons (MLPs)** for IoT intrusion detection using the **CICIoT2023** dataset.

The study compares both architectures across different training-data sizes, focusing on:

- Classification accuracy
- Training time
- Model complexity
- Scalability

The selected ML model is also deployed as a cloud-based service using **Docker, Azure Kubernetes Service (AKS), Terraform, GitHub Actions, Prometheus, and Grafana**.

---

## 🎯 Research Question

> **How do KANs and MLPs compare for IoT intrusion detection in terms of accuracy, parameter efficiency, training cost, and scalability?**

---

## 📊 Key Results

| Samples | KAN | MLP | KAN Params | MLP Params |
|--------:|----:|----:|-----------:|-----------:|
| 1,000 | **92.1%** | 89.2% | **5,737** | 14,727 |
| 5,000 | 85.38% | **86.3%** | **5,737** | 14,727 |
| 10,000 | — | **91.3%** | **5,737** | 14,727 |

### Key observations

- 🟢 **KAN performed strongly with smaller datasets**, achieving 92.1% accuracy with 1,000 samples.
- 🔵 **MLP became more competitive as the dataset increased.**
- 📦 KAN used approximately **61% fewer parameters** than the evaluated MLP.
- ⚡ MLP showed substantially lower training time at larger dataset sizes.
- ⚖️ The results highlight a trade-off between **parameter efficiency and computational cost**.

---

## 🧪 Methodology

```text
CICIoT2023
     │
     ▼
Preprocessing
     │
     ├──────────────┐
     ▼              ▼
    KAN            MLP
     │              │
     └──────┬───────┘
            ▼
      Model Evaluation
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
 Accuracy  Time  Parameters
            │
            ▼
     Cloud Deployment
```

---

## ☁️ ML System Deployment

The trained model is integrated into a cloud-based ML service:

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ▼
Docker
   │
   ▼
Azure Container Registry
   │
   ▼
Azure Kubernetes Service
   │
   ├── Prometheus
   └── Grafana
```

Infrastructure is managed using **Terraform**, with automated CI/CD and security checks through GitHub Actions.

---

## 🛠️ Technologies

**Machine Learning:**  
Python · PyTorch · Scikit-learn · KAN · MLP · Pandas · NumPy

**Cloud & MLOps:**  
Azure · AKS · Docker · Kubernetes · Terraform · GitHub Actions

**Monitoring:**  
Prometheus · Grafana

**Application:**  
Flask · REST API

---

## 🔬 Research Contributions

This project demonstrates:

- Empirical comparison of KAN and MLP architectures for IoT intrusion detection
- Evaluation across different training-data sizes
- Analysis of accuracy, parameter count, and training cost
- Cloud deployment of an ML inference service
- Reproducible infrastructure using Infrastructure as Code
- Automated deployment and security practices
- Monitoring of the deployed ML system

---

## 🚧 Limitations & Future Work

The experiments were conducted on selected subsets of CICIoT2023. The KAN experiment at 10,000 samples became computationally expensive within the available environment.

Future work could investigate:

- Larger datasets
- More efficient KAN implementations
- Hyperparameter optimization
- Feature selection
- Additional IoT intrusion datasets
- Online model retraining
- Deployment cost and latency

---

## 📄 Project Report

For the complete methodology, implementation details, experiments, and analysis:

**[View the Research Project Report](./docs/research-report.pdf)**

---

## 👤 Author

**Thanushree Sridhara**  
M.Sc. Computer & Systems Engineering · TU Ilmenau, Germany

[LinkedIn](https://linkedin.com/in/thanushree-sridhara) · [GitHub](https://github.com/Thanushreesridhara)
