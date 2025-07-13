# 🛡️ Weighted Privacy Protection in Federated Learning using Particle-Antiparticle Method

This repository extends the privacy-preserving federated learning protocol described in the original paper by integrating **weighted aggregation**. Unlike the base approach that assumes equal contribution from all clients, this enhanced version supports **heterogeneous client importance** using a novel **Particle-Antiparticle** encryption-inspired method.

## 📄 Contents

- `final.py` – Python simulation of the proposed protocol using 3 clients and simple synthetic data.
- `explaination.pdf` – Detailed explanation and mathematical proof of the Particle-Antiparticle method with weighted aggregation.
- `original_code/` – Reference implementation based on the original paper's approach.

---

## 📘 Method Overview

Federated Learning (FL) enables clients (e.g., insurers) to collaboratively train a model without sharing their raw data. In this extension:

- Clients **encrypt** their model updates using "particles" and "anti-particles."
- Each client is assigned a **weight** representing its importance (e.g., data volume, credibility).
- The server aggregates encrypted parameters via a **weighted protocol** that preserves privacy while accurately reflecting client significance.
