# Exploring the Scalability and Adaptability of Evolution Strategies in Reinforcement Learning

Welcome! 👋 This project benchmarks two leading reinforcement learning algorithms—**Evolution Strategies (ES)** and **Proximal Policy Optimization (PPO)**—across a suite of continuous control tasks to understand their convergence speed, performance consistency, and scalability under varying policy network sizes .

---

## 🚀 What This Project Does

* **PPO Implementation:** Builds upon the [Simple‑PPO](https://github.com/asdfGuest/Simple-PPO/tree/main) framework by **asdfGuest**, integrated with our own modifications for enhanced logging and adaptive learning‑rate scheduling.
* **Multi‑environment Benchmarking:** Trains and evaluates **PPO** and **ES** on OpenAI Gym tasks including **Bipedal Walker**, **Inverted Double Pendulum**, **Hopper**, **Walker2D**, and **Half‑Cheetah** .
* **Training Dynamics Analysis:** Compares reward‑over‑time curves and 100‑run performance histograms to assess convergence speed and consistency—showing, for example, that PPO converges faster on most tasks while ES can be less noisy on smaller models .
* **Scalability Study:** Varies policy network dimensionality (*n* = 4, 8, 16, 32, 64) to investigate how model size impacts ES vs. PPO performance, revealing that ES excels with compact policies whereas PPO benefits from larger networks .
* **Trade‑off Insights:** Analyzes the interplay between model complexity and sample efficiency, identifying scenarios where smaller ES models train faster than their PPO counterparts and vice versa .

---

## 🎓 Academic Context

This work was conducted as part of the **Reinforcement Learning** course at **Sapienza University of Rome**, in the academic year **2024/2025**, by the student group:

* **Paolo Cursi** (2155622)
* **Pietro Signorino** (2149741);

---

## 📄 Full Presentation

For a more detailed explanation the full slide deck is avaible here:
👉 [📘 View Presentation (PDF)](FinalPresentation.pdf)

---

## 🧑‍💻 Authors

* **Paolo Cursi**
* **Pietro Signorino**
