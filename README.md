
# ✈️ Honeywell Hackathon — Flight Scheduling & Delay Insights

## 📌 Problem Statement

Busy airports like **Mumbai (BOM)** and **Delhi (DEL)** face congestion due to high traffic, limited runway slots, and cascading disruptions. Controllers need **AI-driven insights** to:

1. Suggest the best hours for takeoff and landing.
2. Identify congestion-prone time slots.
3. Tune flight schedules to reduce delays.
4. Detect flights with **cascading impacts** on others.

Dataset: **One week of flights at Mumbai Airport (from Flightradar24 & FlightAware).**
Constraint: **Data only available from 6:00 AM to 12:00 PM daily.**

---

## ✅ Deliverables

### 1. **Proposed Solution**

We built a **Streamlit-based interactive dashboard** integrated with **AI-powered query support**.

* Uses **open-source analytics (pandas, scikit-learn, networkx)** + **Gemini API (optional)** for natural language queries.
* Provides **visual insights** (charts, heatmaps, networks) to simplify scheduling analysis.
* Includes a **What-If simulator**: shift flight times & predict delay changes.
* Uniqueness: We merged **classical data analysis (statistics + ML)** with **plain-English AI queries**, making insights accessible to both technical staff and non-technical operators.

---

### 2. **Technical Approach**

#### ⚙️ Tech Stack

* **Language**: Python
* **Frameworks**:

  * Streamlit (UI/dashboard)
  * pandas, numpy (data analysis)
  * scikit-learn (RandomForest for prediction)
  * networkx + matplotlib (cascading impact graphs)
  * plotly (interactive charts & heatmaps)
* **AI Integration**: Gemini API (via `google-generativeai`) for NLP prompts.
* **Environment Management**: `.env` for API key, venv for dependencies.

#### 📊 Methodology & Pages

1. **Best Hours (Task 1)**

   * Method: Compare **scheduled time vs actual time**, calculate **average/minimum delay** per hour.
   * Visuals: Line plots with **confidence intervals**, bar charts, and heatmaps.
   * Tech terms explained:

     * **Mean delay**: Average waiting time.
     * **Median delay**: Middle value (ignores extremes).
     * **P90 delay**: Delay value below which 90% of flights lie.

2. **Busiest Slots (Task 2)**

   * Method: Group flights into **time slots (15/30/60 minutes)** and count flights per slot.
   * Normalized per day to account for data coverage.
   * Tech terms explained:

     * **Time slots**: Dividing hours into smaller blocks (like 9:00–9:30).
     * **Congestion**: Too many flights in the same slot.

3. **What-If Delay (Task 3)**

   * Method: Train a **RandomForest Regressor** (a machine learning model) using features like:

     * Departure hour
     * Day of week (weekday/weekend)
     * Origin/Destination
   * Simulates delay if departure time is shifted ±120 minutes.
   * Tech terms explained:

     * **RandomForest**: A collection of many decision trees voting together.
     * **Prediction R²**: A score (0–1) showing how well the model fits.
     * **RMSE**: Root Mean Square Error, shows typical prediction error in minutes.

4. **Cascades (Task 4)**

   * Method: Build a **graph network** where:

     * Each flight = a node.
     * Delay passed from one flight to the next = an edge.
     * Edge weight proportional to delay passed.
   * Identify flights with **highest outgoing influence** (top spreaders of delays).
   * Tech terms explained:

     * **Graph**: A network of nodes (flights) and edges (connections).
     * **Cascading impact**: One delay causing a ripple effect.
     * **Influence score**: Sum of all delays a flight passes forward.

5. **AI Query Page (Bonus)**

   * Method: Parse user queries using **offline regex parser** OR **Gemini API**.
   * Supports:

     * Best hours / busiest slots / what-if / cascades.
     * CSV Q\&A (e.g., “show flights from BOM to BAH on July 24”).
   * Tech terms explained:

     * **NLP (Natural Language Processing)**: Teaching computers to understand plain English.
     * **Prompt parsing**: Converting English → structured filters.

---

### 3. **Feasibility & Viability**

#### Constraints

* **Dataset limited to 6:00 AM – 12:00 PM.**

  * Means we cannot predict late evening/night flights.
  * Still useful for **morning congestion**, which is often peak at airports.

#### Time Constraint (Runway Capacity)

* **Technical term**: “Time-window capacity constraint.”
* Meaning: Each runway slot can only handle a **fixed number of flights** due to safety rules.
* Example: *Like only 2 kids can get on the bus every minute; if 10 show up, some must wait.*

#### Risks & Challenges

* **Incomplete data** (only half-day coverage).
* **Model accuracy** (R² \~ 0.5 → medium predictive power).
* **External factors** (weather, maintenance) not included.

#### Mitigation

* Normalized per-day averages to handle missing hours.
* Explain limitations clearly in UI.
* Left hooks for integration with **real-time APIs** in future.

---

### 4. **Research & References**

* FlightRadar24: [https://www.flightradar24.com](https://www.flightradar24.com)
* FlightAware: [https://www.flightaware.com](https://www.flightaware.com)
* Pandas Documentation: [https://pandas.pydata.org](https://pandas.pydata.org)
* Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
* NetworkX Documentation: [https://networkx.org](https://networkx.org)
* Plotly Express: [https://plotly.com/python/plotly-express/](https://plotly.com/python/plotly-express/)

---

## 🚀 How It Solves the Problem

* **Controllers**: See best slots instantly (instead of trial/error scheduling).
* **Airlines**: Use What-If simulation to adjust departure times.
* **ATC**: Identify high-impact flights early to manage resources.
* **Passengers**: Benefit indirectly from reduced congestion & delays.

---

## 📊 Pages in the App

1. **Best Hours** → Find hours with lowest average delays.
2. **Busiest Slots** → Visualize congestion per slot.
3. **What-If Delay** → Simulate impact of shifting a flight.
4. **Cascades** → Spot flights spreading delays.
5. **AI Query** → Ask in plain English, get structured insights.

---

## ✨ Innovation & Uniqueness

* **Combination of statistics + ML + NLP.**
* **Visualization-first** approach (heatmaps, graphs).
* **Accessible to non-technical users** (child-friendly explanations in UI).
* **Scalable design** — ready to plug into **real-time APIs** for deployment.

---

