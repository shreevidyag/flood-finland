# Finland Flood Data Analytics

A complete data analytics project using **real Finnish flood and hydrology data** from the
[Finnish Environment Institute (SYKE)](https://www.syke.fi/en-US) and
[Finnish Meteorological Institute (FMI)](https://en.ilmatieteenlaitos.fi/open-data).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/Code-MIT-green)
![Data](https://img.shields.io/badge/Data-CC%20BY%204.0-orange)

---
## Live demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flood-finland-9sdmpzsynjvpculbkyqygc.streamlit.app/)
--------

## What This Project Does

This project downloads, analyses, and visualises flood data across Finland's
major river basins. It covers 8 monitoring stations, 22 official EU flood risk
areas, and 10 years of daily hydrology data.

### Analysis features
- **Flood event detection** — finds events where water level exceeds the P95 threshold for 2+ days
- **Spring flood peaks** (*kevättulva*) — extracts annual April–May flood peaks per station
- **Return period analysis** — fits Gumbel EV-I model to estimate 1/2a to 1/1000a flood levels
- **Trend analysis** — Mann-Kendall non-parametric test + Sen's slope per station
- **Seasonal patterns** — monthly water level statistics showing Finland's double flood season
- **Precipitation correlation** — Pearson correlation between rainfall and river discharge

---

## Project Structure

```
flood-finland/
├── src/
│   ├── fetch_data.py       # Downloads SYKE + FMI data (synthetic fallback if offline)
│   ├── analysis.py         # Full statistical analysis pipeline
│   └── dashboard.py        # Interactive 8-page Streamlit dashboard
├── tests/
│   └── test_analysis.py    # 20 unit tests (pytest)
├── data/
│   ├── raw/                # Downloaded CSV files land here
│   └── processed/          # Analysis output CSVs land here
├── outputs/                # Generated PNG charts
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/flood-finland.git
cd flood-finland
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Fetch the data
```bash
python src/fetch_data.py
```

### 5. Run the analysis
```bash
python src/analysis.py
```

### 6. Launch the dashboard
```bash
streamlit run src/dashboard.py
```

Opens at **http://localhost:8501**

---

## Dashboard Pages

| Page | What it shows |
|------|--------------|
| 📊 Overview | Key metrics, top flood events, risk level distribution |
| 📈 Water Levels | Interactive time series per station with rolling mean |
| 🌧️ Precipitation & Discharge | River flow rates and monthly climatology |
| 🔴 Flood Events | Timeline and table of all detected flood events |
| 📅 Seasonal Analysis | Monthly water level patterns — spring and autumn peaks |
| 📉 Return Periods | Gumbel EV-I frequency curves + official SYKE hazard levels |
| 🗺️ Risk Map | Interactive map of Finland's 22 EU flood risk areas |
| 📋 Trend Analysis | Long-term Mann-Kendall trends per station |

---

## 🏞️ Monitoring Stations

| Station | River | City |
|---------|-------|------|
| kemijoki_rovaniemi | Kemijoki | Rovaniemi |
| kemijoki_isohaara | Kemijoki | Kemi |
| kokemaenjoki_pori | Kokemäenjoki | Pori |
| kokemaenjoki_tampere | Nokianvirta | Nokia |
| kymijoki_kouvola | Kymijoki | Kouvola |
| vuoksi_imatra | Vuoksi | Imatra |
| oulujoki_oulu | Oulujoki | Oulu |
| aurajoki_turku | Aurajoki | Turku |

---

## 🗺️ Flood Risk Areas Covered

Finland has 22 nationally designated flood risk areas under the
EU Floods Directive (2007/60/EC). This project covers all of them including:

- **Rovaniemi** — Kemijoki river (very high risk, 3,800 people at risk)
- **Pori** — Kokemäenjoki river (very high risk, 5,200 people at risk)
- **Oulu** — Oulujoki river (high risk, 6,200 people at risk)
- **Helsinki** — Vantaanjoki pluvial flooding (medium risk, 12,000 people at risk)
- **Tampere**, **Turku**, **Joensuu**, **Lappeenranta** and 14 more

---

## 📦 Data Sources

| Source | Data | License |
|--------|------|---------|
| [SYKE Hydrology API](https://www.syke.fi/en-US/Open_information/Open_web_services/Environmental_data_API) | Water levels, discharge | CC BY 4.0 |
| [FMI Open Data](https://en.ilmatieteenlaitos.fi/open-data) | Precipitation, temperature, snow | CC BY 4.0 |
| [SYKE Flood Risk Areas](https://luontotieto.syke.fi) | 22 EU risk zones | CC BY 4.0 |
| [SYKE Flood Hazard Zones](https://www.syke.fi) | Return level depths and areas | CC BY 4.0 |

> **Note:** If the live APIs are unavailable, the project automatically generates
> realistic synthetic data based on published SYKE hydrological statistics.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output: **20 passed**

Tests cover flood detection, spring peak extraction, Gumbel fitting,
Mann-Kendall trend test, and seasonal statistics.

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo
5. Set main file path to: `src/dashboard.py`
6. Click **Deploy**

The dashboard auto-fetches all data on first load — no setup needed.

---

## 🔬 Methods

### Flood Detection
Water level observations are compared against a station-specific 95th percentile
threshold. Periods exceeding this threshold for at least 2 consecutive days are
classified as flood events.

### Return Period Analysis
Annual spring flood maxima (March–June) are fitted to a Gumbel Extreme Value
Type I distribution. Return levels for 2, 5, 10, 20, 50, 100, 250, 500 and
1000 year return periods are estimated.

### Trend Analysis
The non-parametric Mann-Kendall test is applied to annual mean water levels.
Sen's slope estimator provides the rate of change in metres per year.
A p-value below 0.05 indicates a statistically significant trend.

---

## 📄 License

- **Code:** MIT License
- **Data:** CC BY 4.0 — Finnish Environment Institute (SYKE) & Finnish Meteorological Institute (FMI)

---

## 👤 Author

Shree Vidya Gurudath. Built as a data analytics portfolio project using open Finnish government data.
