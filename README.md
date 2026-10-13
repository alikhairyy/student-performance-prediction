# Student Performance Prediction

Big Data Analysis (DDSC404) final project, built with Hamza Milad, Youssef Halawa and Abdelrahman Amgad. It follows the full **Data Analytics Lifecycle** to predict students' final grades and flag at-risk students early enough for teachers to step in.

## What's inside
| Path | Contents |
|---|---|
| [`analysis.py`](analysis.py) | End-to-end pipeline: data prep → EDA → modeling → evaluation (saves 9 figures and pickled models to `outputs/`) |
| [`dashboard/index.html`](dashboard/index.html) | Interactive HTML dashboard |
| [`powerbi/`](powerbi) | Power BI dashboard (`.pbix`) with slicers, drill-through and bookmark reset, plus a lab report |
| [`docs/`](docs) | Final report and presentation |

## Pipeline
1. **Discovery:** business problem, stakeholders, KPIs (pass rate > 75%, accuracy > 80%, MAE < 2.5)
2. **Data prep:** label-encode 12 binary categoricals, `StandardScaler`, stratified 80/20 split, Pass = G3 ≥ 10
3. **EDA:** grade distributions, correlation heatmap (G2 → G3 *r* ≈ 0.92), effects of study time, absences and failures
4. **Models:** Linear Regression (predicts G3) · Logistic Regression and Decision Tree (Pass/Fail)
5. **Evaluation:** MAE, RMSE, R², accuracy, precision, recall, F1, confusion matrices

> **Note:** `analysis.py` generates a **synthetic dataset** (395 students, 28 features) that follows the structure of the UCI Student Performance dataset, so the pipeline runs without any download.

## Run it
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python analysis.py
```

**Stack:** Python · scikit-learn · pandas · seaborn · Power BI
