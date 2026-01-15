# HR Benefits & Compensation Analysis Dashboard

A Python-based analytical tool demonstrating data analysis skills for HR/Benefits Administration roles. This project showcases proficiency in compensation benchmarking, benefits enrollment analysis, and leave tracking — core competencies for Benefits & Compensation Administrator positions.

# MAKE MORE PERSONABLE, EXPLAIN EXCITEMENT, ENVITE HR PERSON IN, MAKE MORE COMMUNICATING A SENSE OF UNDERSTANDING AND WHY EACH STEP WAS DONE, THESE ARE THE THINGS I AM LOOKING FOR IN THIS JOB

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-orange.svg)

## 📋 Project Overview

This tool analyzes synthetic HR data to provide insights across three key areas:

### 1. **Compensation Analysis**
- Salary benchmarking by department, location, and job level
- Compa-ratio calculations (salary vs. market midpoint)
- Pay equity analysis and identification of outliers
- Salary vs. tenure trend analysis

### 2. **Benefits Enrollment Analysis**
- Medical, dental, and vision plan enrollment tracking
- 401(k) participation rates and contribution analysis
- Benefits cost analysis by department
- Total cost of benefits per employee

### 3. **Leave & PTO Tracking**
- PTO utilization rates by department
- Leave balance liability calculations
- Sick leave usage patterns
- Extended leave (FMLA, disability) tracking

## 🎯 Skills Demonstrated

This project demonstrates competencies directly relevant to Benefits & Compensation Administration:

| Skill | Application in Project |
|-------|----------------------|
| **Data Analysis** | Pandas for data manipulation, aggregation, and statistical analysis |
| **HRIS Concepts** | Simulated data structures matching real HRIS systems (like Dayforce) |
| **Compensation Benchmarking** | Compa-ratio calculations, market midpoint analysis, pay ranges |
| **Benefits Administration** | Enrollment tracking, cost analysis, participation metrics |
| **Leave Management** | PTO accruals, utilization tracking, liability calculations |
| **Data Visualization** | Professional charts for executive reporting |
| **Reporting** | Automated generation of analytics and exports |

## 📊 Sample Visualizations

The project generates 13 professional visualizations including:

- **Executive Dashboard** — High-level KPIs at a glance
- **Salary Distribution by Department** — Box plots showing pay ranges
- **Compa-Ratio Distribution** — Histogram with market benchmarks
- **401(k) Participation Analysis** — Contribution rate distribution
- **PTO Utilization by Department** — Identifying under-utilization
- **Benefits Enrollment Breakdown** — Pie charts for each benefit type

## 🗂️ Project Structure

```
hr-benefits-analyzer/
├── src/
│   └── main.py              # Main analysis script (fully commented)
├── output/
│   ├── charts/              # Generated visualizations (PNG)
│   └── data/                # Exported CSV data files
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hr-benefits-analyzer.git
   cd hr-benefits-analyzer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

```bash
python src/main.py
```

The script will:
1. Generate synthetic HR data for 150 employees
2. Run comprehensive analysis across all three domains
3. Create 13 professional visualizations
4. Export data to CSV files

All output files will be saved to the `output/` directory.

## 📈 Sample Output

### Console Output
```
======================================================================
   HR BENEFITS & COMPENSATION ANALYSIS DASHBOARD
   Portfolio Project for Benefits Administrator Role
======================================================================

[STEP 1] Generating synthetic HR data...
  Generated 150 employee records
  Generated benefits enrollment data
  Generated leave tracking data

============================================================
COMPENSATION ANALYSIS
============================================================

--- Salary Statistics by Department ---
              count      min      max       mean    median       std
department                                                          
Climbing         22  37000.0  97500.0   54159.09   47250.0  17543.21
Facilities       17  33500.0  85000.0   50264.71   43500.0  15876.43
...

--- Compa-Ratio Analysis ---
Employees below market (compa-ratio < 0.85): 12
Employees above market (compa-ratio > 1.15): 18
Average compa-ratio: 1.02
```

### Generated Files

| File | Description |
|------|-------------|
| `00_executive_dashboard.png` | Summary dashboard with key metrics |
| `01_salary_by_department.png` | Box plot of salaries by department |
| `02_salary_by_location.png` | Average salary comparison by location |
| `03_compa_ratio_distribution.png` | Histogram showing market positioning |
| `04_salary_vs_tenure.png` | Scatter plot with trend analysis |
| `05_headcount_by_dept_level.png` | Stacked bar of org structure |
| `06_medical_enrollment.png` | Pie chart of health plan selections |
| `07_401k_contributions.png` | Distribution of contribution rates |
| `08_benefits_cost_by_dept.png` | Cost analysis by department |
| `09_benefits_overview.png` | Multi-chart benefits summary |
| `10_pto_utilization.png` | PTO usage rates by department |
| `11_leave_balances.png` | Leave balance distributions |
| `12_extended_leave.png` | FMLA/disability leave breakdown |

### Data Exports

| File | Description |
|------|-------------|
| `employee_data.csv` | Complete employee roster with compensation |
| `benefits_enrollment.csv` | Benefits elections and costs |
| `leave_tracking.csv` | PTO and leave balances |
| `combined_hr_data.csv` | Merged comprehensive dataset |

## 🔧 Customization

### Adjusting Sample Size
In `main.py`, modify the `num_employees` parameter:
```python
employee_df = generate_employee_data(num_employees=500)  # Increase to 500
```

### Adding New Locations
Add to the `locations` list and `location_multipliers` dictionary:
```python
locations = [
    ...
    'Austin, TX',
    'Seattle, WA'
]

location_multipliers = {
    ...
    'Austin, TX': 1.00,
    'Seattle, WA': 1.20
}
```

### Modifying Salary Ranges
Adjust the `base_salary_ranges` dictionary:
```python
base_salary_ranges = {
    0: (40000, 50000),   # Entry level
    1: (50000, 70000),   # Mid-level
    2: (70000, 100000)   # Senior level
}
```

## 💡 Key Concepts Explained

### Compa-Ratio
A compa-ratio compares an employee's salary to the market midpoint for their role:
- **< 0.85**: Significantly below market (retention risk)
- **0.85 - 1.15**: Within competitive range
- **> 1.15**: Above market (may indicate compression issues)

### PTO Liability
The financial liability of unused PTO, calculated as:
```
PTO Liability = Unused Hours × Hourly Rate
```
This is important for budgeting and forecasting.

### Benefits Participation Rate
The percentage of eligible employees enrolled in a benefit:
```
Participation Rate = Enrolled Employees / Total Eligible × 100
```
High participation indicates effective benefits communication.

## 🛠️ Technologies Used

- **Python 3.8+** — Core programming language
- **pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization foundation
- **Seaborn** — Statistical data visualization

## 📝 Future Enhancements

Potential improvements for this project:

- [ ] Add interactive dashboards using Plotly or Dash
- [ ] Implement actual salary survey data integration
- [ ] Add Monte Carlo simulation for benefits cost forecasting
- [ ] Create database backend for persistent data storage
- [ ] Build REST API for integration with other HR systems
- [ ] Add predictive analytics for turnover risk

## 👤 Author

**Peyton Cunningham**

This project was created as a portfolio piece demonstrating analytical and technical skills relevant to Benefits & Compensation Administration roles.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with reference to common HRIS data structures
- Compensation benchmarking methodology based on industry best practices
- Inspired by real-world HR analytics challenges

---

*This is a portfolio project using synthetic data. No real employee information is used.*
