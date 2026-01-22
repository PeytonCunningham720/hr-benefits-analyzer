"""
HR Benefits & Compensation Analysis Dashboard
==============================================
A Python project demonstrating data analysis skills for HR/Benefits Administration.
This tool analyzes compensation data, benefits enrollment, and leave tracking
for Movement's Colorado gyms.

Author: Peyton Cunningham
Created for: Benefits & Compensation Administrator Portfolio Project
"""

# ============================================================================
# IMPORTS - Loading all necessary libraries for data analysis and visualization
# ============================================================================

# pandas: The core library for data manipulation and analysis
# Used for reading CSV files, filtering data, calculating statistics
import pandas as pd

# numpy: Numerical computing library for mathematical operations
# Used for calculations like averages, percentiles, and array operations
import numpy as np

# matplotlib.pyplot: The main plotting library for creating visualizations
# Used to create bar charts, line graphs, and other visual representations
import matplotlib.pyplot as plt

# seaborn: Statistical data visualization built on matplotlib
# Provides prettier default styles and easier complex visualizations
import seaborn as sns

# datetime: Built-in module for working with dates and times
# Used for calculating tenure, leave accruals, and date-based analysis
from datetime import datetime, timedelta

# os: Operating system interface for file and directory operations
# Used to create output folders and manage file paths
import os

# warnings: Used to suppress unnecessary warning messages
# Keeps the output clean and focused on our analysis results
import warnings

# Suppress warnings to keep output clean during execution
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Setting up visual styles and output directories
# ============================================================================

# Set the visual style for all seaborn plots to 'whitegrid'
# This provides a clean, professional look with subtle gridlines
sns.set_style('whitegrid')

# Set a consistent color palette for all visualizations
# 'husl' provides distinct, colorblind-friendly colors
sns.set_palette('husl')

# Configure matplotlib to use a larger default figure size
# (12, 8) means 12 inches wide by 8 inches tall
plt.rcParams['figure.figsize'] = (12, 8)

# Set the default font size for better readability
plt.rcParams['font.size'] = 11

# Define the output directory where all generated charts will be saved
OUTPUT_DIR = 'output/charts'

# Create the output directory if it doesn't already exist
# exist_ok=True prevents errors if the directory already exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# DATA GENERATION - Creating realistic synthetic HR data for Movement Colorado
# ============================================================================

def generate_employee_data():
    """
    Generate synthetic employee data for Movement Climbing's Colorado gyms.
    
    This function creates realistic fake data that mimics what you'd find
    in an actual HRIS (Human Resource Information System) like Dayforce.
    
    The data reflects Movement's actual Colorado locations and job positions.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing employee demographic and compensation data
    """
    
    # Set a random seed for reproducibility
    # This ensures the same "random" data is generated each time
    np.random.seed(42)
    
    # =========================================================================
    # MOVEMENT COLORADO GYM LOCATIONS
    # These are the six Movement gyms in the Colorado region
    # =========================================================================
    locations = [
        'Movement Englewood',
        'Movement Baker',
        'Movement RiNo',
        'Movement Golden',
        'Movement Boulder',
        'Movement Centennial'
    ]
    
    # =========================================================================
    # JOB POSITIONS AND STAFFING STRUCTURE
    # Based on typical climbing gym staffing model
    # Each gym has similar structure with management + staff
    # =========================================================================
    
    # Define position categories with their details
    # Format: (job_title, department, job_level, base_salary_min, base_salary_max, count_per_gym)
    # job_level: 0=entry, 1=mid-level/supervisor, 2=management
    
    positions_per_gym = [
        # Management (1 each per gym)
        ('Gym Director', 'Management', 2, 55000, 70000, 1),
        ('Assistant Gym Director', 'Management', 2, 45000, 55000, 1),
        
        # Front Desk (varies by gym size)
        ('Front Desk Supervisor', 'Operations', 1, 38000, 45000, 1),
        ('Front Desk Staff', 'Operations', 0, 30000, 38000, 6),
        
        # Climbing/Route Setting
        ('Head Route Setter', 'Climbing', 2, 50000, 62000, 1),
        ('Route Setter', 'Climbing', 1, 38000, 48000, 4),
        
        # Fitness & Yoga (often part-time, shown as FTE equivalent)
        ('Fitness Instructor', 'Fitness', 0, 32000, 42000, 3),
        ('Yoga Teacher', 'Fitness', 0, 30000, 40000, 3),
        
        # Youth Programs
        ('Youth Program Manager', 'Youth Programs', 1, 40000, 50000, 1),
        ('ASP Instructor', 'Youth Programs', 0, 32000, 40000, 2),
        
        # Competitive Team
        ('Team Coach', 'Youth Programs', 1, 38000, 48000, 2),
    ]
    
    # Initialize empty lists to store generated data for each column
    employee_ids = []        # Unique identifier for each employee
    employee_names = []      # Employee names (placeholder)
    dept_list = []           # Department assignment
    title_list = []          # Job title
    location_list = []       # Work location (gym)
    hire_dates = []          # Date of hire
    salaries = []            # Annual salary
    job_levels = []          # Job level (0=entry, 1=mid, 2=senior)
    
    # Counter for employee IDs
    emp_counter = 1
    
    # Generate employees for each gym location
    for gym in locations:
        
        # For each position type at this gym
        for position in positions_per_gym:
            job_title, department, level, min_sal, max_sal, count = position
            
            # Create the specified number of employees for this position
            for _ in range(count):
                
                # Create a unique employee ID with prefix 'EMP' and 4-digit number
                employee_ids.append(f'EMP{emp_counter:04d}')
                
                # Generate a placeholder name
                employee_names.append(f'Employee {emp_counter}')
                
                # Assign department
                dept_list.append(department)
                
                # Assign job level
                job_levels.append(level)
                
                # Assign job title
                title_list.append(job_title)
                
                # Assign gym location
                location_list.append(gym)
                
                # Generate a random hire date within the last 6 years
                # Movement Colorado has been growing, so mix of tenure
                days_employed = np.random.randint(30, 6*365)
                hire_date = datetime.now() - timedelta(days=days_employed)
                hire_dates.append(hire_date.strftime('%Y-%m-%d'))
                
                # Calculate salary with some variation
                # Add slight randomness within the range
                base_salary = np.random.uniform(min_sal, max_sal)
                
                # Round to nearest $500 for realism
                final_salary = round(base_salary / 500) * 500
                salaries.append(final_salary)
                
                # Increment counter
                emp_counter += 1
    
    # Create a DataFrame by combining all the generated lists
    df = pd.DataFrame({
        'employee_id': employee_ids,
        'name': employee_names,
        'department': dept_list,
        'job_title': title_list,
        'job_level': job_levels,
        'location': location_list,
        'hire_date': hire_dates,
        'annual_salary': salaries
    })
    
    # Convert hire_date column from string to datetime type
    df['hire_date'] = pd.to_datetime(df['hire_date'])
    
    # Calculate tenure in years for each employee
    df['tenure_years'] = (datetime.now() - df['hire_date']).dt.days / 365.25
    
    # Round tenure to 1 decimal place for cleaner display
    df['tenure_years'] = df['tenure_years'].round(1)
    
    # Return the completed DataFrame
    return df


def generate_benefits_data(employee_df):
    """
    Generate synthetic benefits enrollment data.
    
    This simulates data from benefits administration systems,
    tracking which employees are enrolled in which benefit plans.
    
    Parameters:
    -----------
    employee_df : pandas.DataFrame
        The employee data DataFrame (used to get employee IDs)
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing benefits enrollment information
    """
    
    # Set random seed for reproducibility
    np.random.seed(43)
    
    # Get the list of all employee IDs from the employee DataFrame
    employee_ids = employee_df['employee_id'].tolist()
    
    # Define available medical plan options with monthly costs
    # These represent typical employer-sponsored health plan tiers
    medical_plans = {
        'PPO Premium': 650,      # High-cost, most flexibility
        'PPO Standard': 450,     # Mid-range option
        'HDHP with HSA': 300,    # High-deductible, lower premium
        'Waived': 0              # Employee opted out (has other coverage)
    }
    
    # Define dental plan options
    dental_plans = {
        'Dental Plus': 75,       # Comprehensive coverage
        'Dental Basic': 45,      # Basic preventive coverage
        'Waived': 0              # Employee opted out
    }
    
    # Define vision plan options
    vision_plans = {
        'Vision Premium': 25,    # Full coverage including frames
        'Vision Basic': 15,      # Basic exam coverage
        'Waived': 0              # Employee opted out
    }
    
    # Initialize lists to store benefits data for each employee
    medical_selections = []      # Which medical plan each employee chose
    dental_selections = []       # Which dental plan each employee chose
    vision_selections = []       # Which vision plan each employee chose
    retirement_enrolled = []     # Whether enrolled in 401(k)
    retirement_contrib = []      # 401(k) contribution percentage
    life_insurance = []          # Life insurance coverage multiple
    
    # Generate benefits elections for each employee
    for emp_id in employee_ids:
        
        # Medical plan selection with realistic distribution
        # Most employees choose some coverage, few waive
        medical = np.random.choice(
            list(medical_plans.keys()),
            p=[0.25, 0.40, 0.25, 0.10]  # Probabilities for each plan
        )
        medical_selections.append(medical)
        
        # Dental plan selection - high participation rate
        dental = np.random.choice(
            list(dental_plans.keys()),
            p=[0.35, 0.50, 0.15]
        )
        dental_selections.append(dental)
        
        # Vision plan selection - moderate participation
        vision = np.random.choice(
            list(vision_plans.keys()),
            p=[0.30, 0.45, 0.25]
        )
        vision_selections.append(vision)
        
        # 401(k) enrollment - 75% participation rate
        enrolled = np.random.choice([True, False], p=[0.75, 0.25])
        retirement_enrolled.append(enrolled)
        
        # 401(k) contribution rate (if enrolled)
        if enrolled:
            contrib = np.random.choice(
                [3, 4, 5, 6, 7, 8, 10, 12, 15],
                p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
            )
        else:
            contrib = 0
        retirement_contrib.append(contrib)
        
        # Life insurance selection (as multiple of salary)
        life_mult = np.random.choice(
            [1, 2, 3, 4, 5],
            p=[0.40, 0.30, 0.15, 0.10, 0.05]
        )
        life_insurance.append(life_mult)
    
    # Create the benefits DataFrame
    benefits_df = pd.DataFrame({
        'employee_id': employee_ids,
        'medical_plan': medical_selections,
        'dental_plan': dental_selections,
        'vision_plan': vision_selections,
        '401k_enrolled': retirement_enrolled,
        '401k_contribution_pct': retirement_contrib,
        'life_insurance_multiple': life_insurance
    })
    
    # Calculate monthly costs for each benefit type
    benefits_df['medical_monthly_cost'] = benefits_df['medical_plan'].map(medical_plans)
    benefits_df['dental_monthly_cost'] = benefits_df['dental_plan'].map(dental_plans)
    benefits_df['vision_monthly_cost'] = benefits_df['vision_plan'].map(vision_plans)
    
    # Calculate total monthly benefits cost per employee
    benefits_df['total_monthly_cost'] = (
        benefits_df['medical_monthly_cost'] + 
        benefits_df['dental_monthly_cost'] + 
        benefits_df['vision_monthly_cost']
    )
    
    # Calculate annual benefits cost
    benefits_df['annual_benefits_cost'] = benefits_df['total_monthly_cost'] * 12
    
    return benefits_df


def generate_leave_data(employee_df):
    """
    Generate synthetic leave/PTO tracking data.
    
    This simulates data used for leave administration,
    tracking accruals, usage, and balances.
    
    Parameters:
    -----------
    employee_df : pandas.DataFrame
        The employee data DataFrame (for employee IDs and tenure)
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing leave accrual and usage information
    """
    
    # Set random seed for reproducibility
    np.random.seed(44)
    
    # Initialize list for leave records
    leave_records = []
    
    # Define PTO accrual rates based on tenure (hours per year)
    def get_pto_accrual(tenure_years):
        """Determine annual PTO accrual based on years of service."""
        if tenure_years < 1:
            return 80    # 10 days for first year
        elif tenure_years < 3:
            return 120   # 15 days for 1-3 years
        elif tenure_years < 5:
            return 160   # 20 days for 3-5 years
        else:
            return 200   # 25 days for 5+ years
    
    # Generate leave data for each employee
    for _, emp in employee_df.iterrows():
        
        # Get employee's annual PTO accrual based on tenure
        annual_accrual = get_pto_accrual(emp['tenure_years'])
        
        # Calculate PTO used (random, but generally less than accrued)
        usage_rate = np.random.uniform(0.60, 0.95)
        pto_used = int(annual_accrual * usage_rate)
        
        # Calculate remaining PTO balance
        pto_balance = annual_accrual - pto_used
        
        # Generate sick leave data (typically fixed accrual)
        sick_accrual = 64  # 8 days per year
        sick_used = np.random.randint(0, 48)
        sick_balance = sick_accrual - sick_used
        
        # Track any extended leaves (FMLA, disability, etc.)
        has_extended_leave = np.random.choice([True, False], p=[0.08, 0.92])
        
        if has_extended_leave:
            leave_type = np.random.choice(['FMLA', 'Short-term Disability', 'Parental Leave'])
            leave_days = np.random.randint(10, 60)
        else:
            leave_type = 'None'
            leave_days = 0
        
        # Compile the record for this employee
        leave_records.append({
            'employee_id': emp['employee_id'],
            'pto_accrual_hours': annual_accrual,
            'pto_used_hours': pto_used,
            'pto_balance_hours': pto_balance,
            'sick_accrual_hours': sick_accrual,
            'sick_used_hours': sick_used,
            'sick_balance_hours': sick_balance,
            'extended_leave_type': leave_type,
            'extended_leave_days': leave_days
        })
    
    # Convert list of dictionaries to DataFrame
    leave_df = pd.DataFrame(leave_records)
    
    return leave_df


# ============================================================================
# ANALYSIS FUNCTIONS - Core analytical capabilities for HR metrics
# ============================================================================

def analyze_compensation(employee_df):
    """
    Perform comprehensive compensation analysis.
    
    This function calculates key compensation metrics including:
    - Salary statistics by department and location
    - Compa-ratios (comparison to market midpoint)
    - Pay equity analysis
    
    Parameters:
    -----------
    employee_df : pandas.DataFrame
        The employee data DataFrame
    
    Returns:
    --------
    dict
        Dictionary containing various compensation analysis DataFrames
    """
    
    # Print section header
    print("\n" + "="*60)
    print("COMPENSATION ANALYSIS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Calculate salary statistics by department
    # -------------------------------------------------------------------------
    
    dept_stats = employee_df.groupby('department')['annual_salary'].agg([
        ('count', 'count'),
        ('min', 'min'),
        ('max', 'max'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std')
    ]).round(2)
    
    print("\n--- Salary Statistics by Department ---")
    print(dept_stats.to_string())
    
    # -------------------------------------------------------------------------
    # Calculate salary statistics by gym location
    # -------------------------------------------------------------------------
    
    location_stats = employee_df.groupby('location')['annual_salary'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median')
    ]).round(2)
    
    print("\n--- Salary Statistics by Gym Location ---")
    print(location_stats.to_string())
    
    # -------------------------------------------------------------------------
    # Calculate compa-ratios
    # -------------------------------------------------------------------------
    
    # Define market midpoints by job level for Colorado market
    market_midpoints = {
        0: 35000,   # Entry level market midpoint
        1: 44000,   # Mid-level market midpoint  
        2: 58000    # Management level market midpoint
    }
    
    # Calculate compa-ratio for each employee
    employee_df['market_midpoint'] = employee_df['job_level'].map(market_midpoints)
    employee_df['compa_ratio'] = employee_df['annual_salary'] / employee_df['market_midpoint']
    employee_df['compa_ratio'] = employee_df['compa_ratio'].round(2)
    
    # Identify employees significantly below or above market
    below_market = employee_df[employee_df['compa_ratio'] < 0.85]
    above_market = employee_df[employee_df['compa_ratio'] > 1.15]
    
    print(f"\n--- Compa-Ratio Analysis ---")
    print(f"Employees below market (compa-ratio < 0.85): {len(below_market)}")
    print(f"Employees above market (compa-ratio > 1.15): {len(above_market)}")
    print(f"Average compa-ratio: {employee_df['compa_ratio'].mean():.2f}")
    
    # -------------------------------------------------------------------------
    # Pay range analysis by job title
    # -------------------------------------------------------------------------
    
    title_ranges = employee_df.groupby('job_title')['annual_salary'].agg([
        ('count', 'count'),
        ('min', 'min'),
        ('max', 'max'),
        ('range', lambda x: x.max() - x.min())
    ]).round(2)
    
    title_ranges = title_ranges.sort_values('count', ascending=False)
    
    print("\n--- Pay Ranges by Job Title ---")
    print(title_ranges.to_string())
    
    return {
        'department_stats': dept_stats,
        'location_stats': location_stats,
        'below_market': below_market,
        'above_market': above_market,
        'title_ranges': title_ranges,
        # NEW: Keys needed for README template replacement
        'below_market_count': len(below_market),
        'avg_compa_ratio': employee_df['compa_ratio'].mean()
    }


def analyze_benefits(benefits_df, employee_df):
    """
    Analyze benefits enrollment patterns and costs.
    
    Parameters:
    -----------
    benefits_df : pandas.DataFrame
        The benefits enrollment data
    employee_df : pandas.DataFrame
        The employee data (for joining additional info)
    
    Returns:
    --------
    dict
        Dictionary containing benefits analysis results
    """
    
    print("\n" + "="*60)
    print("BENEFITS ENROLLMENT ANALYSIS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Medical plan enrollment distribution
    # -------------------------------------------------------------------------
    
    medical_enrollment = benefits_df['medical_plan'].value_counts()
    medical_pct = (medical_enrollment / len(benefits_df) * 100).round(1)
    
    print("\n--- Medical Plan Enrollment ---")
    for plan, count in medical_enrollment.items():
        print(f"  {plan}: {count} employees ({medical_pct[plan]}%)")
    
    # -------------------------------------------------------------------------
    # 401(k) analysis
    # -------------------------------------------------------------------------
    
    participation_rate = benefits_df['401k_enrolled'].mean() * 100
    enrolled_contribs = benefits_df[benefits_df['401k_enrolled']]['401k_contribution_pct']
    avg_contribution = enrolled_contribs.mean()
    
    print(f"\n--- 401(k) Analysis ---")
    print(f"  Participation Rate: {participation_rate:.1f}%")
    print(f"  Average Contribution Rate: {avg_contribution:.1f}%")
    print(f"  Employees at 10%+ contribution: {len(enrolled_contribs[enrolled_contribs >= 10])}")
    
    # -------------------------------------------------------------------------
    # Cost analysis
    # -------------------------------------------------------------------------
    
    total_annual_cost = benefits_df['annual_benefits_cost'].sum()
    avg_annual_cost = benefits_df['annual_benefits_cost'].mean()
    
    print(f"\n--- Benefits Cost Analysis ---")
    print(f"  Total Annual Benefits Cost: ${total_annual_cost:,.2f}")
    print(f"  Average Cost per Employee: ${avg_annual_cost:,.2f}")
    
    # Merge with employee data to analyze by department
    merged_df = benefits_df.merge(employee_df[['employee_id', 'department']], on='employee_id')
    
    dept_costs = merged_df.groupby('department')['annual_benefits_cost'].agg([
        ('total_cost', 'sum'),
        ('avg_cost', 'mean'),
        ('employee_count', 'count')
    ]).round(2)
    
    print("\n--- Benefits Cost by Department ---")
    print(dept_costs.to_string())
    
    return {
        'medical_enrollment': medical_enrollment,
        'participation_rate': participation_rate,
        'avg_contribution': avg_contribution,
        'dept_costs': dept_costs,
        # NEW: Keys needed for README template replacement
        'total_annual_cost': total_annual_cost,
        'employee_count': len(benefits_df)
    }


def analyze_leave(leave_df, employee_df):
    """
    Analyze leave usage patterns and balances.
    
    Parameters:
    -----------
    leave_df : pandas.DataFrame
        The leave tracking data
    employee_df : pandas.DataFrame
        The employee data
    
    Returns:
    --------
    dict
        Dictionary containing leave analysis results
    """
    
    print("\n" + "="*60)
    print("LEAVE & PTO ANALYSIS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # PTO utilization analysis
    # -------------------------------------------------------------------------
    
    leave_df['pto_utilization'] = leave_df['pto_used_hours'] / leave_df['pto_accrual_hours']
    
    avg_utilization = leave_df['pto_utilization'].mean() * 100
    total_pto_balance = leave_df['pto_balance_hours'].sum()
    
    print(f"\n--- PTO Utilization ---")
    print(f"  Average PTO Utilization Rate: {avg_utilization:.1f}%")
    print(f"  Total Unused PTO Hours: {total_pto_balance:,}")
    print(f"  Total Unused PTO Days: {total_pto_balance/8:,.0f}")
    
    # Merge with employee data to get salary for liability calculation
    merged_leave = leave_df.merge(
        employee_df[['employee_id', 'annual_salary', 'department', 'location']], 
        on='employee_id'
    )
    
    # Calculate PTO liability
    merged_leave['hourly_rate'] = merged_leave['annual_salary'] / 2080
    merged_leave['pto_liability'] = merged_leave['pto_balance_hours'] * merged_leave['hourly_rate']
    
    total_liability = merged_leave['pto_liability'].sum()
    
    print(f"  Total PTO Liability: ${total_liability:,.2f}")
    
    # -------------------------------------------------------------------------
    # Sick leave analysis
    # -------------------------------------------------------------------------
    
    avg_sick_used = leave_df['sick_used_hours'].mean()
    total_sick_balance = leave_df['sick_balance_hours'].sum()
    
    print(f"\n--- Sick Leave Usage ---")
    print(f"  Average Sick Hours Used: {avg_sick_used:.1f}")
    print(f"  Total Unused Sick Hours: {total_sick_balance:,}")
    
    # -------------------------------------------------------------------------
    # Extended leave analysis
    # -------------------------------------------------------------------------
    
    extended_leaves = leave_df[leave_df['extended_leave_type'] != 'None']
    
    print(f"\n--- Extended Leave Summary ---")
    print(f"  Employees on Extended Leave: {len(extended_leaves)}")
    
    if len(extended_leaves) > 0:
        leave_type_counts = extended_leaves['extended_leave_type'].value_counts()
        for leave_type, count in leave_type_counts.items():
            print(f"    {leave_type}: {count} employees")
    
    # PTO utilization by gym location
    gym_utilization = merged_leave.groupby('location').agg({
        'pto_utilization': 'mean',
        'pto_liability': 'sum'
    }).round(2)
    
    gym_utilization['pto_utilization'] = (gym_utilization['pto_utilization'] * 100).round(1)
    gym_utilization.columns = ['utilization_pct', 'pto_liability']
    
    print("\n--- PTO Utilization by Gym Location ---")
    print(gym_utilization.to_string())
    
    # NEW: Find lowest utilization gym for README template
    lowest_util_gym = gym_utilization['utilization_pct'].idxmin()
    lowest_util_pct = gym_utilization['utilization_pct'].min()
    
    return {
        'avg_utilization': avg_utilization,
        'total_liability': total_liability,
        'extended_leaves': extended_leaves,
        'gym_utilization': gym_utilization,
        # NEW: Keys needed for README template replacement
        'lowest_util_gym': lowest_util_gym,
        'lowest_util_pct': lowest_util_pct
    }


# ============================================================================
# VISUALIZATION FUNCTIONS - Creating charts and graphs
# ============================================================================

def create_compensation_visualizations(employee_df, analysis_results):
    """
    Generate visualizations for compensation analysis.
    """
    
    print("\n" + "="*60)
    print("GENERATING COMPENSATION VISUALIZATIONS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Chart 1: Salary Distribution by Department (Box Plot)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(14, 8))
    
    dept_order = employee_df.groupby('department')['annual_salary'].median().sort_values(ascending=False).index
    
    sns.boxplot(
        data=employee_df,
        x='department',
        y='annual_salary',
        order=dept_order,
        palette='viridis'
    )
    
    plt.title('Salary Distribution by Department - Movement Colorado', fontsize=16, fontweight='bold')
    plt.xlabel('Department', fontsize=12)
    plt.ylabel('Annual Salary ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_salary_by_department.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/01_salary_by_department.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 2: Average Salary by Gym Location (Horizontal Bar Chart)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(12, 8))
    
    location_avg = employee_df.groupby('location')['annual_salary'].mean().sort_values()
    
    colors = sns.color_palette('viridis', len(location_avg))
    bars = plt.barh(location_avg.index, location_avg.values, color=colors)
    
    for bar, val in zip(bars, location_avg.values):
        plt.text(val + 500, bar.get_y() + bar.get_height()/2, 
                f'${val:,.0f}', va='center', fontsize=10)
    
    plt.title('Average Salary by Gym Location', fontsize=16, fontweight='bold')
    plt.xlabel('Average Annual Salary ($)', fontsize=12)
    plt.ylabel('Gym Location', fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_salary_by_location.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/02_salary_by_location.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 3: Compa-Ratio Distribution (Enhanced)
    # -------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Calculate key metrics
    below_market = len(employee_df[employee_df['compa_ratio'] < 0.85])
    at_market = len(employee_df[(employee_df['compa_ratio'] >= 0.85) & (employee_df['compa_ratio'] <= 1.15)])
    above_market = len(employee_df[employee_df['compa_ratio'] > 1.15])
    avg_compa = employee_df['compa_ratio'].mean()
    median_compa = employee_df['compa_ratio'].median()
    
    # Create histogram with better binning
    n, bins, patches = ax.hist(employee_df['compa_ratio'], bins=25, color='#5DADE2', 
                                edgecolor='white', alpha=0.85, linewidth=1.2)
    
    # Color bars based on market position
    for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
        if bin_left < 0.85:
            patch.set_facecolor('#E74C3C')  # Red - below market
        elif bin_left > 1.15:
            patch.set_facecolor('#F39C12')  # Orange - above market
        else:
            patch.set_facecolor('#27AE60')  # Green - at market
    
    # Add vertical reference lines
    ax.axvline(x=0.85, color='#E74C3C', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(x=1.0, color='#2C3E50', linestyle='-', linewidth=3, alpha=0.9)
    ax.axvline(x=1.15, color='#F39C12', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(x=avg_compa, color='#9B59B6', linestyle=':', linewidth=2.5, alpha=0.9)
    
    # Add shaded regions
    ax.axvspan(0, 0.85, alpha=0.08, color='#E74C3C')
    ax.axvspan(1.15, 2, alpha=0.08, color='#F39C12')
    ax.axvspan(0.85, 1.15, alpha=0.08, color='#27AE60')
    
    # Title and labels
    ax.set_title('Compa-Ratio Distribution\nHow Our Pay Compares to Market', 
                 fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('Compa-Ratio (Employee Salary ÷ Market Midpoint)', fontsize=13, labelpad=10)
    ax.set_ylabel('Number of Employees', fontsize=13, labelpad=10)
    
    # Add text boxes with key insights
    textbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7', alpha=0.95)
    
    # Summary stats box (top right)
    stats_text = f'Summary Statistics\n' \
                 f'─────────────────\n' \
                 f'Average: {avg_compa:.2f}\n' \
                 f'Median: {median_compa:.2f}\n' \
                 f'Total Employees: {len(employee_df)}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=textbox_props,
            fontfamily='monospace')
    
    # Distribution breakdown box (top left)
    breakdown_text = f'Market Position Breakdown\n' \
                     f'──────────────────────────\n' \
                     f'🔴 Below Market (<0.85): {below_market} ({below_market/len(employee_df)*100:.1f}%)\n' \
                     f'🟢 At Market (0.85-1.15): {at_market} ({at_market/len(employee_df)*100:.1f}%)\n' \
                     f'🟠 Above Market (>1.15): {above_market} ({above_market/len(employee_df)*100:.1f}%)'
    ax.text(0.02, 0.98, breakdown_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', bbox=textbox_props,
            fontfamily='monospace')
    
    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='white', label=f'Below Market (Retention Risk)'),
        Patch(facecolor='#27AE60', edgecolor='white', label=f'Competitive Range'),
        Patch(facecolor='#F39C12', edgecolor='white', label=f'Above Market'),
        Line2D([0], [0], color='#2C3E50', linewidth=3, label='Market Midpoint (1.0)'),
        Line2D([0], [0], color='#9B59B6', linewidth=2.5, linestyle=':', label=f'Our Average ({avg_compa:.2f})')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Style improvements
    ax.set_xlim(0.65, 1.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_compa_ratio_distribution.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {OUTPUT_DIR}/03_compa_ratio_distribution.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 4: Salary vs Tenure Scatter Plot
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=employee_df,
        x='tenure_years',
        y='annual_salary',
        hue='job_level',
        palette={0: '#3498db', 1: '#2ecc71', 2: '#e74c3c'},
        s=80,
        alpha=0.7
    )
    
    z = np.polyfit(employee_df['tenure_years'], employee_df['annual_salary'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(employee_df['tenure_years'].min(), employee_df['tenure_years'].max(), 100)
    plt.plot(x_line, p(x_line), 'k--', alpha=0.5, label='Trend Line')
    
    plt.title('Salary vs. Tenure by Job Level', fontsize=16, fontweight='bold')
    plt.xlabel('Years of Service', fontsize=12)
    plt.ylabel('Annual Salary ($)', fontsize=12)
    
    handles, labels = scatter.get_legend_handles_labels()
    new_labels = ['Entry Level', 'Mid Level', 'Management']
    plt.legend(handles[:3], new_labels, title='Job Level', loc='upper left')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_salary_vs_tenure.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/04_salary_vs_tenure.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 5: Headcount by Gym and Job Level (Stacked Bar)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(14, 8))
    
    level_counts = employee_df.pivot_table(
        index='location',
        columns='job_level',
        values='employee_id',
        aggfunc='count',
        fill_value=0
    )
    
    level_counts.columns = ['Entry Level', 'Mid Level', 'Management']
    level_counts = level_counts.loc[level_counts.sum(axis=1).sort_values(ascending=True).index]
    
    level_counts.plot(kind='barh', stacked=True, color=['#3498db', '#2ecc71', '#e74c3c'], ax=plt.gca())
    
    plt.title('Staff Distribution by Gym Location and Job Level', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Employees', fontsize=12)
    plt.ylabel('Gym Location', fontsize=12)
    plt.legend(title='Job Level', loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_headcount_by_gym_level.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/05_headcount_by_gym_level.png")
    plt.close()


def create_benefits_visualizations(benefits_df, employee_df):
    """
    Generate visualizations for benefits enrollment analysis.
    """
    
    print("\n" + "="*60)
    print("GENERATING BENEFITS VISUALIZATIONS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Chart 6: Medical Plan Enrollment (Pie Chart)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(10, 10))
    
    medical_counts = benefits_df['medical_plan'].value_counts()
    colors = ['#3498db', '#2ecc71', '#f39c12', '#95a5a6']
    
    plt.pie(
        medical_counts.values,
        labels=medical_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.02] * len(medical_counts),
        shadow=True,
        startangle=90
    )
    
    plt.title('Medical Plan Enrollment Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_medical_enrollment.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/06_medical_enrollment.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 7: 401(k) Contribution Distribution (Enhanced)
    # -------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Get enrolled employees and calculate metrics
    enrolled = benefits_df[benefits_df['401k_enrolled']]
    not_enrolled = benefits_df[~benefits_df['401k_enrolled']]
    
    participation_rate = len(enrolled) / len(benefits_df) * 100
    mean_contrib = enrolled['401k_contribution_pct'].mean()
    median_contrib = enrolled['401k_contribution_pct'].median()
    
    # Count by contribution tier
    tier_1_5 = len(enrolled[enrolled['401k_contribution_pct'] <= 5])
    tier_6_10 = len(enrolled[(enrolled['401k_contribution_pct'] > 5) & (enrolled['401k_contribution_pct'] <= 10)])
    tier_11_plus = len(enrolled[enrolled['401k_contribution_pct'] > 10])
    
    # Create histogram
    n, bins, patches = ax.hist(enrolled['401k_contribution_pct'], bins=range(0, 18, 1), 
                                color='#27AE60', edgecolor='white', alpha=0.85, linewidth=1.2)
    
    # Color bars by tier
    for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
        if bin_left < 6:
            patch.set_facecolor('#F39C12')  # Orange - low contribution
        elif bin_left < 10:
            patch.set_facecolor('#27AE60')  # Green - good contribution
        else:
            patch.set_facecolor('#3498DB')  # Blue - excellent contribution
    
    # Add reference lines
    ax.axvline(x=mean_contrib, color='#E74C3C', linestyle='-', linewidth=3, alpha=0.9)
    ax.axvline(x=6, color='#7F8C8D', linestyle='--', linewidth=2, alpha=0.7)  # Common match threshold
    ax.axvline(x=10, color='#7F8C8D', linestyle='--', linewidth=2, alpha=0.7)  # High saver threshold
    
    # Title and labels
    ax.set_title('401(k) Contribution Rate Distribution\nEmployee Retirement Savings Analysis', 
                 fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('Contribution Rate (%)', fontsize=13, labelpad=10)
    ax.set_ylabel('Number of Employees', fontsize=13, labelpad=10)
    
    # Add text boxes with key insights
    textbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7', alpha=0.95)
    
    # Participation stats box (top right)
    stats_text = f'Participation Summary\n' \
                 f'─────────────────────\n' \
                 f'Enrolled: {len(enrolled)} ({participation_rate:.1f}%)\n' \
                 f'Not Enrolled: {len(not_enrolled)} ({100-participation_rate:.1f}%)\n' \
                 f'Avg Contribution: {mean_contrib:.1f}%\n' \
                 f'Median Contribution: {median_contrib:.1f}%'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=textbox_props,
            fontfamily='monospace')
    
    # Contribution tier breakdown (top left)
    tier_text = f'Contribution Tier Breakdown\n' \
                f'───────────────────────────\n' \
                f'🟠 1-5% (Building): {tier_1_5} ({tier_1_5/len(enrolled)*100:.1f}%)\n' \
                f'🟢 6-10% (On Track): {tier_6_10} ({tier_6_10/len(enrolled)*100:.1f}%)\n' \
                f'🔵 11%+ (Maximizing): {tier_11_plus} ({tier_11_plus/len(enrolled)*100:.1f}%)'
    ax.text(0.02, 0.98, tier_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', bbox=textbox_props,
            fontfamily='monospace')
    
    # Opportunity callout box (bottom center)
    opportunity_text = f'💡 Opportunity: {tier_1_5} employees at 1-5% could benefit from\n' \
                       f'    education on increasing contributions to maximize employer match'
    ax.text(0.5, 0.02, opportunity_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', edgecolor='#F4D03F', alpha=0.95))
    
    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#F39C12', edgecolor='white', label='1-5% (Building Savings)'),
        Patch(facecolor='#27AE60', edgecolor='white', label='6-10% (On Track)'),
        Patch(facecolor='#3498DB', edgecolor='white', label='11%+ (Maximizing)'),
        Line2D([0], [0], color='#E74C3C', linewidth=3, label=f'Average ({mean_contrib:.1f}%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Style improvements
    ax.set_xlim(0, 17)
    ax.set_xticks(range(0, 17, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_401k_contributions.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {OUTPUT_DIR}/07_401k_contributions.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 8: Benefits Cost by Department
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(14, 8))
    
    merged = benefits_df.merge(employee_df[['employee_id', 'department']], on='employee_id')
    dept_costs = merged.groupby('department')['annual_benefits_cost'].mean().sort_values()
    
    bars = plt.bar(range(len(dept_costs)), dept_costs.values, color=sns.color_palette('Blues_r', len(dept_costs)))
    
    for bar, val in zip(bars, dept_costs.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'${val:,.0f}', ha='center', fontsize=10)
    
    plt.title('Average Annual Benefits Cost by Department', fontsize=16, fontweight='bold')
    plt.xlabel('Department', fontsize=12)
    plt.ylabel('Average Annual Cost ($)', fontsize=12)
    plt.xticks(range(len(dept_costs)), dept_costs.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_benefits_cost_by_dept.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/08_benefits_cost_by_dept.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 9: Benefits Enrollment Summary
    # -------------------------------------------------------------------------
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Dental enrollment
    dental_counts = benefits_df['dental_plan'].value_counts()
    axes[0].pie(dental_counts.values, labels=dental_counts.index, autopct='%1.1f%%',
                colors=['#3498db', '#2ecc71', '#95a5a6'])
    axes[0].set_title('Dental Plan Enrollment', fontsize=14, fontweight='bold')
    
    # Vision enrollment
    vision_counts = benefits_df['vision_plan'].value_counts()
    axes[1].pie(vision_counts.values, labels=vision_counts.index, autopct='%1.1f%%',
                colors=['#9b59b6', '#1abc9c', '#95a5a6'])
    axes[1].set_title('Vision Plan Enrollment', fontsize=14, fontweight='bold')
    
    # Life insurance distribution
    life_counts = benefits_df['life_insurance_multiple'].value_counts().sort_index()
    axes[2].bar(life_counts.index.astype(str) + 'x', life_counts.values, color='#e74c3c')
    axes[2].set_title('Life Insurance Coverage', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Coverage Multiple')
    axes[2].set_ylabel('Employees')
    
    plt.suptitle('Benefits Enrollment Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_benefits_overview.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/09_benefits_overview.png")
    plt.close()


def create_leave_visualizations(leave_df, employee_df):
    """
    Generate visualizations for leave and PTO analysis.
    """
    
    print("\n" + "="*60)
    print("GENERATING LEAVE VISUALIZATIONS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Chart 10: PTO Utilization by Gym Location (Enhanced)
    # -------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    leave_df['pto_utilization'] = leave_df['pto_used_hours'] / leave_df['pto_accrual_hours'] * 100
    merged = leave_df.merge(employee_df[['employee_id', 'location', 'annual_salary']], on='employee_id')
    
    # Calculate metrics by gym
    gym_stats = merged.groupby('location').agg({
        'pto_utilization': 'mean',
        'pto_balance_hours': 'sum',
        'annual_salary': 'mean'
    }).round(2)
    
    # Calculate PTO liability per gym
    merged['hourly_rate'] = merged['annual_salary'] / 2080
    merged['pto_liability'] = merged['pto_balance_hours'] * merged['hourly_rate']
    gym_liability = merged.groupby('location')['pto_liability'].sum()
    gym_stats['pto_liability'] = gym_liability
    
    # Sort by utilization
    gym_stats = gym_stats.sort_values('pto_utilization')
    
    # Calculate overall metrics
    total_liability = gym_stats['pto_liability'].sum()
    avg_utilization = gym_stats['pto_utilization'].mean()
    total_unused_hours = gym_stats['pto_balance_hours'].sum()
    
    # Create horizontal bar chart
    colors = ['#E74C3C' if x < 70 else '#F39C12' if x < 80 else '#27AE60' for x in gym_stats['pto_utilization']]
    bars = ax.barh(range(len(gym_stats)), gym_stats['pto_utilization'], color=colors, 
                   edgecolor='white', linewidth=1.5, height=0.7)
    
    # Add reference lines and shading
    ax.axvline(x=70, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=80, color='#27AE60', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(0, 70, alpha=0.05, color='#E74C3C')
    ax.axvspan(70, 80, alpha=0.05, color='#F39C12')
    ax.axvspan(80, 100, alpha=0.05, color='#27AE60')
    
    # Add data labels with utilization % and liability
    for i, (bar, (gym, row)) in enumerate(zip(bars, gym_stats.iterrows())):
        util_val = row['pto_utilization']
        liability_val = row['pto_liability']
        
        # Utilization label on bar
        ax.text(util_val + 1.5, bar.get_y() + bar.get_height()/2, 
                f'{util_val:.1f}%', va='center', fontsize=12, fontweight='bold')
        
        # Liability label at end
        ax.text(98, bar.get_y() + bar.get_height()/2, 
                f'${liability_val:,.0f}', va='center', ha='right', fontsize=10, 
                color='#7F8C8D', style='italic')
    
    # Set y-axis labels (gym names)
    gym_labels = [name.replace('Movement ', '') for name in gym_stats.index]
    ax.set_yticks(range(len(gym_stats)))
    ax.set_yticklabels(gym_labels, fontsize=12)
    
    # Title and labels
    ax.set_title('PTO Utilization by Gym Location\nIdentifying Time-Off Patterns & Financial Liability', 
                 fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('PTO Utilization Rate (%)', fontsize=13, labelpad=10)
    ax.set_ylabel('Gym Location', fontsize=13, labelpad=10)
    
    # Add text boxes with key insights
    textbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7', alpha=0.95)
    
    # Summary stats box (top right)
    stats_text = f'Organization Summary\n' \
                 f'─────────────────────\n' \
                 f'Avg Utilization: {avg_utilization:.1f}%\n' \
                 f'Unused PTO Hours: {total_unused_hours:,.0f}\n' \
                 f'Total PTO Liability: ${total_liability:,.2f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=textbox_props,
            fontfamily='monospace')
    
    # Status legend box (top left)
    status_text = f'Utilization Status Guide\n' \
                  f'────────────────────────\n' \
                  f'🔴 Below 70%: Concern (burnout risk)\n' \
                  f'🟠 70-80%: Monitor (encourage PTO)\n' \
                  f'🟢 Above 80%: Healthy utilization'
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', bbox=textbox_props,
            fontfamily='monospace')
    
    # Find lowest utilization gym for callout
    lowest_gym = gym_stats['pto_utilization'].idxmin().replace('Movement ', '')
    lowest_util = gym_stats['pto_utilization'].min()
    lowest_liability = gym_stats.loc[gym_stats['pto_utilization'].idxmin(), 'pto_liability']
    
    # Action item callout
    action_text = f'⚠️ Action Item: {lowest_gym} has lowest utilization ({lowest_util:.1f}%)\n' \
                  f'    with ${lowest_liability:,.0f} in PTO liability. Consider encouraging time off.'
    ax.text(0.5, 0.02, action_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEDEC', edgecolor='#E74C3C', alpha=0.95))
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='white', label='Below 70% (Concern)'),
        Patch(facecolor='#F39C12', edgecolor='white', label='70-80% (Monitor)'),
        Patch(facecolor='#27AE60', edgecolor='white', label='Above 80% (Healthy)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Style improvements
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_pto_utilization.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {OUTPUT_DIR}/10_pto_utilization.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 11: Leave Balance Distribution
    # -------------------------------------------------------------------------
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PTO balance histogram
    axes[0].hist(leave_df['pto_balance_hours'], bins=15, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=leave_df['pto_balance_hours'].mean(), color='red', linestyle='--',
                    label=f"Mean: {leave_df['pto_balance_hours'].mean():.0f} hrs")
    axes[0].set_title('PTO Balance Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Hours Remaining')
    axes[0].set_ylabel('Number of Employees')
    axes[0].legend()
    
    # Sick leave balance histogram
    axes[1].hist(leave_df['sick_balance_hours'], bins=15, color='#e74c3c', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=leave_df['sick_balance_hours'].mean(), color='blue', linestyle='--',
                    label=f"Mean: {leave_df['sick_balance_hours'].mean():.0f} hrs")
    axes[1].set_title('Sick Leave Balance Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Hours Remaining')
    axes[1].set_ylabel('Number of Employees')
    axes[1].legend()
    
    plt.suptitle('Leave Balance Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_leave_balances.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/11_leave_balances.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 12: Extended Leave Summary
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(10, 8))
    
    extended = leave_df[leave_df['extended_leave_type'] != 'None']
    
    if len(extended) > 0:
        leave_types = extended['extended_leave_type'].value_counts()
        
        plt.pie(leave_types.values, labels=leave_types.index, autopct='%1.1f%%',
                colors=['#3498db', '#e74c3c', '#27ae60'], 
                wedgeprops=dict(width=0.6))
        
        plt.text(0, 0, f'{len(extended)}\nEmployees', ha='center', va='center', fontsize=16, fontweight='bold')
        plt.title('Extended Leave by Type', fontsize=16, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Extended Leaves', ha='center', va='center', fontsize=16)
        plt.title('Extended Leave Summary', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/12_extended_leave.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/12_extended_leave.png")
    plt.close()


def create_executive_summary(employee_df, benefits_df, leave_df):
    """
    Generate an executive summary dashboard with key metrics.
    """
    
    print("\n" + "="*60)
    print("GENERATING EXECUTIVE SUMMARY DASHBOARD")
    print("="*60)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Row 1: Key Metrics Cards
    # -------------------------------------------------------------------------
    
    total_employees = len(employee_df)
    avg_salary = employee_df['annual_salary'].mean()
    avg_tenure = employee_df['tenure_years'].mean()
    benefits_participation = (benefits_df['medical_plan'] != 'Waived').mean() * 100
    
    metrics = [
        ('Total Employees', f'{total_employees:,}', '#3498db'),
        ('Avg Salary', f'${avg_salary:,.0f}', '#27ae60'),
        ('Avg Tenure', f'{avg_tenure:.1f} years', '#9b59b6'),
        ('Benefits Enrollment', f'{benefits_participation:.0f}%', '#e74c3c')
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.15))
        ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=28, fontweight='bold', color=color)
        ax.text(0.5, 0.3, label, ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    # -------------------------------------------------------------------------
    # Row 2: Key Charts
    # -------------------------------------------------------------------------
    
    # Salary by Department
    ax1 = fig.add_subplot(gs[1, :2])
    dept_salary = employee_df.groupby('department')['annual_salary'].mean().sort_values()
    ax1.barh(dept_salary.index, dept_salary.values, color='#3498db')
    ax1.set_title('Average Salary by Department', fontweight='bold')
    ax1.set_xlabel('Annual Salary ($)')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Headcount by Location
    ax2 = fig.add_subplot(gs[1, 2:])
    location_counts = employee_df['location'].value_counts()
    ax2.pie(location_counts.values, labels=[loc.replace('Movement ', '') for loc in location_counts.index],
            autopct='%1.0f%%', colors=sns.color_palette('husl', len(location_counts)))
    ax2.set_title('Headcount by Gym', fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Row 3: Additional Insights
    # -------------------------------------------------------------------------
    
    # Compa-ratio distribution
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.hist(employee_df['compa_ratio'], bins=15, color='#27ae60', edgecolor='white', alpha=0.8)
    ax3.axvline(x=1.0, color='red', linestyle='--', label='Market Midpoint')
    ax3.set_title('Compa-Ratio Distribution', fontweight='bold')
    ax3.set_xlabel('Compa-Ratio')
    ax3.set_ylabel('Employees')
    ax3.legend()
    
    # 401k participation by contribution level
    ax4 = fig.add_subplot(gs[2, 2:])
    enrolled = benefits_df[benefits_df['401k_enrolled']]
    contrib_bins = pd.cut(enrolled['401k_contribution_pct'], bins=[0, 5, 10, 15, 20], labels=['1-5%', '6-10%', '11-15%', '16%+'])
    contrib_counts = contrib_bins.value_counts().sort_index()
    ax4.bar(contrib_counts.index, contrib_counts.values, color='#9b59b6')
    ax4.set_title('401(k) Contribution Rates', fontweight='bold')
    ax4.set_xlabel('Contribution Rate')
    ax4.set_ylabel('Employees')
    
    fig.suptitle('Movement Colorado - HR Benefits & Compensation Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/00_executive_dashboard.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/00_executive_dashboard.png")
    plt.close()


# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def export_data(employee_df, benefits_df, leave_df):
    """
    Export generated data to CSV files for further analysis.
    """
    
    print("\n" + "="*60)
    print("EXPORTING DATA FILES")
    print("="*60)
    
    data_dir = 'output/data'
    os.makedirs(data_dir, exist_ok=True)
    
    employee_df.to_csv(f'{data_dir}/employee_data.csv', index=False)
    print(f"  Saved: {data_dir}/employee_data.csv")
    
    benefits_df.to_csv(f'{data_dir}/benefits_enrollment.csv', index=False)
    print(f"  Saved: {data_dir}/benefits_enrollment.csv")
    
    leave_df.to_csv(f'{data_dir}/leave_tracking.csv', index=False)
    print(f"  Saved: {data_dir}/leave_tracking.csv")
    
    combined = employee_df.merge(benefits_df, on='employee_id')
    combined = combined.merge(leave_df, on='employee_id')
    
    combined.to_csv(f'{data_dir}/combined_hr_data.csv', index=False)
    print(f"  Saved: {data_dir}/combined_hr_data.csv")
    
    print(f"\n  Total records exported: {len(combined)}")


# ============================================================================
# ADD THIS FUNCTION TO main.py (before the main() function)
# ============================================================================

def update_readme_insights(comp_results, benefits_results, leave_results):
    """
    Update the README.md file by replacing {{template_variables}} with 
    actual computed values from the current data run.
    """
    
    print("\n" + "="*60)
    print("UPDATING README WITH DYNAMIC INSIGHTS")
    print("="*60)
    
    readme_path = 'README.md'
    
    if not os.path.exists(readme_path):
        print(f"  Warning: {readme_path} not found. Skipping README update.")
        return
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Calculate values for template variables
    below_market_count = comp_results['below_market_count']
    avg_compa_ratio = f"{comp_results['avg_compa_ratio']:.2f}"
    
    if comp_results['avg_compa_ratio'] < 0.95:
        market_assessment = "we're paying slightly below market rate"
    elif comp_results['avg_compa_ratio'] > 1.05:
        market_assessment = "we're paying above market rate"
    else:
        market_assessment = "we're paying competitively at market rate"
    
    participation_rate = f"{benefits_results['participation_rate']:.1f}"
    avg_contribution = f"{benefits_results['avg_contribution']:.1f}"
    total_annual_cost = f"{benefits_results['total_annual_cost']:,.0f}"
    employee_count = benefits_results['employee_count']
    
    total_pto_liability = f"{leave_results['total_liability']:,.2f}"
    lowest_util_gym = leave_results['lowest_util_gym'].replace('Movement ', '')
    lowest_util_pct = f"{leave_results['lowest_util_pct']:.0f}"
    
    # Replace all {{template_variables}} with actual values
    replacements = {
        '{{below_market_count}}': str(below_market_count),
        '{{avg_compa_ratio}}': avg_compa_ratio,
        '{{market_assessment}}': market_assessment,
        '{{participation_rate}}': participation_rate,
        '{{avg_contribution}}': avg_contribution,
        '{{total_annual_cost}}': total_annual_cost,
        '{{employee_count}}': str(employee_count),
        '{{total_pto_liability}}': total_pto_liability,
        '{{lowest_util_gym}}': lowest_util_gym,
        '{{lowest_util_pct}}': lowest_util_pct,
    }
    
    for template_var, value in replacements.items():
        content = content.replace(template_var, value)
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  Updated: {readme_path}")
    print(f"\n  Replaced template variables:")
    for template_var, value in replacements.items():
        print(f"    {template_var} → {value}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the complete HR analysis pipeline.
    """
    
    print("\n" + "="*70)
    print("   MOVEMENT COLORADO - HR BENEFITS & COMPENSATION DASHBOARD")
    print("   Portfolio Project for Benefits Administrator Role")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Generate Data
    # -------------------------------------------------------------------------
    
    print("\n[STEP 1] Generating synthetic HR data for Movement Colorado...")
    
    employee_df = generate_employee_data()
    print(f"  Generated {len(employee_df)} employee records across 6 gyms")
    
    benefits_df = generate_benefits_data(employee_df)
    print(f"  Generated benefits enrollment data")
    
    leave_df = generate_leave_data(employee_df)
    print(f"  Generated leave tracking data")
    
    # -------------------------------------------------------------------------
    # Step 2: Run Analysis
    # -------------------------------------------------------------------------
    
    print("\n[STEP 2] Running analysis...")
    
    comp_results = analyze_compensation(employee_df)
    benefits_results = analyze_benefits(benefits_df, employee_df)
    leave_results = analyze_leave(leave_df, employee_df)
    
    # -------------------------------------------------------------------------
    # Step 3: Generate Visualizations
    # -------------------------------------------------------------------------
    
    print("\n[STEP 3] Generating visualizations...")
    
    create_compensation_visualizations(employee_df, comp_results)
    create_benefits_visualizations(benefits_df, employee_df)
    create_leave_visualizations(leave_df, employee_df)
    create_executive_summary(employee_df, benefits_df, leave_df)
    
    # -------------------------------------------------------------------------
    # Step 4: Export Data
    # -------------------------------------------------------------------------
    
    print("\n[STEP 4] Exporting data files...")
    export_data(employee_df, benefits_df, leave_df)

    print("\n[STEP 5] Updating README with current statistics...")
    update_readme_insights(comp_results, benefits_results, leave_results)
    
    # -------------------------------------------------------------------------
    # Completion Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "="*70)
    print("   ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n  Charts saved to: {OUTPUT_DIR}/")
    print(f"  Data files saved to: output/data/")
    print("\n  Files generated:")
    print("    - 00_executive_dashboard.png (Summary Dashboard)")
    print("    - 01_salary_by_department.png")
    print("    - 02_salary_by_location.png")
    print("    - 03_compa_ratio_distribution.png")
    print("    - 04_salary_vs_tenure.png")
    print("    - 05_headcount_by_gym_level.png")
    print("    - 06_medical_enrollment.png")
    print("    - 07_401k_contributions.png")
    print("    - 08_benefits_cost_by_dept.png")
    print("    - 09_benefits_overview.png")
    print("    - 10_pto_utilization.png")
    print("    - 11_leave_balances.png")
    print("    - 12_extended_leave.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
