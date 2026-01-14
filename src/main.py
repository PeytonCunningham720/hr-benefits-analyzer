"""
HR Benefits & Compensation Analysis Dashboard
==============================================
A Python project demonstrating data analysis skills for HR/Benefits Administration.
This tool analyzes compensation data, benefits enrollment, and leave tracking.

Author: [Your Name]
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
# DATA GENERATION - Creating realistic synthetic HR data for analysis
# ============================================================================

def generate_employee_data(num_employees=150):
    """
    Generate synthetic employee data for compensation analysis.
    
    This function creates realistic fake data that mimics what you'd find
    in an actual HRIS (Human Resource Information System) like Dayforce.
    
    Parameters:
    -----------
    num_employees : int
        The number of employee records to generate (default: 150)
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing employee demographic and compensation data
    """
    
    # Set a random seed for reproducibility
    # This ensures the same "random" data is generated each time
    np.random.seed(42)
    
    # Define the departments that exist in a climbing gym organization
    # These align with Movement's actual organizational structure
    departments = [
        'Operations',      # Gym floor staff, front desk
        'Fitness',         # Personal trainers, yoga instructors
        'Climbing',        # Route setters, climbing instructors
        'Marketing',       # Brand, social media, communications
        'Finance',         # Accounting, payroll
        'HR',              # Human resources, recruiting
        'Facilities',      # Maintenance, cleaning
        'Management'       # General managers, directors
    ]
    
    # Define job titles mapped to each department
    # This creates realistic job hierarchies within each department
    job_titles = {
        'Operations': ['Front Desk Associate', 'Operations Coordinator', 'Operations Manager'],
        'Fitness': ['Fitness Instructor', 'Personal Trainer', 'Fitness Director'],
        'Climbing': ['Route Setter', 'Climbing Instructor', 'Head Route Setter'],
        'Marketing': ['Marketing Coordinator', 'Content Creator', 'Marketing Manager'],
        'Finance': ['Accounting Clerk', 'Staff Accountant', 'Finance Manager'],
        'HR': ['HR Coordinator', 'HR Generalist', 'HR Manager'],
        'Facilities': ['Maintenance Technician', 'Facilities Coordinator', 'Facilities Manager'],
        'Management': ['Assistant Manager', 'General Manager', 'Regional Director']
    }
    
    # Define gym locations matching Movement's actual geographic footprint
    # These are the regions mentioned in the job posting
    locations = [
        'Portland, OR',
        'San Francisco, CA',
        'Los Angeles, CA',
        'Denver, CO',
        'Boulder, CO',
        'Dallas, TX',
        'Chicago, IL',
        'New York, NY',
        'Philadelphia, PA',
        'Washington, DC'
    ]
    
    # Define regional cost of living multipliers
    # Higher values = higher cost of living = higher pay adjustments
    # These affect compensation benchmarking calculations
    location_multipliers = {
        'Portland, OR': 1.05,       # Moderate cost of living
        'San Francisco, CA': 1.35,  # Very high cost of living
        'Los Angeles, CA': 1.25,    # High cost of living
        'Denver, CO': 1.10,         # Moderate-high cost of living
        'Boulder, CO': 1.15,        # Higher due to desirability
        'Dallas, TX': 0.95,         # Lower cost of living
        'Chicago, IL': 1.10,        # Moderate-high cost of living
        'New York, NY': 1.40,       # Highest cost of living
        'Philadelphia, PA': 1.05,   # Moderate cost of living
        'Washington, DC': 1.30      # High cost of living
    }
    
    # Define base salary ranges for each job level (entry, mid, senior)
    # These represent market-competitive ranges before location adjustments
    base_salary_ranges = {
        0: (35000, 45000),   # Entry level positions (index 0 in job_titles lists)
        1: (45000, 60000),   # Mid-level positions (index 1 in job_titles lists)
        2: (60000, 85000)    # Senior/Manager positions (index 2 in job_titles lists)
    }
    
    # Initialize empty lists to store generated data for each column
    # We'll populate these lists and then combine them into a DataFrame
    employee_ids = []        # Unique identifier for each employee
    employee_names = []      # Employee names (randomly generated)
    dept_list = []           # Department assignment
    title_list = []          # Job title
    location_list = []       # Work location
    hire_dates = []          # Date of hire
    salaries = []            # Annual salary
    job_levels = []          # Job level (0=entry, 1=mid, 2=senior)
    
    # Generate data for each employee
    for i in range(num_employees):
        
        # Create a unique employee ID with prefix 'EMP' and 4-digit number
        # Example: EMP0001, EMP0002, etc.
        employee_ids.append(f'EMP{i+1:04d}')
        
        # Generate a placeholder name
        # In a real project, you might use a library like 'faker' for realistic names
        employee_names.append(f'Employee {i+1}')
        
        # Randomly select a department for this employee
        # np.random.choice picks one item randomly from the list
        dept = np.random.choice(departments)
        dept_list.append(dept)
        
        # Determine job level based on weighted probabilities
        # 50% entry level, 35% mid-level, 15% senior - realistic org pyramid
        level = np.random.choice([0, 1, 2], p=[0.50, 0.35, 0.15])
        job_levels.append(level)
        
        # Get the appropriate job title based on department and level
        title_list.append(job_titles[dept][level])
        
        # Randomly assign a location
        location = np.random.choice(locations)
        location_list.append(location)
        
        # Generate a random hire date within the last 8 years
        # This creates realistic tenure distribution
        days_employed = np.random.randint(30, 8*365)  # 30 days to 8 years
        hire_date = datetime.now() - timedelta(days=days_employed)
        hire_dates.append(hire_date.strftime('%Y-%m-%d'))
        
        # Calculate salary based on job level and location
        # Get the base range for this job level
        min_sal, max_sal = base_salary_ranges[level]
        
        # Generate a random base salary within the range
        base_salary = np.random.uniform(min_sal, max_sal)
        
        # Apply location cost of living multiplier
        adjusted_salary = base_salary * location_multipliers[location]
        
        # Round to nearest $500 for realism
        final_salary = round(adjusted_salary / 500) * 500
        salaries.append(final_salary)
    
    # Create a DataFrame by combining all the generated lists
    # Each list becomes a column in the DataFrame
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
    # This enables date-based calculations and filtering
    df['hire_date'] = pd.to_datetime(df['hire_date'])
    
    # Calculate tenure in years for each employee
    # (today's date - hire date) gives timedelta, .days converts to days
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
        # This is a key metric for benefits administrators
        enrolled = np.random.choice([True, False], p=[0.75, 0.25])
        retirement_enrolled.append(enrolled)
        
        # 401(k) contribution rate (if enrolled)
        # Distribution based on typical employee behavior
        if enrolled:
            # Most contribute 3-10%, some go higher to max out
            contrib = np.random.choice(
                [3, 4, 5, 6, 7, 8, 10, 12, 15],
                p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
            )
        else:
            contrib = 0
        retirement_contrib.append(contrib)
        
        # Life insurance selection (as multiple of salary)
        # Most employers offer 1x base, employees can buy more
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
    # map() applies the dictionary lookup to each value in the column
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
    
    # Initialize lists for leave data
    leave_records = []
    
    # Define PTO accrual rates based on tenure (hours per year)
    # This is a common tiered accrual structure
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
        # Employees typically use 60-95% of their PTO
        usage_rate = np.random.uniform(0.60, 0.95)
        pto_used = int(annual_accrual * usage_rate)
        
        # Calculate remaining PTO balance
        pto_balance = annual_accrual - pto_used
        
        # Generate sick leave data (typically fixed accrual)
        # Many companies offer 40-80 hours sick leave
        sick_accrual = 64  # 8 days per year
        sick_used = np.random.randint(0, 48)  # 0-6 days used
        sick_balance = sick_accrual - sick_used
        
        # Track any extended leaves (FMLA, disability, etc.)
        # Small percentage of employees have extended leave
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
    
    # Group employees by department and calculate aggregate statistics
    # agg() allows multiple aggregation functions on the same column
    dept_stats = employee_df.groupby('department')['annual_salary'].agg([
        ('count', 'count'),           # Number of employees
        ('min', 'min'),               # Minimum salary
        ('max', 'max'),               # Maximum salary
        ('mean', 'mean'),             # Average salary
        ('median', 'median'),         # Median salary (50th percentile)
        ('std', 'std')                # Standard deviation (salary spread)
    ]).round(2)
    
    # Print the department statistics
    print("\n--- Salary Statistics by Department ---")
    print(dept_stats.to_string())
    
    # -------------------------------------------------------------------------
    # Calculate salary statistics by location
    # -------------------------------------------------------------------------
    
    # Similar grouping but by geographic location
    location_stats = employee_df.groupby('location')['annual_salary'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median')
    ]).round(2)
    
    print("\n--- Salary Statistics by Location ---")
    print(location_stats.to_string())
    
    # -------------------------------------------------------------------------
    # Calculate compa-ratios
    # -------------------------------------------------------------------------
    
    # Compa-ratio = Employee Salary / Market Midpoint for their role
    # A compa-ratio of 1.0 means the employee is paid at market midpoint
    # < 1.0 means below market, > 1.0 means above market
    
    # Define market midpoints by job level (these would come from salary surveys)
    market_midpoints = {
        0: 42000,   # Entry level market midpoint
        1: 55000,   # Mid-level market midpoint  
        2: 75000    # Senior level market midpoint
    }
    
    # Calculate compa-ratio for each employee
    # map() looks up the market midpoint based on job level
    employee_df['market_midpoint'] = employee_df['job_level'].map(market_midpoints)
    employee_df['compa_ratio'] = employee_df['annual_salary'] / employee_df['market_midpoint']
    
    # Round compa-ratio to 2 decimal places
    employee_df['compa_ratio'] = employee_df['compa_ratio'].round(2)
    
    # Identify employees significantly below or above market
    # Below 0.85 = underpaid, Above 1.15 = overpaid (by market standards)
    below_market = employee_df[employee_df['compa_ratio'] < 0.85]
    above_market = employee_df[employee_df['compa_ratio'] > 1.15]
    
    print(f"\n--- Compa-Ratio Analysis ---")
    print(f"Employees below market (compa-ratio < 0.85): {len(below_market)}")
    print(f"Employees above market (compa-ratio > 1.15): {len(above_market)}")
    print(f"Average compa-ratio: {employee_df['compa_ratio'].mean():.2f}")
    
    # -------------------------------------------------------------------------
    # Pay range analysis by job title
    # -------------------------------------------------------------------------
    
    # Calculate pay ranges for each job title
    title_ranges = employee_df.groupby('job_title')['annual_salary'].agg([
        ('count', 'count'),
        ('min', 'min'),
        ('max', 'max'),
        ('range', lambda x: x.max() - x.min())  # Salary spread within title
    ]).round(2)
    
    # Sort by employee count descending
    title_ranges = title_ranges.sort_values('count', ascending=False)
    
    print("\n--- Pay Ranges by Job Title ---")
    print(title_ranges.head(10).to_string())
    
    # Return all analysis results as a dictionary
    return {
        'department_stats': dept_stats,
        'location_stats': location_stats,
        'below_market': below_market,
        'above_market': above_market,
        'title_ranges': title_ranges
    }


def analyze_benefits(benefits_df, employee_df):
    """
    Analyze benefits enrollment patterns and costs.
    
    This function provides insights into:
    - Plan enrollment distribution
    - Cost analysis by plan type
    - 401(k) participation rates
    
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
    
    # Print section header
    print("\n" + "="*60)
    print("BENEFITS ENROLLMENT ANALYSIS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Medical plan enrollment distribution
    # -------------------------------------------------------------------------
    
    # Count how many employees are enrolled in each medical plan
    # value_counts() returns a Series with counts for each unique value
    medical_enrollment = benefits_df['medical_plan'].value_counts()
    
    # Calculate percentages
    medical_pct = (medical_enrollment / len(benefits_df) * 100).round(1)
    
    print("\n--- Medical Plan Enrollment ---")
    for plan, count in medical_enrollment.items():
        print(f"  {plan}: {count} employees ({medical_pct[plan]}%)")
    
    # -------------------------------------------------------------------------
    # 401(k) analysis
    # -------------------------------------------------------------------------
    
    # Calculate 401(k) participation rate
    participation_rate = benefits_df['401k_enrolled'].mean() * 100
    
    # Calculate average contribution rate (among participants)
    # Filter to only enrolled employees before calculating average
    enrolled_contribs = benefits_df[benefits_df['401k_enrolled']]['401k_contribution_pct']
    avg_contribution = enrolled_contribs.mean()
    
    print(f"\n--- 401(k) Analysis ---")
    print(f"  Participation Rate: {participation_rate:.1f}%")
    print(f"  Average Contribution Rate: {avg_contribution:.1f}%")
    print(f"  Employees at 10%+ contribution: {len(enrolled_contribs[enrolled_contribs >= 10])}")
    
    # -------------------------------------------------------------------------
    # Cost analysis
    # -------------------------------------------------------------------------
    
    # Calculate total and average benefits costs
    total_annual_cost = benefits_df['annual_benefits_cost'].sum()
    avg_annual_cost = benefits_df['annual_benefits_cost'].mean()
    
    print(f"\n--- Benefits Cost Analysis ---")
    print(f"  Total Annual Benefits Cost: ${total_annual_cost:,.2f}")
    print(f"  Average Cost per Employee: ${avg_annual_cost:,.2f}")
    
    # Merge with employee data to analyze by department
    merged_df = benefits_df.merge(employee_df[['employee_id', 'department']], on='employee_id')
    
    # Calculate benefits cost by department
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
        'dept_costs': dept_costs
    }


def analyze_leave(leave_df, employee_df):
    """
    Analyze leave usage patterns and balances.
    
    This function identifies:
    - PTO utilization rates
    - Leave balance liabilities
    - Extended leave patterns
    
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
    
    # Print section header
    print("\n" + "="*60)
    print("LEAVE & PTO ANALYSIS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # PTO utilization analysis
    # -------------------------------------------------------------------------
    
    # Calculate utilization rate (hours used / hours accrued)
    leave_df['pto_utilization'] = leave_df['pto_used_hours'] / leave_df['pto_accrual_hours']
    
    # Overall utilization statistics
    avg_utilization = leave_df['pto_utilization'].mean() * 100
    total_pto_balance = leave_df['pto_balance_hours'].sum()
    
    print(f"\n--- PTO Utilization ---")
    print(f"  Average PTO Utilization Rate: {avg_utilization:.1f}%")
    print(f"  Total Unused PTO Hours: {total_pto_balance:,}")
    print(f"  Total Unused PTO Days: {total_pto_balance/8:,.0f}")
    
    # Merge with employee data to get salary for liability calculation
    merged_leave = leave_df.merge(
        employee_df[['employee_id', 'annual_salary', 'department']], 
        on='employee_id'
    )
    
    # Calculate hourly rate and PTO liability
    # PTO liability = unused hours * hourly rate
    merged_leave['hourly_rate'] = merged_leave['annual_salary'] / 2080  # 2080 = standard work hours/year
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
    
    # Filter to employees with extended leave
    extended_leaves = leave_df[leave_df['extended_leave_type'] != 'None']
    
    print(f"\n--- Extended Leave Summary ---")
    print(f"  Employees on Extended Leave: {len(extended_leaves)}")
    
    # Count by leave type
    if len(extended_leaves) > 0:
        leave_type_counts = extended_leaves['extended_leave_type'].value_counts()
        for leave_type, count in leave_type_counts.items():
            print(f"    {leave_type}: {count} employees")
    
    # PTO utilization by department
    dept_utilization = merged_leave.groupby('department').agg({
        'pto_utilization': 'mean',
        'pto_liability': 'sum'
    }).round(2)
    
    # Convert utilization to percentage for display
    dept_utilization['pto_utilization'] = (dept_utilization['pto_utilization'] * 100).round(1)
    dept_utilization.columns = ['utilization_pct', 'pto_liability']
    
    print("\n--- PTO Utilization by Department ---")
    print(dept_utilization.to_string())
    
    return {
        'avg_utilization': avg_utilization,
        'total_liability': total_liability,
        'extended_leaves': extended_leaves,
        'dept_utilization': dept_utilization
    }


# ============================================================================
# VISUALIZATION FUNCTIONS - Creating charts and graphs
# ============================================================================

def create_compensation_visualizations(employee_df, analysis_results):
    """
    Generate visualizations for compensation analysis.
    
    Creates multiple charts showing salary distributions,
    comparisons, and trends.
    
    Parameters:
    -----------
    employee_df : pandas.DataFrame
        The employee data
    analysis_results : dict
        Results from analyze_compensation()
    """
    
    print("\n" + "="*60)
    print("GENERATING COMPENSATION VISUALIZATIONS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Chart 1: Salary Distribution by Department (Box Plot)
    # -------------------------------------------------------------------------
    
    # Create a new figure with specified size
    plt.figure(figsize=(14, 8))
    
    # Create box plot showing salary distribution for each department
    # Box plots show: median, quartiles, and outliers
    # order parameter sorts departments by median salary
    dept_order = employee_df.groupby('department')['annual_salary'].median().sort_values(ascending=False).index
    
    # Create the box plot using seaborn
    sns.boxplot(
        data=employee_df,
        x='department',
        y='annual_salary',
        order=dept_order,
        palette='viridis'
    )
    
    # Add title and labels
    plt.title('Salary Distribution by Department', fontsize=16, fontweight='bold')
    plt.xlabel('Department', fontsize=12)
    plt.ylabel('Annual Salary ($)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis to show dollar amounts with commas
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure to the output directory
    plt.savefig(f'{OUTPUT_DIR}/01_salary_by_department.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/01_salary_by_department.png")
    
    # Close the figure to free memory
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 2: Average Salary by Location (Horizontal Bar Chart)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(12, 8))
    
    # Calculate average salary by location and sort
    location_avg = employee_df.groupby('location')['annual_salary'].mean().sort_values()
    
    # Create horizontal bar chart
    # barh() creates horizontal bars, which work better for location names
    bars = plt.barh(location_avg.index, location_avg.values, color=sns.color_palette('viridis', len(location_avg)))
    
    # Add value labels on the bars
    for bar, val in zip(bars, location_avg.values):
        plt.text(val + 500, bar.get_y() + bar.get_height()/2, 
                f'${val:,.0f}', va='center', fontsize=10)
    
    plt.title('Average Salary by Location', fontsize=16, fontweight='bold')
    plt.xlabel('Average Annual Salary ($)', fontsize=12)
    plt.ylabel('Location', fontsize=12)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_salary_by_location.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/02_salary_by_location.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 3: Compa-Ratio Distribution (Histogram)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(12, 7))
    
    # Create histogram of compa-ratios
    # bins=20 divides the data into 20 intervals
    plt.hist(employee_df['compa_ratio'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    
    # Add vertical reference lines
    # Red dashed line at 0.85 (below market threshold)
    plt.axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Below Market (0.85)')
    # Green dashed line at 1.0 (market midpoint)
    plt.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Market Midpoint (1.0)')
    # Orange dashed line at 1.15 (above market threshold)
    plt.axvline(x=1.15, color='orange', linestyle='--', linewidth=2, label='Above Market (1.15)')
    
    plt.title('Compa-Ratio Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Compa-Ratio (Salary / Market Midpoint)', fontsize=12)
    plt.ylabel('Number of Employees', fontsize=12)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_compa_ratio_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/03_compa_ratio_distribution.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 4: Salary vs Tenure Scatter Plot
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color coding by job level
    # Each point represents one employee
    scatter = sns.scatterplot(
        data=employee_df,
        x='tenure_years',
        y='annual_salary',
        hue='job_level',
        palette={0: '#3498db', 1: '#2ecc71', 2: '#e74c3c'},
        s=80,  # Point size
        alpha=0.7  # Transparency
    )
    
    # Add trend line
    # polyfit calculates the best-fit line (degree=1 for linear)
    z = np.polyfit(employee_df['tenure_years'], employee_df['annual_salary'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(employee_df['tenure_years'].min(), employee_df['tenure_years'].max(), 100)
    plt.plot(x_line, p(x_line), 'k--', alpha=0.5, label='Trend Line')
    
    plt.title('Salary vs. Tenure by Job Level', fontsize=16, fontweight='bold')
    plt.xlabel('Years of Service', fontsize=12)
    plt.ylabel('Annual Salary ($)', fontsize=12)
    
    # Update legend labels
    handles, labels = scatter.get_legend_handles_labels()
    new_labels = ['Entry Level', 'Mid Level', 'Senior Level']
    plt.legend(handles[:3], new_labels, title='Job Level', loc='upper left')
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_salary_vs_tenure.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/04_salary_vs_tenure.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 5: Employee Count by Department and Job Level (Stacked Bar)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(14, 8))
    
    # Create a pivot table for stacked bar chart
    # This counts employees by department and job level
    level_counts = employee_df.pivot_table(
        index='department',
        columns='job_level',
        values='employee_id',
        aggfunc='count',
        fill_value=0
    )
    
    # Rename columns for clarity
    level_counts.columns = ['Entry Level', 'Mid Level', 'Senior Level']
    
    # Sort by total employees
    level_counts = level_counts.loc[level_counts.sum(axis=1).sort_values(ascending=True).index]
    
    # Create stacked horizontal bar chart
    level_counts.plot(kind='barh', stacked=True, color=['#3498db', '#2ecc71', '#e74c3c'], ax=plt.gca())
    
    plt.title('Employee Distribution by Department and Job Level', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Employees', fontsize=12)
    plt.ylabel('Department', fontsize=12)
    plt.legend(title='Job Level', loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_headcount_by_dept_level.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/05_headcount_by_dept_level.png")
    plt.close()


def create_benefits_visualizations(benefits_df, employee_df):
    """
    Generate visualizations for benefits enrollment analysis.
    
    Creates charts showing enrollment distributions,
    participation rates, and cost breakdowns.
    
    Parameters:
    -----------
    benefits_df : pandas.DataFrame
        The benefits enrollment data
    employee_df : pandas.DataFrame
        The employee data
    """
    
    print("\n" + "="*60)
    print("GENERATING BENEFITS VISUALIZATIONS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Chart 6: Medical Plan Enrollment (Pie Chart)
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(10, 10))
    
    # Count enrollments for each medical plan
    medical_counts = benefits_df['medical_plan'].value_counts()
    
    # Define colors for each plan type
    colors = ['#3498db', '#2ecc71', '#f39c12', '#95a5a6']
    
    # Create pie chart
    # autopct adds percentage labels, explode slightly separates slices
    plt.pie(
        medical_counts.values,
        labels=medical_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.02] * len(medical_counts),  # Slight separation
        shadow=True,
        startangle=90
    )
    
    plt.title('Medical Plan Enrollment Distribution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_medical_enrollment.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/06_medical_enrollment.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 7: 401(k) Contribution Distribution
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(12, 7))
    
    # Filter to enrolled employees only
    enrolled = benefits_df[benefits_df['401k_enrolled']]
    
    # Create histogram of contribution rates
    plt.hist(enrolled['401k_contribution_pct'], bins=range(0, 18, 1), 
            color='#27ae60', edgecolor='white', alpha=0.8)
    
    # Add mean line
    mean_contrib = enrolled['401k_contribution_pct'].mean()
    plt.axvline(x=mean_contrib, color='red', linestyle='--', linewidth=2,
                label=f'Average: {mean_contrib:.1f}%')
    
    plt.title('401(k) Contribution Rate Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Contribution Rate (%)', fontsize=12)
    plt.ylabel('Number of Employees', fontsize=12)
    plt.legend()
    
    # Set x-axis ticks at each percentage point
    plt.xticks(range(0, 17, 2))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_401k_contributions.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/07_401k_contributions.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart 8: Benefits Cost by Department
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(14, 8))
    
    # Merge benefits with employee data
    merged = benefits_df.merge(employee_df[['employee_id', 'department']], on='employee_id')
    
    # Calculate average benefits cost by department
    dept_costs = merged.groupby('department')['annual_benefits_cost'].mean().sort_values()
    
    # Create bar chart
    bars = plt.bar(range(len(dept_costs)), dept_costs.values, color=sns.color_palette('Blues_r', len(dept_costs)))
    
    # Add value labels on bars
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
    # Chart 9: Benefits Enrollment Summary (Grouped Bar)
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
    
    Creates charts showing utilization rates,
    balances, and leave patterns.
    
    Parameters:
    -----------
    leave_df : pandas.DataFrame
        The leave tracking data
    employee_df : pandas.DataFrame
        The employee data
    """
    
    print("\n" + "="*60)
    print("GENERATING LEAVE VISUALIZATIONS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Chart 10: PTO Utilization by Department
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(14, 8))
    
    # Calculate utilization
    leave_df['pto_utilization'] = leave_df['pto_used_hours'] / leave_df['pto_accrual_hours'] * 100
    
    # Merge with employee data
    merged = leave_df.merge(employee_df[['employee_id', 'department']], on='employee_id')
    
    # Calculate average utilization by department
    dept_util = merged.groupby('department')['pto_utilization'].mean().sort_values()
    
    # Create horizontal bar chart
    colors = ['#e74c3c' if x < 70 else '#f39c12' if x < 80 else '#27ae60' for x in dept_util.values]
    bars = plt.barh(dept_util.index, dept_util.values, color=colors)
    
    # Add reference lines
    plt.axvline(x=70, color='red', linestyle='--', alpha=0.5, label='Low Utilization (70%)')
    plt.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='Target (80%)')
    
    # Add value labels
    for bar, val in zip(bars, dept_util.values):
        plt.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')
    
    plt.title('PTO Utilization Rate by Department', fontsize=16, fontweight='bold')
    plt.xlabel('Utilization Rate (%)', fontsize=12)
    plt.ylabel('Department', fontsize=12)
    plt.legend(loc='lower right')
    plt.xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_pto_utilization.png', dpi=150, bbox_inches='tight')
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
    
    # Count extended leave types
    extended = leave_df[leave_df['extended_leave_type'] != 'None']
    
    if len(extended) > 0:
        leave_types = extended['extended_leave_type'].value_counts()
        
        # Create donut chart
        plt.pie(leave_types.values, labels=leave_types.index, autopct='%1.1f%%',
                colors=['#3498db', '#e74c3c', '#27ae60'], 
                wedgeprops=dict(width=0.6))  # Width creates donut effect
        
        # Add center text
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
    
    This creates a single visualization showing the most important
    KPIs for leadership review.
    
    Parameters:
    -----------
    employee_df : pandas.DataFrame
        The employee data
    benefits_df : pandas.DataFrame
        The benefits enrollment data
    leave_df : pandas.DataFrame
        The leave tracking data
    """
    
    print("\n" + "="*60)
    print("GENERATING EXECUTIVE SUMMARY DASHBOARD")
    print("="*60)
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    
    # Define a grid layout: 3 rows, 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Row 1: Key Metrics Cards
    # -------------------------------------------------------------------------
    
    # Calculate key metrics
    total_employees = len(employee_df)
    avg_salary = employee_df['annual_salary'].mean()
    avg_tenure = employee_df['tenure_years'].mean()
    benefits_participation = (benefits_df['medical_plan'] != 'Waived').mean() * 100
    retirement_participation = benefits_df['401k_enrolled'].mean() * 100
    
    # Create metric display boxes
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
        
        # Create colored background
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.15))
        
        # Add text
        ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=28, fontweight='bold', color=color)
        ax.text(0.5, 0.3, label, ha='center', va='center', fontsize=14, color='gray')
        
        # Remove axes
        ax.axis('off')
    
    # -------------------------------------------------------------------------
    # Row 2: Key Charts
    # -------------------------------------------------------------------------
    
    # Salary by Department (mini version)
    ax1 = fig.add_subplot(gs[1, :2])
    dept_salary = employee_df.groupby('department')['annual_salary'].mean().sort_values()
    ax1.barh(dept_salary.index, dept_salary.values, color='#3498db')
    ax1.set_title('Average Salary by Department', fontweight='bold')
    ax1.set_xlabel('Annual Salary ($)')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Headcount by Location (mini version)
    ax2 = fig.add_subplot(gs[1, 2:])
    location_counts = employee_df['location'].value_counts()
    ax2.pie(location_counts.values, labels=[loc.split(',')[0] for loc in location_counts.index],
            autopct='%1.0f%%', colors=sns.color_palette('husl', len(location_counts)))
    ax2.set_title('Headcount by Location', fontweight='bold')
    
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
    
    # Add main title
    fig.suptitle('HR Benefits & Compensation Executive Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/00_executive_dashboard.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/00_executive_dashboard.png")
    plt.close()


# ============================================================================
# DATA EXPORT FUNCTIONS - Saving analysis results
# ============================================================================

def export_data(employee_df, benefits_df, leave_df):
    """
    Export generated data to CSV files for further analysis.
    
    This demonstrates ability to work with data exports
    and prepare data for other systems/tools.
    
    Parameters:
    -----------
    employee_df : pandas.DataFrame
        The employee data
    benefits_df : pandas.DataFrame
        The benefits enrollment data
    leave_df : pandas.DataFrame
        The leave tracking data
    """
    
    print("\n" + "="*60)
    print("EXPORTING DATA FILES")
    print("="*60)
    
    # Create data output directory
    data_dir = 'output/data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Export employee data
    employee_df.to_csv(f'{data_dir}/employee_data.csv', index=False)
    print(f"  Saved: {data_dir}/employee_data.csv")
    
    # Export benefits data
    benefits_df.to_csv(f'{data_dir}/benefits_enrollment.csv', index=False)
    print(f"  Saved: {data_dir}/benefits_enrollment.csv")
    
    # Export leave data
    leave_df.to_csv(f'{data_dir}/leave_tracking.csv', index=False)
    print(f"  Saved: {data_dir}/leave_tracking.csv")
    
    # Create a combined summary report
    # Merge all data into one comprehensive view
    combined = employee_df.merge(benefits_df, on='employee_id')
    combined = combined.merge(leave_df, on='employee_id')
    
    combined.to_csv(f'{data_dir}/combined_hr_data.csv', index=False)
    print(f"  Saved: {data_dir}/combined_hr_data.csv")
    
    print(f"\n  Total records exported: {len(combined)}")


# ============================================================================
# MAIN EXECUTION - Running the complete analysis pipeline
# ============================================================================

def main():
    """
    Main function to run the complete HR analysis pipeline.
    
    This orchestrates the entire analysis process:
    1. Generate synthetic data
    2. Run analytical functions
    3. Create visualizations
    4. Export results
    """
    
    # Print welcome banner
    print("\n" + "="*70)
    print("   HR BENEFITS & COMPENSATION ANALYSIS DASHBOARD")
    print("   Portfolio Project for Benefits Administrator Role")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Generate Data
    # -------------------------------------------------------------------------
    
    print("\n[STEP 1] Generating synthetic HR data...")
    
    # Generate employee data (150 employees across the organization)
    employee_df = generate_employee_data(num_employees=150)
    print(f"  Generated {len(employee_df)} employee records")
    
    # Generate benefits enrollment data
    benefits_df = generate_benefits_data(employee_df)
    print(f"  Generated benefits enrollment data")
    
    # Generate leave tracking data
    leave_df = generate_leave_data(employee_df)
    print(f"  Generated leave tracking data")
    
    # -------------------------------------------------------------------------
    # Step 2: Run Analysis
    # -------------------------------------------------------------------------
    
    print("\n[STEP 2] Running analysis...")
    
    # Analyze compensation
    comp_results = analyze_compensation(employee_df)
    
    # Analyze benefits
    benefits_results = analyze_benefits(benefits_df, employee_df)
    
    # Analyze leave
    leave_results = analyze_leave(leave_df, employee_df)
    
    # -------------------------------------------------------------------------
    # Step 3: Generate Visualizations
    # -------------------------------------------------------------------------
    
    print("\n[STEP 3] Generating visualizations...")
    
    # Create compensation charts
    create_compensation_visualizations(employee_df, comp_results)
    
    # Create benefits charts
    create_benefits_visualizations(benefits_df, employee_df)
    
    # Create leave charts
    create_leave_visualizations(leave_df, employee_df)
    
    # Create executive summary dashboard
    create_executive_summary(employee_df, benefits_df, leave_df)
    
    # -------------------------------------------------------------------------
    # Step 4: Export Data
    # -------------------------------------------------------------------------
    
    print("\n[STEP 4] Exporting data files...")
    export_data(employee_df, benefits_df, leave_df)
    
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
    print("    - 05_headcount_by_dept_level.png")
    print("    - 06_medical_enrollment.png")
    print("    - 07_401k_contributions.png")
    print("    - 08_benefits_cost_by_dept.png")
    print("    - 09_benefits_overview.png")
    print("    - 10_pto_utilization.png")
    print("    - 11_leave_balances.png")
    print("    - 12_extended_leave.png")
    print("\n" + "="*70)


# This block ensures main() only runs when the script is executed directly
# (not when imported as a module by another script)
if __name__ == "__main__":
    main()
