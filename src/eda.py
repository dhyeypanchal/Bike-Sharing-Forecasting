import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import CONTINUOUS_COLS, FEATURE_COLS, TARGET_COL, load_data

ATTRIBUTE_INFO = {
    'season':     ('Categorical (nominal)', '1=winter, 2=spring, 3=summer, 4=fall'),
    'yr':         ('Binary', '0=2011, 1=2012'),
    'mnth':       ('Categorical (cyclical)', 'month 1-12'),
    'hr':         ('Categorical (cyclical)', 'hour 0-23'),
    'holiday':    ('Binary', '1 = public holiday'),
    'weekday':    ('Categorical (cyclical)', '0=Sunday ... 6=Saturday'),
    'workingday': ('Binary', '1 = neither weekend nor holiday'),
    'weathersit': ('Categorical (ordinal)', '1=clear ... 4=heavy rain/snow'),
    'temp':       ('Continuous', 'temperature in C divided by 41 (pre-normalized)'),
    'atemp':      ('Continuous', 'feels-like temperature in C divided by 50 (pre-normalized)'),
    'hum':        ('Continuous', 'relative humidity divided by 100 (pre-normalized)'),
    'windspeed':  ('Continuous', 'wind speed divided by 67 (pre-normalized)'),
    'cnt':        ('Integer count (TARGET)', 'total rentals in the hour'),
}


def attribute_table(df):
    """Build summary table with types, ranges and distribution stats."""
    rows = []
    for col in FEATURE_COLS + [TARGET_COL]:
        kind, meaning = ATTRIBUTE_INFO[col]
        row = {
            'Attribute': col,
            'Type': kind,
            'Meaning': meaning,
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Distinct values': df[col].nunique(),
        }
        if col in CONTINUOUS_COLS + [TARGET_COL]:
            row.update({
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Median': df[col].median(),
                'Q1': df[col].quantile(0.25),
                'Q3': df[col].quantile(0.75),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_hourly_profile(df, output_path):
    """Plot average hourly rental demand for working days vs weekends/holidays."""
    profile = df.groupby(['workingday', 'hr'])['cnt'].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(profile.index, profile[1], color='#2a78d6', marker='o', markersize=5,
            linewidth=2, label='Working day')
    ax.plot(profile.index, profile[0], color='#eb6834', marker='s', markersize=5,
            linewidth=2, linestyle='--', label='Weekend / holiday')
    
    ax.set_xticks(range(0, 24))
    ax.set_xlabel('Hour of day', fontsize=10)
    ax.set_ylabel('Average bikes rented per hour', fontsize=10)
    ax.set_title('Average hourly demand, 2011-2012 (all 17,379 hours)', fontsize=11, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e5e5e2', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return profile


def run_eda(df, results_dir):
    """Run full EDA, saving Table 1 and Figure 1 to results directory."""
    os.makedirs(results_dir, exist_ok=True)
    table = attribute_table(df)
    table.to_csv(os.path.join(results_dir, 'table1_attribute_summary.csv'), index=False)
    profile = plot_hourly_profile(df, os.path.join(results_dir, 'fig1_hourly_demand_profile.png'))
    return table, profile


if __name__ == '__main__':
    df = load_data()
    table, _ = run_eda(df, os.path.join('assignment1', 'results'))
    print("EDA completed. Generated Table 1:")
    print(table[['Attribute', 'Type', 'Min', 'Max']].to_string(index=False))
