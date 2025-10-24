import matplotlib.pyplot as plt
import pandas as pd

# ============================================
# Matplotlib Style Configuration
# ============================================
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['grid.color'] = '#e1e1e1'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['axes.titlesize'] = 52
plt.rcParams['axes.labelsize'] = 52
plt.rcParams['lines.linewidth'] = 8
plt.rcParams['lines.markersize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 52
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['figure.dpi'] = 600

# ============================================
# Data: Repository Size and Model Count
# ============================================
# Total repository size in TB
total_size_data = {
    '2020Q1': 0.84, '2020Q2': 2.06, '2020Q3': 3.58, '2020Q4': 6.58,
    '2021Q1': 10.11, '2021Q2': 13.65, '2021Q3': 18.53, '2021Q4': 24.41,
    '2022Q1': 33.94, '2022Q2': 51.78, '2022Q3': 79.50, '2022Q4': 147.68,
    '2023Q1': 281.66, '2023Q2': 527.11, '2023Q3': 1023.47, '2023Q4': 1486.72,
    '2024Q1': 2589.30, '2024Q2': 4369.60, '2024Q3': 7373.95, '2024Q4': 10871.34,
    '2025Q1': 14592,
}

# Number of models in repository
model_count_data = {
    '2020Q1': 380, '2020Q2': 2006, '2020Q3': 3385, '2020Q4': 4623,
    '2021Q1': 7243, '2021Q2': 10525, '2021Q3': 16476, '2021Q4': 23818,
    '2022Q1': 33683, '2022Q2': 53635, '2022Q3': 72360, '2022Q4': 104636,
    '2023Q1': 162828, '2023Q2': 240640, '2023Q3': 356091, '2023Q4': 450752,
    '2024Q1': 576662, '2024Q2': 744030, '2024Q3': 1016317, '2024Q4': 1334253,
    '2025Q1': 1567579,
}

# Prepare data for plotting
date_list = list(total_size_data.keys())
year_labels = [d[:4] if d.endswith('Q1') else "" for d in date_list]  # Show year only at Q1

# Create DataFrames
df_size = pd.DataFrame({
    'Date': date_list,
    'Size': list(total_size_data.values())
})
df_count = pd.DataFrame({
    'Date': date_list,
    'Count': [v / 1000 for v in model_count_data.values()]  # Convert to thousands
})

# ============================================
# Create Repository Growth Plot
# ============================================
fig, ax_left = plt.subplots()
fig.set_size_inches(14, 10)

# Create second y-axis
ax_right = ax_left.twinx()

# Plot model count (left axis, red dashed line)
ax_left.plot(df_count['Date'], df_count['Count'], 
             'r--', linewidth=4, label='Model Count (K)')

# Plot total size (right axis, black solid line)
ax_right.plot(df_size['Date'], df_size['Size'], 
              'k-', linewidth=4, label='Total Size (TB)')

# Configure axes labels
ax_left.set_xlabel("Year")
ax_left.set_ylabel("Model Count (K)")
ax_right.set_ylabel("Total Size (TB)", fontsize=50)

# Configure x-axis ticks
ax_left.set_xticks(df_count['Date'])
ax_left.set_xticklabels(year_labels, rotation=0, fontsize=45)

# Set logarithmic scale for both y-axes
ax_left.set_yscale('log')
ax_right.set_yscale('log')

# Combine legends from both axes
lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_right.legend(lines_left + lines_right, labels_left + labels_right, 
                loc='upper left', fontsize=35)

# Save figure
fig.tight_layout()
fig.savefig("./repo_growth_2col.pdf", bbox_inches="tight")

print("Figure saved as 'repo_growth_2col.pdf'")