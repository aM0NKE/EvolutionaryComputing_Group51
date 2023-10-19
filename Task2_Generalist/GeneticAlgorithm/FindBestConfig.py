import pandas as pd

test_statistic = 'max_gain'

# Load the data from data.txt into a Pandas DataFrame
df = pd.read_csv('Results/FINALParameterTuning/results.txt', delim_whitespace=True, header=0, names=["parameter_comb", "trial", "mean_fitness", "max_fitness", "mean_gain", "max_gain"])
print(df)
# Group the data by 'parameter_comb' and calculate the mean of 'mean_gain' for each group
mean_stat_by_comb = df.groupby('parameter_comb')[test_statistic].mean()

# Find the parameter_comb with the highest average mean_gain
best_parameter_comb = mean_stat_by_comb.idxmax()
best_mean_stat = mean_stat_by_comb.max()

# Print the best parameter_comb and its corresponding mean_gain
print(f"The best parameter_comb is {best_parameter_comb} with an average {test_statistic} of {best_mean_stat:.4f}")
