#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
from statsmodels.stats.power import TTestIndPower
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm
from scipy.stats import beta


# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000]
})

# Mean and standard deviation
mean_age = data['age'].mean()
std_income = data['income'].std()

print(f"Mean age: {mean_age}")
print(f"Standard deviation of income: {std_income}")



# Sample data
sample_data = [25, 30, 35, 40, 45]

# Conducting a t-test
t_statistic, p_value = stats.ttest_1samp(sample_data, popmean=30)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Using pandas for correlation
correlation_matrix = data.corr()

print(correlation_matrix)


# Sample data
X = data['age']
y = data['income']

# Adding intercept term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print summary of regression results
print(model.summary())

# Sample data
data = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'C'],
    'value': [10, 15, 20, 25, 30]
})

# Fit ANOVA model
model = ols('value ~ group', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

# Sample data
group1 = [1, 2, 3, 4, 5]
group2 = [6, 7, 8, 9, 10]

# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(group1, group2)

print(f"Mann-Whitney U statistic: {statistic}")
print(f"P-value: {p_value}")


# Sample time series data
dates = pd.date_range('2023-01-01', periods=100)
data = pd.Series(range(100), index=dates)

# Decompose time series
decomposition = seasonal_decompose(data, model='additive')

# Plot results
decomposition.plot()

# Power analysis for independent t-test
effect_size = 0.5
alpha = 0.05
power = 0.8

power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)

print(f"Required sample size: {sample_size}")

# Sample data
X = data[['age']]
y = data['income']

# Fit linear regression model
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Print summary of regression results
print(model.summary())


# Sample data
sample_mean = 100
sample_std = 15
n = 50

# Calculate confidence interval
z = norm.ppf(0.975)  # 95% confidence level
margin_of_error = z * (sample_std / (n**0.5))
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"Confidence interval: {confidence_interval}")


# Beta distribution parameters
alpha_prior = 5
beta_prior = 10

# Update belief with new evidence (e.g., data)
alpha_posterior = alpha_prior + 20
beta_posterior = beta_prior + 30

# Generate posterior distribution
posterior_distribution = beta(alpha_posterior, beta_posterior)

# Calculate credible interval
credible_interval = posterior_distribution.interval(0.95)

print(f"Credible interval: {credible_interval}")



# In[ ]:




