# Load necessary libraries
library(MASS)  # For truehist function
library(car)   # For qqPlot function

# Generate example data (replace this with your actual data)
set.seed(123)
data_chi2 <- rchisq(100, df = 4)  # Example chi-squared data with 4 degrees of freedom

# Create QQ plot comparing chi-squared to normal distribution
qqPlot(data_chi2, distribution = "norm", pch = 20, col = "blue", main = "QQ Plot - Chi-Squared vs. Normal")

