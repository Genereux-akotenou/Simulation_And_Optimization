# Generate data
data_chi2 <- rchisq(100, df = 4)
data_normal <- rnorm(100)

# Function to create QQ plot and test distribution
test_distribution <- function(data, distribution_name) {
  # Create QQ plot
  qqnorm(data, main = paste("QQ Plot -", distribution_name), col = "blue", pch = 20)
  qqline(data, col = "red")
}

# Test chi-squared distribution
test_distribution(data_chi2, "Chi-Squared")

# Test normal distribution
test_distribution(data_normal, "Normal")

