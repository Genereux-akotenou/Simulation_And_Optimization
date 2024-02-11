library(tidyverse)

set.seed(0)
norms <- rnorm(200)
#norms <- rchisq(200, 4)
df <- data.frame(norms)

ggplot(data = df, aes(sample = norms)) +
  geom_qq() +
  geom_qq_line()

# Let see the histogram of the data
qplot(x = norms)

# To get help
??ggplot

# Let considere that for our dataframe we have many groups
df$group <- factor(sample(1:3, 200, replace = T))
ggplot(data = df, aes(sample = norms, col = group)) +
  geom_qq() +
  geom_qq_line() + 
  facet_wrap(~group) +
  theme_minimal() + 
  scale_color_brewer(palette = "Dark2")

# Test fro chi2
chisq <- rchisq(200, 4)
df2 <- data.frame(chisq)
ggplot(data= df2, aes(sample = chisq)) +
  geom_qq(distribution = qchisq, dparams = list(df = 4)) +
  geom_qq_line(distribution = qchisq, dparams = list(df = 4))




