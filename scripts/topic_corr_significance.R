library(readxl)

df <- read_xlsx("results_table.xlsx", sheet="Human rating correlations")

t.test(df$sentecon_corr, df$sentecon_plus_corr, paired=TRUE, alternative = "two.sided")

t.test(df$sentecon_corr, df$liwc_corr, paired=TRUE, alternative = "two.sided")

t.test(df$sentecon_plus_corr, df$liwc_corr, paired=TRUE, alternative = "two.sided")