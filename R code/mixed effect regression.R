# Analysis of the correlation of criteria

# Mixed effects (used for comparison with python implementation)
# Values of mixed effects model are used for analysis
library(lme4)
mixed <- lmer(diversity ~ ILS + (1|id), data = all)
summary(mixed)


