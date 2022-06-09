# Diversity calculations
# -----------------------------------------------------------------------------------------------

# Read necessary data from file
library(tidyverse)
library(readxl)

# Read dataset from the local file with the study that you want to investigate
# Save the Excel file as "all" 
# all 

uhigh <- filter(all, all$treatment == "Recommendation-based (high-ILS)")
ulow <- filter(all, all$treatment == "Recommendation-based (low-ILS)")
coll <- filter(all, all$treatment == "Sequels")

umin <- filter(all, all$treatment == "Recommendation-based (mid-ILS minimize similarity of neighbors)")
umax <- filter(all, all$treatment == "Recommendation-based (mid-ILS maximize similarity of neighbors)")
ushuf <- filter(all, all$treatment == "Recommendation-based (mid-ILS shuffled)")

hmax <- filter(all, all$treatment == "Homogenous (maximize similarity of neighbors)")
hmin <- filter(all, all$treatment == "Homogenous (minimize similarity of neighbors)")
hshuf <- filter(all, all$treatment == "Homogenous (shuffled)")

# Pairwise comparison using Wilcoxon rank with continuity correction on all data
kruskal.test(all$diversity~all$treatment)
kruskal.test(all$variety~all$treatment)
kruskal.test(all$similarity~all$treatment)
kruskal.test(all$easiness~all$treatment)
kruskal.test(all$confidence~all$treatment)

# Decision time T-test
# -----------------------------------------------------------------------------------------------
# ILS difference
t.test(ubas$logtime~ubas$treatment, var.equal = TRUE, alternative = "two.sided")

# Post-hoc TukeyHSD for decision time
all %>%
  group_by(all$treatment) %>%
  summarise(mean = mean(logtime), sd = sd(logtime), n = n())

model <- aov(firstChoice$logtime ~ firstChoice$treatment, data=firstChoice)
TukeyHSD(model, conf.level=.95)

# Post-hoc analysis (only on significant results)
pairwise.wilcox.test(all$diversity,all$treatment, p.adjust="bonferroni")

# Analysis of comparison to user-based (mid-ILS)
# -----------------------------------------------------------------------------------------------
comparison_all <- # Read the file with the corresponding study

# Analysis for movies
c_avatar <- filter(comparison_all, comparison_all$`comparison movie 1`=='Avatar')
c_inception <- filter(comparison_all, comparison_all$`comparison movie 1`=='Inception')
c_avengers <- filter(comparison_all, comparison_all$`comparison movie 1`=='The Avengers')
c_lord <- filter(comparison_all, comparison_all$`comparison movie 1`=='The Lord of the Rings: The Two Towers')
c_avatar$group <- "comparison"
c_inception$group <- "comparison"
c_avengers$group <- "comparison"
c_lord$group <- "comparison"
umin$group <- "umin"
umin.temp <- umin %>% rename (similarity1 = 'similarity')
c_avatarTest <- bind_rows(c_avatar, umin.temp)
c_inceptionTest <- bind_rows(c_inception, umin.temp)
c_avengersTest <- bind_rows(c_avengers, umin.temp)
c_lordTest <- bind_rows(c_lord, umin.temp)
wilcox.test(c_avatarTest$similarity1~c_avatarTest$group, exact = FALSE, correct = FALSE, conf.int = FALSE)
wilcox.test(c_inceptionTest$similarity1~c_inceptionTest$group, exact = FALSE, correct = FALSE, conf.int = FALSE)
wilcox.test(c_avengersTest$similarity1~c_avengersTest$group, exact = FALSE, correct = FALSE, conf.int = FALSE)
wilcox.test(c_lordTest$similarity1~c_lordTest$group, exact = FALSE, correct = FALSE, conf.int = FALSE)

# Analysis for recipes
c_cookies <- filter(comparison_all, comparison_all$`comparison 1`=='Award Winning Soft Chocolate Chip Cookies')
c_banana <- filter(comparison_all, comparison_all$`comparison 1`=='Banana Crumb Muffins')
c_ham <- filter(comparison_all, comparison_all$`comparison 1`=='Delicious Ham and Potato Soup')
c_pancakes <- filter(comparison_all, comparison_all$`comparison 1`=='Fluffy Pancakes')
c_cookies$group <- "comparison"
c_banana$group <- "comparison"
c_ham$group <- "comparison"
c_pancakes$group <- "comparison"
umin$group <- "umin"
umin.temp <- umin %>% rename ('comparison similarity' = similarity)
c_cookies.test <- bind_rows(c_cookies, umin.temp)
c_banana.test <- bind_rows(c_banana, umin.temp)
c_ham.test <- bind_rows(c_ham, umin.temp)
c_pancakes.test <- bind_rows(c_pancakes, umin.temp)
wilcox.test(c_cookies.test$'comparison similarity'~c_cookies.test$group, exact = FALSE, correct = FALSE, conf.int = FALSE)
wilcox.test(c_banana.test$'comparison similarity'~c_banana.test$group, exact = FALSE, correct = FALSE, conf.int = FALSE)
wilcox.test(c_ham.test$'comparison similarity'~c_ham.test$group, exact = FALSE, correct = FALSE, conf.int = FALSE)
wilcox.test(c_pancakes.test$'comparison similarity'~c_pancakes.test$group, exact = FALSE, correct = FALSE, conf.int = FALSE)

# Analysis of Cronbachs Alpha
# -----------------------------------------------------------------------------------------------
alpha(subset(Diversity_Study_Analysis_Work, select = c(diversity, variety, similarity)), keys = c("similarity"))
alpha(subset(Analysis_Recipe, select = c(diversity, variety, similarity)), keys = c("similarity"))
alpha(subset(ILS_Study_Movie_and_Recipe, select = c(diversity, variety, similarity)), keys = c("similarity"))

# Pearson Correlation
# -----------------------------------------------------------------------------------------------
spearman.sel <- cor.test(all_user$selection, all_user$ILS, method = "spearman")
spearman.div <- cor.test(all_user$diversity, all_user$ILS, method = "spearman")
spearman.var <- cor.test(all_user$variety, all_user$ILS, method = "spearman")
spearman.sim <- cor.test(all_user$similarity, all_user$ILS, method = "spearman")
spearman.eas <- cor.test(all_user$easiness, all_user$ILS, method = "spearman")
spearman.con <- cor.test(all_user$confidence, all_user$ILS, method = "spearman")

# Boxplot of variables (optional)
# -----------------------------------------------------------------------------------------------
par(mar=c(15,5,2,1))
ggplot(all, aes(confidence, treatment1)) + geom_boxplot() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
