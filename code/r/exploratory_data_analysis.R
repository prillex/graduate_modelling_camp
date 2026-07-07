# Data Modelling Camp 2026

# Libraries ----
library(tidyverse)
library(scales)


# Data ----
data_descriptive <- read.csv("data/raw/Cambridge data descriptive.csv")
cambridge_data <- read.csv("data/cleaned/Cambridge data_cleaned.csv")
spatial_data <- read.csv("data/spatial/spatial_data.csv")
  
cambridge_data_cleaned <- merge(cambridge_data, spatial_data[, c("MSOA21CD", "LAT", "LONG")], by.x = "msoa21", by.y = "MSOA21CD", all.x = TRUE)

detached_houses <- cambridge_data_cleaned[cambridge_data_cleaned$property_type == "Detached house", ]

# Exploratory Plots ---
# Exploratory Plots ----
# 1. PRICE BY PROPERTY TYPE
ggplot(cambridge_data, aes(property_type, price_sold, fill = property_type)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(alpha = 0.1, size = 0.5, width = 0.2) +
  coord_cartesian(ylim = c(0, quantile(cambridge_data$price_sold, 0.98, na.rm = TRUE))) +
  labs(title = "House Prices by Property Type", x = "Property Type", y = "Price (£)") +
  scale_y_continuous(labels = comma) +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

# Summary stats
cambridge_data %>%
  group_by(property_type) %>%
  summarise(across(price_sold, list(n = ~n(), mean = mean, median = median,
                                    min = min, max = max, sd = sd)))

# 2. PRICE BY BEDROOMS
ggplot(cambridge_data, aes(as.factor(num_bed_), price_sold)) +
  geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.shape = NA) +
  geom_jitter(alpha = 0.05, size = 0.5, width = 0.2) +
  coord_cartesian(ylim = c(0, quantile(cambridge_data$price_sold, 0.98, na.rm = TRUE))) +
  labs(title = "House Prices by Bedrooms", x = "Number of Bedrooms", y = "Price (£)") +
  scale_y_continuous(labels = comma) +
  theme_minimal()

# Summary stats
cambridge_data %>%
  group_by(num_bed_) %>%
  summarise(across(price_sold, list(n = ~n(), mean = mean, median = median,
                                    min = min, max = max)))

# 3. CORRELATIONS WITH PRICE
cors <- cambridge_data %>%
  select(where(is.numeric)) %>%
  cor(use = "complete.obs") %>%
  .[, "price_sold"] %>%
  .[names(.) != "price_sold"] %>%
  sort(decreasing = TRUE)

print(cors)

cor_df <- tibble(variable = names(cors), correlation = cors) %>%
  slice_max(abs(correlation), n = 15)

ggplot(cor_df, aes(x = reorder(variable, correlation), y = correlation,
                   fill = correlation > 0)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "coral"), guide = "none") +
  labs(title = "Top 15 Correlations with Price", x = NULL, y = "Correlation") +
  theme_minimal()

# Linear models ----
summary(lm(log(price_sold) ~ num_bed_ + num_bath, data = detached_houses))
summary(lm(log(price_sold) ~ num_bed_ + num_bath + LAT + LONG, data = detached_houses))
summary(lm(log(price_sold) ~ num_bed_ + num_bath + factor(msoa21), data = detached_houses))

summary(lm(log(price_sold) ~ num_bed_ + num_bath, data = detached_houses))
summary(lm(log(price_sold) ~ num_bed_ + num_bath + LAT + LONG, data = detached_houses))
summary(lm(log(price_sold) ~ num_bed_ + num_bath + factor(msoa21), data = detached_houses))
