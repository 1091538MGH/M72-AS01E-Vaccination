library(rstan)
library(ggplot2)
library(dplyr)
library(tidyr)

# Set Stan options
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ==============================================================================
# BASIC SETUP
# ==============================================================================

# Time setup
start_year <- 2000
end_year <- 2035
fit_end_year <- 2023
n_fit <- fit_end_year - start_year + 1
n_forecast <- end_year - fit_end_year
n_total <- end_year - start_year + 1
t0 <- -0.1
ts <- seq(from = 0, to = n_total - 1, length.out = n_total)
years <- start_year + ts

# Initial conditions
y0 <- c(S = 1777700, L = 0, As = 465, Ac = 1613, I = 123, R = 170,
        S1 = 15535729, L1 = 0, As1 = 7902, Ac1 = 8911, I1 = 1070, R1 = 10042)
N_pop <- sum(y0)

# Read observed data
observed <- read.csv("Stan_Data.csv")
observed_cases <- as.matrix(observed)

# Model settings
n_chains <- 2
n_iter_total <- 600
n_iter_warmup <- 150
control_params <- list(adapt_delta = 0.95, max_treedepth = 12)

# ==============================================================================
# SCENARIO PARAMETERS
# ==============================================================================

# Baseline parameters (no vaccination)
x_r_baseline <- c(
  mu = 1/71, mu1 = 1/66, p = 0.47, c_Ac1 = 0.8, c_As1 = 0.54,
  sigma = 1/10, kappa = 0.85, chi = 0.25,
  vaccine_efficacy = 0.5, vaccine_coverage = 0.0,  # No vaccination
  revaccination_interval = 4.0, vaccination_start_time = 2026, vaccine_waning_rate = 0.2310
)

# Low coverage parameters (25%)
x_r_low <- c(
  mu = 1/71, mu1 = 1/66, p = 0.47, c_Ac1 = 0.8, c_As1 = 0.54,
  sigma = 1/10, kappa = 0.85, chi = 0.25,
  vaccine_efficacy = 0.5, vaccine_coverage = 0.25,  # 25% coverage
  revaccination_interval = 4.0, vaccination_start_time = 2026, vaccine_waning_rate = 0.2310
)

# High coverage parameters (75%)
x_r_high <- c(
  mu = 1/71, mu1 = 1/66, p = 0.47, c_Ac1 = 0.8, c_As1 = 0.54,
  sigma = 1/10, kappa = 0.85, chi = 0.25,
  vaccine_efficacy = 0.5, vaccine_coverage = 0.75,  # 75% coverage
  revaccination_interval = 4.0, vaccination_start_time = 2026, vaccine_waning_rate = 0.2310
)

# Stan data for each scenario
stan_data_baseline <- list(n_fit = n_fit, n_forecast = n_forecast, n_total = n_total,
                          ts = ts, t0 = t0, y0 = y0, x_r = x_r_baseline, 
                          N_pop = N_pop, observed_cases = observed_cases)

stan_data_low <- list(n_fit = n_fit, n_forecast = n_forecast, n_total = n_total,
                     ts = ts, t0 = t0, y0 = y0, x_r = x_r_low, 
                     N_pop = N_pop, observed_cases = observed_cases)

stan_data_high <- list(n_fit = n_fit, n_forecast = n_forecast, n_total = n_total,
                      ts = ts, t0 = t0, y0 = y0, x_r = x_r_high, 
                      N_pop = N_pop, observed_cases = observed_cases)

# ==============================================================================
# RUN ALL SCENARIOS AT ONCE
# ==============================================================================

cat("Compiling Stan model...\n")
compiled_model <- stan_model(file = "tb_model_vac.stan")

cat("Running all scenarios simultaneously...\n")

# Run baseline scenario
cat("Starting baseline scenario...\n")
fit_baseline <- sampling(compiled_model, data = stan_data_baseline, chains = n_chains,
                        iter = n_iter_total, warmup = n_iter_warmup, 
                        control = control_params, refresh = 0, seed = 125446)

# Run low coverage scenario  
cat("Starting low coverage scenario...\n")
fit_low <- sampling(compiled_model, data = stan_data_low, chains = n_chains,
                   iter = n_iter_total, warmup = n_iter_warmup,
                   control = control_params, refresh = 0, seed = 125447)

# Run high coverage scenario
cat("Starting high coverage scenario...\n") 
fit_high <- sampling(compiled_model, data = stan_data_high, chains = n_chains,
                    iter = n_iter_total, warmup = n_iter_warmup,
                    control = control_params, refresh = 0, seed = 125448)

cat("All scenarios completed!\n\n")

# ==============================================================================
# EXTRACT RESULTS - Use this if you run individual scenarios 
# ==============================================================================


# Extract predictions from baseline scenario only
pred_baseline <- rstan::extract(fit_baseline, pars = c("predicted_adults_cases_full", "predicted_children_cases_full"))

# Calculate means and confidence intervals
adult_baseline_mean <- apply(pred_baseline$predicted_adults_cases_full, 2, mean)
adult_baseline_lower <- apply(pred_baseline$predicted_adults_cases_full, 2, quantile, 0.025)
adult_baseline_upper <- apply(pred_baseline$predicted_adults_cases_full, 2, quantile, 0.975)

children_baseline_mean <- apply(pred_baseline$predicted_children_cases_full, 2, mean)
children_baseline_lower <- apply(pred_baseline$predicted_children_cases_full, 2, quantile, 0.025)
children_baseline_upper <- apply(pred_baseline$predicted_children_cases_full, 2, quantile, 0.975)

# Create data frame for plotting
plot_data_baseline <- data.frame(
  year = years,
  adult_mean = adult_baseline_mean,
  adult_lower = adult_baseline_lower,
  adult_upper = adult_baseline_upper,
  children_mean = children_baseline_mean,
  children_lower = children_baseline_lower,
  children_upper = children_baseline_upper,
  total_mean = adult_baseline_mean + children_baseline_mean,
  total_lower = adult_baseline_lower + children_baseline_lower,
  total_upper = adult_baseline_upper + children_baseline_upper
)

# Color for baseline scenario
baseline_color <- "#E31A1C"

# Adult cases plot - Baseline Only
plot_adult_baseline <- ggplot(plot_data_baseline, aes(x = year, y = adult_mean)) +
  geom_ribbon(aes(ymin = adult_lower, ymax = adult_upper), fill = baseline_color, alpha = 0.2) +
  geom_line(color = baseline_color, size = 1.2) +
  geom_vline(xintercept = 2026, linetype = "dashed", alpha = 0.7) +
  labs(title = "Adult TB Cases - Baseline Scenario (0%)",
       x = "Year", y = "Adult TB Cases") +
  theme_minimal()

# Children cases plot - Baseline Only
plot_children_baseline <- ggplot(plot_data_baseline, aes(x = year, y = children_mean)) +
  geom_ribbon(aes(ymin = children_lower, ymax = children_upper), fill = baseline_color, alpha = 0.2) +
  geom_line(color = baseline_color, size = 1.2) +
  geom_vline(xintercept = 2026, linetype = "dashed", alpha = 0.7) +
  labs(title = "Children TB Cases - Baseline Scenario (0%)",
       x = "Year", y = "Children TB Cases") +
  theme_minimal()

# Display plots
print(plot_adult_baseline)
print(plot_children_baseline)

# Print total cases values for each year
cat("\nTotal TB Cases by Year - Baseline Scenario (0%):\n")
cat("Year\tTotal Cases\n")
cat("----\t-----------\n")
for(i in 1:length(years)) {
  cat(sprintf("%.0f\t%.0f\n", years[i], plot_data_baseline$total_mean[i]))
}


# Extract predictions from low coverage scenario only
pred_low <- extract(fit_low, pars = c("predicted_adults_cases_full", "predicted_children_cases_full"))

# Calculate means and confidence intervals
adult_low_mean <- apply(pred_low$predicted_adults_cases_full, 2, mean)
adult_low_lower <- apply(pred_low$predicted_adults_cases_full, 2, quantile, 0.025)
adult_low_upper <- apply(pred_low$predicted_adults_cases_full, 2, quantile, 0.975)

children_low_mean <- apply(pred_low$predicted_children_cases_full, 2, mean)
children_low_lower <- apply(pred_low$predicted_children_cases_full, 2, quantile, 0.025)
children_low_upper <- apply(pred_low$predicted_children_cases_full, 2, quantile, 0.975)

# Create data frame for plotting
plot_data_low <- data.frame(
  year = years,
  adult_mean = adult_low_mean,
  adult_lower = adult_low_lower,
  adult_upper = adult_low_upper,
  children_mean = children_low_mean,
  children_lower = children_low_lower,
  children_upper = children_low_upper,
  total_mean = adult_low_mean + children_low_mean,
  total_lower = adult_low_lower + children_low_lower,
  total_upper = adult_low_upper + children_low_upper
)

# Color for low coverage scenario
low_color <- "#FF7F00"

# Adult cases plot - Low Coverage Only
plot_adult_low <- ggplot(plot_data_low, aes(x = year, y = adult_mean)) +
  geom_ribbon(aes(ymin = adult_lower, ymax = adult_upper), fill = low_color, alpha = 0.2) +
  geom_line(color = low_color, size = 1.2) +
  geom_vline(xintercept = 2026, linetype = "dashed", alpha = 0.7) +
  labs(title = "Adult TB Cases - Low Coverage Scenario (25%)",
       x = "Year", y = "Adult TB Cases") +
  theme_minimal()

# Children cases plot - Low Coverage Only
plot_children_low <- ggplot(plot_data_low, aes(x = year, y = children_mean)) +
  geom_ribbon(aes(ymin = children_lower, ymax = children_upper), fill = low_color, alpha = 0.2) +
  geom_line(color = low_color, size = 1.2) +
  geom_vline(xintercept = 2026, linetype = "dashed", alpha = 0.7) +
  labs(title = "Children TB Cases - Low Coverage Scenario (25%)",
       x = "Year", y = "Children TB Cases") +
  theme_minimal()

# Display plots
print(plot_adult_low)
print(plot_children_low)

# Print total cases values for each year
cat("\nTotal TB Cases by Year - Low Coverage Scenario (25%):\n")
cat("Year\tTotal Cases\n")
cat("----\t-----------\n")
for(i in 1:length(years)) {
  cat(sprintf("%.0f\t%.0f\n", years[i], plot_data_low$total_mean[i]))
}
# ==============================================================================
# EXTRACT RESULTS - Use this if you run all scenarios first
# ==============================================================================

cat("Extracting results...\n")

# Extract predictions from each scenario
pred_baseline <- extract(fit_baseline, pars = c("predicted_adults_cases_full", "predicted_children_cases_full"))
pred_low <- extract(fit_low, pars = c("predicted_adults_cases_full", "predicted_children_cases_full"))
pred_high <- extract(fit_high, pars = c("predicted_adults_cases_full", "predicted_children_cases_full"))

# Calculate means and confidence intervals
adult_baseline_mean <- apply(pred_baseline$predicted_adults_cases_full, 2, mean)
adult_baseline_lower <- apply(pred_baseline$predicted_adults_cases_full, 2, quantile, 0.025)
adult_baseline_upper <- apply(pred_baseline$predicted_adults_cases_full, 2, quantile, 0.975)

adult_low_mean <- apply(pred_low$predicted_adults_cases_full, 2, mean)
adult_low_lower <- apply(pred_low$predicted_adults_cases_full, 2, quantile, 0.025)
adult_low_upper <- apply(pred_low$predicted_adults_cases_full, 2, quantile, 0.975)

adult_high_mean <- apply(pred_high$predicted_adults_cases_full, 2, mean)
adult_high_lower <- apply(pred_high$predicted_adults_cases_full, 2, quantile, 0.025)
adult_high_upper <- apply(pred_high$predicted_adults_cases_full, 2, quantile, 0.975)

children_baseline_mean <- apply(pred_baseline$predicted_children_cases_full, 2, mean)
children_low_mean <- apply(pred_low$predicted_children_cases_full, 2, mean)
children_high_mean <- apply(pred_high$predicted_children_cases_full, 2, mean)

# Create simple data frame for plotting
plot_data <- data.frame(
  year = rep(years, 3),
  scenario = rep(c("Baseline (0%)", "Low Coverage (25%)", "High Coverage (75%)"), each = length(years)),
  adult_mean = c(adult_baseline_mean, adult_low_mean, adult_high_mean),
  adult_lower = c(adult_baseline_lower, adult_low_lower, adult_high_lower),
  adult_upper = c(adult_baseline_upper, adult_low_upper, adult_high_upper),
  children_mean = c(children_baseline_mean, children_low_mean, children_high_mean),
  total_mean = c(adult_baseline_mean + children_baseline_mean, 
                adult_low_mean + children_low_mean,
                adult_high_mean + children_high_mean)
)

cat("Results extracted!\n\n")

# ==============================================================================
# CREATE PLOTS
# ==============================================================================

cat("Creating plots...\n")

# Colors for scenarios
colors <- c("Baseline (0%)" = "#E31A1C", 
           "Low Coverage (25%)" = "#FF7F00", 
           "High Coverage (75%)" = "#1F78B4")

# Adult cases plot
plot_adult <- ggplot(plot_data, aes(x = year, y = adult_mean, color = scenario, fill = scenario)) +
  geom_ribbon(aes(ymin = adult_lower, ymax = adult_upper), alpha = 0.2, color = NA) +
  geom_line(size = 1.2) +
  geom_vline(xintercept = 2026, linetype = "dashed", alpha = 0.7) +
  scale_color_manual(values = colors) +
  scale_fill_manual(values = colors) +
  labs(title = "Adult TB Cases by Vaccination Scenario",
       x = "Year", y = "Adult TB Cases", color = "Scenario", fill = "Scenario") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Children cases plot  
plot_children <- ggplot(plot_data, aes(x = year, y = children_mean, color = scenario)) +
  geom_line(size = 1.2) +
  geom_vline(xintercept = 2026, linetype = "dashed", alpha = 0.7) +
  scale_color_manual(values = colors) +
  labs(title = "Children TB Cases by Vaccination Scenario",
       x = "Year", y = "Children TB Cases", color = "Scenario") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Total cases plot
plot_total <- ggplot(plot_data, aes(x = year, y = total_mean, color = scenario)) +
  geom_line(size = 1.2) +
  geom_vline(xintercept = 2026, linetype = "dashed", alpha = 0.7) +
  scale_color_manual(values = colors) +
  labs(title = "Total TB Cases by Vaccination Scenario",
       x = "Year", y = "Total TB Cases", color = "Scenario") +
  theme_minimal() +
  theme(legend.position = "bottom")

cat("Plots created!\n\n")

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================

cat("DISPLAYING PLOTS:\n")
cat("==================\n\n")

print(plot_adult)
print(plot_children) 
print(plot_total)

# Calculate simple impact summary
post_2026 <- plot_data$year >= 2026

baseline_adult_total <- sum(plot_data$adult_mean[plot_data$scenario == "Baseline (0%)" & post_2026])
low_adult_total <- sum(plot_data$adult_mean[plot_data$scenario == "Low Coverage (25%)" & post_2026])
high_adult_total <- sum(plot_data$adult_mean[plot_data$scenario == "High Coverage (75%)" & post_2026])

cat("\nIMPACT SUMMARY (2026-2035):\n")
cat("============================\n")
cat("Adult cases prevented by 25% coverage:", round(baseline_adult_total - low_adult_total), "\n")
cat("Adult cases prevented by 75% coverage:", round(baseline_adult_total - high_adult_total), "\n")
cat("Percent reduction with 25% coverage:", round((baseline_adult_total - low_adult_total)/baseline_adult_total * 100, 1), "%\n")
cat("Percent reduction with 75% coverage:", round((baseline_adult_total - high_adult_total)/baseline_adult_total * 100, 1), "%\n")

cat("\nANALYSIS COMPLETE!\n")

