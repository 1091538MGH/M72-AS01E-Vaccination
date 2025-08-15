library(rstan)
library(ggplot2)
library(corrplot)
library(bayesplot)
library(dplyr)
library(tidyr)
library(posterior)
library(loo)
library(gridExtra)
library(viridis)

# Set Stan options for better performance
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

start_year <- 2000
end_year <- 2035
fit_end_year <- 2023

# Calculate time dimensions
n_fit <- fit_end_year - start_year + 1      # 2000-2023: 24 years
n_forecast <- end_year - fit_end_year       # 2024-2035: 11 years  
n_total <- end_year - start_year + 1        # 2000-2035: 35 years_total - min(years_total) # Time in years since 2000

# Time points setup
t0 <- -0.1  # Year 2000
ts <- seq(from = 0, to = n_total- 1 , length.out = n_total)  # 1 to 36 (years since 2000)
years <- start_year + ts
fit_years <- years[1:n_fit]
forecast_years <- years[(n_fit + 1):n_total]


# Population data - Initial conditions
y0 <- c(
  S = 1777700,
  L = 0,
  As = 465,
  Ac = 1613,
  I = 123,
  R = 170,
  S1 = 15535729,
  L1 = 0,
  As1 = 7902,
  Ac1 = 8911,
  I1 = 1070,
  R1 = 10042
)
N_pop <- sum(y0)

# Fixed parameters array
x_r <- c(
  mu = 1/71,
  mu1 = 1/66,
  p = 0.47, 
  c_Ac1 = 0.8,
  c_As1 = 0.54,
  sigma = 1/10,
  kappa = 0.85,
  chi = 0.25  
)

# ==============================================================================
# 2. OBSERVED DATA PREPARATION
# ==============================================================================
observed <- read.csv("Stan_Data.csv")
observed_cases <- as.matrix(observed) 

# ==============================================================================
# 3. STAN DATA PREPARATION
# ==============================================================================
stan_data <- list(
  n_fit = n_fit,
  n_forecast = n_forecast,
  n_total = n_total,
  ts = ts,
  t0 = t0,
  y0 = y0,
  x_r = x_r,
  N_pop = N_pop,
  observed_cases = observed_cases 
)

# ==============================================================================
# 4. MODEL CONFIGURATION
# ==============================================================================
n_chains <- 4  
n_iter_total <- 700 
n_iter_warmup <- 300 

control_params <- list(
  adapt_delta = 0.95,
  max_treedepth = 12)

# ==============================================================================
# 6. MODEL COMPILATION AND FITTING
# ==============================================================================
# Ensure the corrected Stan file is named "Baseline.stan"
stan_model <- stan_model(file = "Baseline.stan")

start_time <- Sys.time()
fit <- sampling(
  object = stan_model,
  data = stan_data,
  chains = n_chains,
  iter = n_iter_total,
  warmup = n_iter_warmup,
  control = control_params,
  refresh = 100,
  seed = 125446,
  verbose = TRUE
)
end_time <- Sys.time()
runtime <- end_time - start_time
print(paste("Total runtime:", runtime))


# ==============================================================================
# 7. ENHANCED DIAGNOSTICS
# ==============================================================================

# Check for convergence issues
print(fit, pars = c("theta_param_base", "beta", "L", "L1", "theta1_base","f_base", "mu_sev1_base", "mu_sev_base",
                    "rho", "w", "alpha", "alpha_1", "eta_base", "eta1_base","f1_base","s_base","s1_base"),
      digits = 4)  # Increase decimal places to 4
# Check energy diagnostics
mcmc_nuts_energy(nuts_params(sampler_params))

# ==============================================================================
# 8. COMPREHENSIVE STAN DIAGNOSTICS
# ==============================================================================

# Define key parameters to analyze
key_params <- c("theta_param_base", "beta", "theta1_base", "rho", "w", 
                "alpha", "alpha_1", "eta_base", "eta1_base")

# ==============================================================================
# 8.1 TRACE PLOTS
# ==============================================================================

# Extract posterior samples for key parameters
posterior_samples <- rstan::extract(fit, pars = key_params, inc_warmup = FALSE, permuted = FALSE)

# Create trace plots using bayesplot
trace_plots <- mcmc_trace(posterior_samples, pars = key_params)
print(trace_plots)

# ==============================================================================
# 8.2 AUTOCORRELATION PLOTS
# ==============================================================================

# Extract draws for autocorrelation analysis
draws <- as_draws_array(fit)

# Autocorrelation plots for key parameters
autocorr_plots <- mcmc_acf(draws, pars = key_params, lags = 20)
print(autocorr_plots)

# ==============================================================================
# 8.3 POSTERIOR DISTRIBUTION PLOTS
# ==============================================================================

# Density plots for posterior distributions
posterior_density <- mcmc_dens_overlay(posterior_samples, pars = key_params)
print(posterior_density)

# Violin plots for posteriors
#posterior_violin <- mcmc_violin(posterior_samples, pars = key_params)
#print(posterior_violin)

# ==============================================================================
# 8.6 PAIRS PLOTS FOR CORRELATION ANALYSIS
# ==============================================================================

# Pairs plot for key parameters (shows correlations)
pairs_plot <- mcmc_pairs(
  posterior_samples, 
  pars = key_params[1:4],  # Limit to first 4 for readability
  off_diag_args = list(size = 0.5, alpha = 0.5)
)
print(pairs_plot)

# ==============================================================================
# 8.8 SAVE ALL PLOTS IN HIGH QUALITY
# ==============================================================================

# Save trace plots in high quality
ggsave("trace_plots_combined.png", trace_plots, width = 14, height = 10, dpi = 300)

# Save autocorrelation plots
ggsave("autocorrelation_combined.png", autocorr_plots, width = 14, height = 10, dpi = 300)

# Save posterior distributions
ggsave("posterior_distributions_combined.png", posterior_density, width = 14, height = 10, dpi = 300)

# ==============================================================================
# 8. MODEL FITTING VISUALIZATION WITH 95% CREDIBLE INTERVALS
# ==============================================================================

# Extract predicted cases from the Stan fit object
predicted_children_cases_full_samples <- rstan::extract(fit, pars = "predicted_children_cases_full")$predicted_children_cases_full
predicted_adults_cases_full_samples <- rstan::extract(fit, pars = "predicted_adults_cases_full")$predicted_adults_cases_full

# Calculate mean and 95% credible intervals
mean_predicted_children_cases <- apply(predicted_children_cases_full_samples, 2, mean)
mean_predicted_adults_cases <- apply(predicted_adults_cases_full_samples, 2, mean)

lower_children_cases <- apply(predicted_children_cases_full_samples, 2, quantile, probs = 0.025)
upper_children_cases <- apply(predicted_children_cases_full_samples, 2, quantile, probs = 0.975)
lower_adults_cases <- apply(predicted_adults_cases_full_samples, 2, quantile, probs = 0.025)
upper_adults_cases <- apply(predicted_adults_cases_full_samples, 2, quantile, probs = 0.975)

# Get observed data
observed_children_cases <- stan_data$observed_cases[, 1]
observed_adults_cases <- stan_data$observed_cases[, 2]

# Create data frames for plotting
children_plot_df <- data.frame(
  year = fit_years,
  predicted_mean = mean_predicted_children_cases[1:n_fit],
  predicted_lower = lower_children_cases[1:n_fit],
  predicted_upper = upper_children_cases[1:n_fit],
  observed = observed_children_cases[1:n_fit]
)

adults_plot_df <- data.frame(
  year = fit_years,
  predicted_mean = mean_predicted_adults_cases[1:n_fit],
  predicted_lower = lower_adults_cases[1:n_fit],
  predicted_upper = upper_adults_cases[1:n_fit],
  observed = observed_adults_cases[1:n_fit]
)

# Quick diagnostic
cat("CI width - Children:", mean(children_plot_df$predicted_upper - children_plot_df$predicted_lower), "\n")
cat("CI width - Adults:", mean(adults_plot_df$predicted_upper - adults_plot_df$predicted_lower), "\n")

# Plot Children Cases with enhanced credible intervals
plot_children <- ggplot(children_plot_df, aes(x = year)) +
  geom_ribbon(aes(ymin = predicted_lower, ymax = predicted_upper), 
              fill = "lightblue", alpha = 0.6, color = "blue", size = 0.2) +
  geom_line(aes(y = predicted_mean, color = "Predicted"), size = 1.2) +
  geom_point(aes(y = observed, color = "Observed"), size = 2) +
  labs(
    title = "Children Cases: Model Fit with 95% Credible Intervals",
    x = "Year", y = "Number of Cases"
  ) +
  scale_color_manual(values = c("Predicted" = "blue", "Observed" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

# Plot Adults Cases with enhanced credible intervals  
plot_adults <- ggplot(adults_plot_df, aes(x = year)) +
  geom_ribbon(aes(ymin = predicted_lower, ymax = predicted_upper), 
              fill = "lightgreen", alpha = 0.6, color = "darkgreen", size = 0.2) +
  geom_line(aes(y = predicted_mean, color = "Predicted"), size = 1.2) +
  geom_point(aes(y = observed, color = "Observed"), size = 2) +
  labs(
    title = "Adults Cases: Model Fit with 95% Credible Intervals",
    x = "Year", y = "Number of Cases"
  ) +
  scale_color_manual(values = c("Predicted" = "darkgreen", "Observed" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

# Display and save plots
grid.arrange(plot_children, plot_adults, ncol = 2)
ggsave("children_cases_with_CI.png", plot = plot_children, width = 10, height = 6, dpi = 300)
ggsave("adults_cases_with_CI.png", plot = plot_adults, width = 10, height = 6, dpi = 300)

# ==============================================================================
# 10. FORECASTING VISUALIZATION
# ==============================================================================

# Extract full time series predictions (including forecast)
predicted_children_full <- rstan::extract(fit, pars = "predicted_children_cases_full")$predicted_children_cases_full
predicted_adults_full <- rstan::extract(fit, pars = "predicted_adults_cases_full")$predicted_adults_cases_full

# Calculate statistics for full time series
mean_children_full <- apply(predicted_children_full, 2, mean)
mean_adults_full <- apply(predicted_adults_full, 2, mean)

lower_children_full <- apply(predicted_children_full, 2, quantile, probs = 0.025)
upper_children_full <- apply(predicted_children_full, 2, quantile, probs = 0.975)
lower_adults_full <- apply(predicted_adults_full, 2, quantile, probs = 0.025)
upper_adults_full <- apply(predicted_adults_full, 2, quantile, probs = 0.975)

# Create continuous forecast data frames
forecast_children_df <- data.frame(
  year = years,
  predicted_mean = mean_children_full,
  predicted_lower = lower_children_full,
  predicted_upper = upper_children_full,
  period = ifelse(years <= fit_end_year, "Fit", "Forecast")
)

forecast_adults_df <- data.frame(
  year = years,
  predicted_mean = mean_adults_full,
  predicted_lower = lower_adults_full,
  predicted_upper = upper_adults_full,
  period = ifelse(years <= fit_end_year, "Fit", "Forecast")
)

# Add observed data for plotting
observed_children_extended <- c(observed_children_cases, rep(NA, n_forecast))
observed_adults_extended <- c(observed_adults_cases, rep(NA, n_forecast))

forecast_children_df$observed <- observed_children_extended
forecast_adults_df$observed <- observed_adults_extended

# Quick diagnostic
cat("Forecast CI width - Children:", mean(forecast_children_df$predicted_upper - forecast_children_df$predicted_lower), "\n")
cat("Forecast CI width - Adults:", mean(forecast_adults_df$predicted_upper - forecast_adults_df$predicted_lower), "\n")

# Plot children forecast with continuous ribbon
plot_forecast_children <- ggplot(forecast_children_df, aes(x = year)) +
  geom_ribbon(aes(ymin = predicted_lower, ymax = predicted_upper, fill = period), 
              alpha = 0.7, color = "darkblue", size = 0.2) +
  geom_line(aes(y = predicted_mean, color = period), size = 1.2) +
  geom_point(aes(y = observed), color = "red", size = 2, na.rm = TRUE) +
  geom_vline(xintercept = fit_end_year + 0.5, linetype = "dashed", color = "black", size = 1) +
  labs(
    title = "Children TB Cases: Fit (2000-2023) and Forecast (2024-2035)",
    x = "Year",
    y = "Number of Cases"
  ) +
  scale_fill_manual(values = c("Fit" = "lightblue", "Forecast" = "lightcoral"),
                    name = "Period") +
  scale_color_manual(values = c("Fit" = "blue", "Forecast" = "darkred"),
                     name = "Period") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot adults forecast with continuous ribbon
plot_forecast_adults <- ggplot(forecast_adults_df, aes(x = year)) +
  geom_ribbon(aes(ymin = predicted_lower, ymax = predicted_upper, fill = period), 
              alpha = 0.7, color = "darkgreen", size = 0.2) +
  geom_line(aes(y = predicted_mean, color = period), size = 1.2) +
  geom_point(aes(y = observed), color = "red", size = 2, na.rm = TRUE) +
  geom_vline(xintercept = fit_end_year + 0.5, linetype = "dashed", color = "black", size = 1) +
  labs(
    title = "Adult TB Cases: Fit (2000-2023) and Forecast (2024-2035)",
    x = "Year",
    y = "Number of Cases"
  ) +
  scale_fill_manual(values = c("Fit" = "lightgreen", "Forecast" = "lightsalmon"),
                    name = "Period") +
  scale_color_manual(values = c("Fit" = "darkgreen", "Forecast" = "darkred"),
                     name = "Period") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Display forecast plots
grid.arrange(plot_forecast_children, plot_forecast_adults, ncol = 2)

# Save forecast plots
ggsave("children_forecast_continuous.png", plot = plot_forecast_children, width = 12, height = 6, dpi = 300)
ggsave("adults_forecast_continuous.png", plot = plot_forecast_adults, width = 12, height = 6, dpi = 300)

print("Forecasting plots complete with continuous credible intervals.")
