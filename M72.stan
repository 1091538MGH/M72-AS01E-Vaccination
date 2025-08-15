functions {
  // Smooth sigmoid transition function
  real smooth_transition(real t, real t_start, real t_end, real val_start, real val_end, real steepness) {
    if (t <= t_start) return val_start;
    if (t >= t_end) return val_end;
    
    real midpoint = (t_start + t_end) / 2.0;
    real scale = steepness / (t_end - t_start);
    real sigmoid = 1.0 / (1.0 + exp(-scale * (t - midpoint)));
    
    return val_start + (val_end - val_start) * sigmoid;
  }
  
  // Helper function for vaccine protection with periodic re-vaccination
  real protection_status(real t, real vaccination_start_time, real waning_rate, real revaccination_interval) {
    if (t < vaccination_start_time) return 0.0;
    
    // Calculate time since the most recent vaccination campaign
    real time_since_start = t - vaccination_start_time;
    real cycles_completed = floor(time_since_start / revaccination_interval);
    real time_since_last_vaccination = time_since_start - cycles_completed * revaccination_interval;
    
    return exp(-waning_rate * time_since_last_vaccination);
  }
  vector tb_ode(real t, vector y, vector theta, real[] x_r, int[] x_i) {
    // Unpack fixed parameters from x_r
    real mu = x_r[1];
    real mu1 = x_r[2];
    real p = x_r[3];
    real c_Ac1 = x_r[4];
    real c_As1 = x_r[5];
    real sigma = x_r[6];
    real kappa = x_r[7];
    real chi = x_r[8];
    real equilibrium_end_time = x_r[9];

    // Unpack estimated parameters from theta array
    real theta_param = theta[1]; 
    real rho = theta[2]; 
    real w = theta[3];
    real alpha = theta[4]; 
    real f = theta[5]; 
    real s = theta[6];  
    real mu_sev = theta[7]; 
    real theta1 = theta[8]; 
    real alpha_1 = theta[9]; 
    real f1 = theta[10]; 
    real s1 = theta[11]; 
    real mu_sev1 = theta[12]; 
    real beta = theta[13];
    real eta = theta[14];
    real eta1 = theta[15];
    real DOTS = theta[11];
    real ACT = theta[12];
    real TPT_c = theta[13];
    real TPT_a = theta[14];
    real beta_base = theta[15];
    real eta_base = theta[16];
    real eta1_base = theta[17];
    real TPT_min_fraction = theta[18];
    real TPT_future_trend_c = theta[19];  
    real TPT_future_trend_a = theta[20];
    
    // Unpack compartment states
    real S = y[1];   real L = y[2];   real As = y[3];  real Ac = y[4];  real I = y[5];   real R = y[6];
    real S1 = y[7];  real L1 = y[8];  real As1 = y[9]; real Ac1 = y[10]; real I1 = y[11]; real R1 = y[12];
    
    // Population and transmission calculations
    real P_children = S + L + As + Ac + I + R;
    real P_adults = S1 + L1 + As1 + Ac1 + I1 + R1;
    real Total_pop = P_children + P_adults;
    real Total_infectious = c_Ac1 * Ac1 + c_As1 * As1;  // Only adults infectious
    
    // Birth rate and force of infection
    real Birth = mu_sev * Ac + mu * P_children + mu_sev1 * Ac1 + mu1 * P_adults;
    
    real lambda = beta * Total_infectious / Total_pop;

    // ODE system
    vector[12] dydt;
    // Children: get infected by adults, don't transmit
    dydt[1] = Birth - (mu + sigma) * S;
    dydt[2] = lambda * S - (sigma + mu + theta_param) * L;
    dydt[3] = p * theta_param * L + rho * Ac + alpha * R - (w + sigma + mu + chi) * As;
    dydt[4] = (1 - p) * theta_param * L + w * As + eta * I - (sigma + mu + mu_sev + f + s + rho) * Ac;
    dydt[5] = f * Ac - (kappa + mu + sigma + eta) * I;
    dydt[6] = s * Ac + chi * As + kappa * I - (alpha + mu + sigma) * R;

    // Adults: get infected by adults, can transmit
    dydt[7] = sigma * S - (mu1 + lambda) * S1;
    dydt[8] = sigma * L + lambda * S1 - (mu1 + theta1) * L1;
    dydt[9] = sigma * As + p * theta1 * L1 + rho * Ac1 + alpha_1 * R1 - (w + mu1 + chi) * As1;
    dydt[10] = sigma * Ac + (1 - p) * theta1 * L1 + w * As1 + eta1 * I1 - (mu1 + mu_sev1 + f1 + s1 + rho) * Ac1;
    dydt[11] = sigma * I + f1 * Ac1 - (kappa + mu1 + eta1) * I1;
    dydt[12] = sigma * R + s1 * Ac1 + chi * As1 + kappa * I1 - (alpha_1 + mu1) * R1;

    return dydt;
  }
}

data {
  int<lower=1> n_fit;
  int<lower=1> n_forecast;
  int<lower=1> n_total;
  array[n_total] real ts;
  real t0;
  int<lower=1> N_pop;
  vector<lower=0>[12] y0;
  array[9] real x_r;  
  array[n_fit, 2] real observed_rates;
}

transformed data {
  array[1] int x_i = { N_pop }; 
  array[n_fit] real observed_incidence = observed_rates[, 1];
  array[n_fit] real observed_mortality = observed_rates[, 2];
}

parameters {
  real<lower=0, upper=0.6> theta_param;
  real<lower=0, upper=0.5> theta1;
  real<lower=0, upper=0.5> rho;
  real<lower=0, upper=0.2> w;
  real<lower=0.01, upper=0.5> beta;  // Reduced upper bound for stability
  real<lower=0, upper=0.5> alpha; 
  real<lower=0, upper=0.5> alpha_1; 
  real<lower=0, upper=0.2> eta; 
  real<lower=0, upper=0.2> eta1; 
  real<lower=1000, upper=50000> L;
  real<lower=100000, upper=1800000> L1;   
  
  simplex[3] treatment_outcomes_children;  // [f, s, mu_sev]
  simplex[3] treatment_outcomes_adults;    // [f1, s1, mu_sev1]
  
  real<lower=0> sigma_total_incidence;
  real<lower=0> sigma_total_mortality;
}

transformed parameters {
  // Treatment parameters
  real f = treatment_outcomes_children[1];
  real s = treatment_outcomes_children[2];
  real mu_sev = treatment_outcomes_children[3];
  real f1 = treatment_outcomes_adults[1];
  real s1 = treatment_outcomes_adults[2];
  real mu_sev1 = treatment_outcomes_adults[3];
  
  // SIMPLIFIED: Direct initial state construction
  vector[12] initial_state = y0;
  initial_state[2] = L;  
  initial_state[8] = L1;  
  
  // SIMPLIFIED: Automatic parameter packing using array literal
  vector[15] theta_ode = [theta_param, rho, w, alpha, f, s, mu_sev, 
                         theta1, alpha_1, f1, s1, mu_sev1, beta, eta, eta1]';

  // Solve ODE for fitting period
  array[n_fit] vector[12] y_hat = ode_rk45(tb_ode, initial_state, t0, ts[1:n_fit], theta_ode, x_r, x_i);
  
  // Calculate rates - SIMPLIFIED with local scope
  array[n_fit] vector[2] rates_hat;
  for (n in 1:n_fit) {
    real P_total = sum(y_hat[n]);  // Total population
    real total_incidence = theta_param * y_hat[n, 2] + theta1 * y_hat[n, 8];
    real total_mortality = mu_sev * y_hat[n, 4] + mu_sev1 * y_hat[n, 10];
    
    rates_hat[n, 1] = total_incidence * 100000.0 / P_total;
    rates_hat[n, 2] = total_mortality * 100000.0 / P_total;
  }
}

model {
  // Priors
  theta_param ~ lognormal(log(0.3), 0.45);
  theta1 ~ lognormal(log(0.2), 0.4);
  rho ~ lognormal(log(0.05), 0.05);
  w ~ lognormal(log(0.05), 0.25);
  beta ~ lognormal(log(0.05), 0.4);  // Lower mean for stability
  alpha ~ lognormal(log(0.08), 0.25);
  alpha_1 ~ lognormal(log(0.04), 0.3);
  eta ~ lognormal(log(0.06), 0.35);
  eta1 ~ lognormal(log(0.06), 0.25);
  L ~ lognormal(log(10000), 0.7);
  L1 ~ lognormal(log(100000), 0.75);
  
  treatment_outcomes_children ~ dirichlet([1.5, 8, 0.5]');
  treatment_outcomes_adults ~ dirichlet([1.5, 8, 0.5]');
  
  sigma_total_incidence ~ exponential(0.1);   
  sigma_total_mortality ~ exponential(0.2);   
  
  // Likelihood - SIMPLIFIED
  observed_incidence ~ normal(rates_hat[, 1], sigma_total_incidence);
  observed_mortality ~ normal(rates_hat[, 2], sigma_total_mortality);
}

generated quantities {
  // Full trajectory
  array[n_total] vector[12] y_full = ode_rk45(tb_ode, initial_state, t0, ts, theta_ode, x_r, x_i);
  
  // Predictions - SIMPLIFIED with vectorized operations
  array[n_total] real predicted_total_incidence_per_100k;
  array[n_total] real predicted_total_mortality_per_100k;
  
  for (n in 1:n_total) {
    real P_total = sum(y_full[n]);
    real total_incidence = theta_param * y_full[n, 2] + theta1 * y_full[n, 8];
    real total_mortality = mu_sev * y_full[n, 4] + mu_sev1 * y_full[n, 10];
    
    predicted_total_incidence_per_100k[n] = total_incidence * 100000.0 / P_total;
    predicted_total_mortality_per_100k[n] = total_mortality * 100000.0 / P_total;
  }
}
