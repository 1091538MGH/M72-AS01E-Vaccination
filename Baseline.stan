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
  
  // Modified TPT function 
  real get_smooth_TPT(real t, real TPT_max_pre2021, real TPT_max_post2021, real TPT_min_fraction, real TPT_future_trend) {
    real TPT_max = TPT_max_pre2021;
    if (t > 2021) {
      TPT_max = TPT_max_post2021;
    }
    real TPT_min = TPT_max * TPT_min_fraction;
    
    if (t <= 2014) {
      return smooth_transition(t, 2010, 2014, 0.0, TPT_max, 2.5);
    }
    if (t <= 2021) {  
      return smooth_transition(t, 2014, 2021, TPT_max, TPT_min, 2.5);
    }
    
    if (t <= 2023) {
      return smooth_transition(t, 2021, 2023, TPT_min, TPT_max, 1.8);
    }
    
    real years_since_2023 = t - 2023;
    real future_effect = TPT_max + TPT_future_trend * years_since_2023;
    return fmax(0.0, fmin(2.0 * TPT_max, future_effect));
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

    // Unpack estimated parameters from theta array
    real theta_param_base = theta[1]; 
    real rho = theta[2]; 
    real w = theta[3];
    real alpha = theta[4]; 
    real f_base = theta[5]; 
    real s_base = theta[6];  
    real mu_sev_base = theta[7]; 
    real theta1_base = theta[8]; 
    real alpha_1 = theta[9]; 
    real f1_base = theta[10]; 
    real s1_base = theta[11]; 
    real mu_sev1_base = theta[12]; 
    
    real DOTS = theta[13];
    real ACT = theta[14];
    real TPT_c_max_pre2021 = theta[15]; 
    real TPT_a_max_pre2021 = theta[16]; 
    real beta = theta[17];
    real eta_base = theta[18];
    real eta1_base = theta[19];
    real TPT_min_fraction = theta[20];
    real TPT_future_trend_c = theta[21];  
    real TPT_future_trend_a = theta[22];
    real TPT_c_max_post2021 = theta[23];
    real TPT_a_max_post2021 = theta[24];

    // Unpack compartment states
    real S = y[1];   real L = y[2];   real As = y[3];  real Ac = y[4]; 
    real I = y[5];   real R = y[6];   real S1 = y[7];  real L1 = y[8];  
    real As1 = y[9]; real Ac1 = y[10]; real I1 = y[11]; real R1 = y[12];
    
    // Calculate intervention effects 
    real DOTS_effect = DOTS;
    real ACT_effect = ACT;
    real TPT_eff_c = get_smooth_TPT(t + 2000, TPT_c_max_pre2021, TPT_c_max_post2021, TPT_min_fraction, TPT_future_trend_c);
    real TPT_eff_a = get_smooth_TPT(t + 2000, TPT_a_max_pre2021, TPT_a_max_post2021, TPT_min_fraction, TPT_future_trend_a);
    
    // TPT reduces progression from latent to active (multiply by reduction factor)
    real theta_param = theta_param_base * (1.0 - TPT_eff_c);
    real theta1 = theta1_base * (1.0 - TPT_eff_a);
    
    // Ensure parameters stay within bounds
    theta_param = fmax(0.01, fmin(2.5, theta_param));
    theta1 = fmax(0.01, fmin(2.5, theta1));
    
    // Apply intervention effects to treatment outcomes
    real s = fmax(0.5, fmin(0.99, s_base + DOTS_effect + ACT_effect));
    real s1 = fmax(0.5, fmin(0.99, s1_base + DOTS_effect + ACT_effect));
    real f = fmax(0.001, fmin(0.1, f_base - DOTS_effect * 0.5 - ACT_effect * 0.5));
    real f1 = fmax(0.001, fmin(0.1, f1_base - DOTS_effect * 0.5 - ACT_effect * 0.5));
    
    // Treatment completion affects relapse rates
    real eta = fmax(0.001, fmin(0.2, eta_base - DOTS_effect * 0.3));
    real eta1 = fmax(0.001, fmin(0.2, eta1_base - DOTS_effect * 0.3));
    
    real mu_sev = mu_sev_base;
    real mu_sev1 = mu_sev1_base;
    
    // Population and transmission calculations
    real P_children = S + L + As + Ac + I + R;
    real P_adults = S1 + L1 + As1 + Ac1 + I1 + R1;
    real Total_pop = P_children + P_adults;
    
    real Total_infectious = c_Ac1 * Ac1 + c_As1 * As1;
    
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
  array[8] real x_r;  
  array[n_fit, 2] int observed_cases; 
}

transformed data {
  array[1] int x_i = { N_pop }; 
  array[n_fit] int observed_children_cases = observed_cases[, 1];
  array[n_fit] int observed_adults_cases = observed_cases[, 2];   
}

parameters {
  // Basic TB progression parameters
  real<lower=0.01, upper=0.8> theta_param_base;
  real<lower=0.01, upper=0.9> theta1_base;
  real<lower=0.0, upper=0.3> rho;
  real<lower=0.0, upper=0.15> w;
  real<lower=0.5, upper=8> beta; 
  real<lower=0.0, upper=0.4> alpha;  
  real<lower=0.0, upper=0.3> alpha_1; 
  real<lower=0.001, upper=0.15> eta_base; 
  real<lower=0.001, upper=0.15> eta1_base; 
  
  // Latent initial states
  real<lower=5000, upper=80000> L;
  real<lower=50000, upper=1000000> L1;  
  
  // Treatment outcomes as simplexes
  simplex[3] treatment_outcomes_children;
  simplex[3] treatment_outcomes_adults;
  
  // Intervention parameters 
  real<lower=0, upper=0.6> DOTS_effects;
  real<lower=0, upper=0.5> ACT_effects; 
  real<lower=0, upper=0.8> TPT_c_max_pre2021;
  real<lower=0, upper=0.95> TPT_a_max_pre2021;
  real<lower=0, upper=0.6> TPT_c_max_post2021;
  real<lower=0, upper=0.7> TPT_a_max_post2021;  
  
  // TPT dynamic parameters
  real<lower=0.05, upper=0.8> TPT_min_fraction;
  real<lower=-0.03, upper=0.03> TPT_future_trend_c;  
  real<lower=-0.03, upper=0.03> TPT_future_trend_a; 
}

transformed parameters {
  // Extract parameters from simplexes
  real f_base = treatment_outcomes_children[1] * 0.1;  // Scale appropriately
  real s_base = treatment_outcomes_children[2];
  real mu_sev_base = treatment_outcomes_children[3] * 0.05;
  real f1_base = treatment_outcomes_adults[1] * 0.1;
  real s1_base = treatment_outcomes_adults[2];
  real mu_sev1_base = treatment_outcomes_adults[3] * 0.05;

  vector[12] initial_state = y0;
  initial_state[2] = L;  
  initial_state[8] = L1;  
  
  vector[24] theta_ode = [theta_param_base, rho, w, alpha, f_base, s_base, mu_sev_base,
                          theta1_base, alpha_1, f1_base, s1_base, mu_sev1_base, DOTS_effects, ACT_effects, 
                          TPT_c_max_pre2021, TPT_a_max_pre2021, beta, eta_base, eta1_base, TPT_min_fraction,
                          TPT_future_trend_c, TPT_future_trend_a, TPT_c_max_post2021, TPT_a_max_post2021]';

  // Solve ODE for fitting period
  array[n_fit] vector[12] y_hat = ode_rk45(tb_ode, initial_state, t0, ts[1:n_fit], theta_ode, x_r, x_i);
}
 
model {
  // Improved priors
  theta_param_base ~ lognormal(log(0.3), 0.5);
  theta1_base ~ lognormal(log(0.5), 0.5);
  rho ~ lognormal(log(0.03), 0.5);
  w ~ lognormal(log(0.008), 0.3);
  beta ~ normal(2.5, 0.8);
  alpha ~ lognormal(log(0.02), 0.3);
  alpha_1 ~ lognormal(log(0.015), 0.3);
  eta_base ~ lognormal(log(0.03), 0.3);
  eta1_base ~ lognormal(log(0.025), 0.3);

  L ~ lognormal(log(11000), 0.2);
  L1 ~ lognormal(log(450000), 0.2);

  DOTS_effects ~ beta(3, 5); 
  ACT_effects ~ beta(3, 7);
  TPT_c_max_pre2021 ~ beta(7, 3);
  TPT_a_max_pre2021 ~ beta(9, 1);
  TPT_c_max_post2021 ~ beta(5, 5);
  TPT_a_max_post2021 ~ beta(4, 6);
  
  TPT_min_fraction ~ beta(2, 6);
  TPT_future_trend_c ~ normal(0, 0.008);
  TPT_future_trend_a ~ normal(0, 0.008);
  
  treatment_outcomes_children ~ dirichlet([2, 12, 1]);
  treatment_outcomes_adults ~ dirichlet([2, 12, 1]);
  
  // Likelihood with overdispersion handling
  for (n in 1:n_fit) {
    real predicted_children_cases = y_hat[n, 3] + y_hat[n, 4];
    real predicted_adults_cases = y_hat[n, 9] + y_hat[n, 10];
    
    observed_children_cases[n] ~ poisson(predicted_children_cases);
    observed_adults_cases[n] ~ poisson(predicted_adults_cases);
  }
}

generated quantities {
  // Full trajectory
  array[n_total] vector[12] y_full = ode_rk45(tb_ode, initial_state, t0, ts, theta_ode, x_r, x_i);
  
  // Predictions for children and adult cases
  array[n_total] real predicted_children_cases_full;
  array[n_total] real predicted_adults_cases_full;
  
  for (n in 1:n_total) {
    predicted_children_cases_full[n] = y_full[n, 3] + y_full[n, 4];
    predicted_adults_cases_full[n] = y_full[n, 9] + y_full[n, 10];
  }
}
