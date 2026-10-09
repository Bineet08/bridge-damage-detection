%% INFLUENCE-LINE-BASED BRIDGE DAMAGE DETECTION
% Static Euler-Bernoulli beam FE model with a moving unit load.
% Inverse model: delta_y ~= J*d, where d(e) is fractional EI loss in [0,1].
%
% MATLAB: R2018b or newer (uses xline and yline).
% No Optimization Toolbox is required.
%
% IMPORTANT:
% 1) This is a STATIC, linear-elastic model; it does not model bridge dynamics.
% 2) The inverse problem uses raw (unsmoothed) observations.
% 3) The damage indicator is plotted against LOAD POSITION. Its peak is not
%    automatically the physical damage location.
% 4) The inverse model is first-order (linearized about the healthy structure).
%    Large damage can cause linearization bias; validate severity estimates.

clear; clc; close all;
rng(42, "twister");

%% 1. MODEL AND SIMULATION SETTINGS
L = 10.0;                 % Beam span (m)
n_elem = 10;              % Number of Euler-Bernoulli beam elements
E = 2.1e11;               % Young's modulus (Pa)
I = 1.0e-4;               % Second moment of area (m^4)
EI = E * I;               % Healthy flexural rigidity (N m^2)

damage_elem = 5;          % Damaged element index: 1,...,n_elem
damage_severity = 0.30;   % Fractional EI loss: 0.30 means 30% loss

n_load_pts = 5*n_elem + 1;
x_load = linspace(0, L, n_load_pts);

% Sensors are specified in physical coordinates, not rounded node indices.
x_sensor = [0.25, 0.50, 0.75] * L;
sensor_names = {'Quarter-span', 'Midspan', 'Three-quarter-span'};

% Add independent Gaussian noise to both healthy and damaged measurements.
% sigma is set as a fraction of the theoretical midspan deflection under a
% unit point load. This is a simulation assumption, not a universal sensor model.
noise_level = 0.01;       % 1% of reference deflection

% Finite-difference step for dimensionless fractional-damage sensitivities.
delta_d = 1e-4;

%% 2. VALIDATE INPUTS
assert(L > 0 && n_elem >= 2 && n_elem == floor(n_elem), ...
    'L must be positive and n_elem must be an integer >= 2.');
assert(damage_elem >= 1 && damage_elem <= n_elem && ...
    damage_elem == floor(damage_elem), 'damage_elem is out of range.');
assert(damage_severity >= 0 && damage_severity <= 1, ...
    'damage_severity must be between 0 and 1.');
assert(all(x_sensor >= 0 & x_sensor <= L), ...
    'All sensor positions must lie on the beam.');

%% 3. COMPUTE HEALTHY AND DAMAGED INFLUENCE LINES
d_true = zeros(n_elem, 1);
d_true(damage_elem) = damage_severity;

EI_healthy = EI * ones(n_elem, 1);
EI_damaged = EI * (1 - d_true);

% Each output matrix is n_sensors-by-n_load_positions.
Y_healthy_clean = compute_influence_lines( ...
    EI_healthy, L, n_elem, x_load, x_sensor);
Y_damaged_clean = compute_influence_lines( ...
    EI_damaged, L, n_elem, x_load, x_sensor);

% Noise-free reference scale for a simply supported beam under a central
% point load: delta_max = P*L^3/(48*EI), with P = 1 N.
P_unit = 1.0;
reference_deflection = P_unit * L^3 / (48 * EI);
sigma_measurement = noise_level * reference_deflection;

Y_healthy_obs = Y_healthy_clean + ...
    sigma_measurement * randn(size(Y_healthy_clean));
Y_damaged_obs = Y_damaged_clean + ...
    sigma_measurement * randn(size(Y_damaged_clean));

% Flatten with MATLAB column-major order; b and every sensitivity column use the same order.
b = Y_damaged_obs(:) - Y_healthy_obs(:);
n_obs = numel(b);

% Difference of two independent measurement-noise terms has standard
% deviation sqrt(2)*sigma_measurement under the assumptions above.
sigma_difference = sqrt(2) * sigma_measurement;
morozov_target = sqrt(n_obs) * sigma_difference;



%% 4. FINITE-DIFFERENCE SENSITIVITY MATRIX
% J(:,e) = derivative of the stacked influence-line observations with
% respect to fractional EI loss d(e), evaluated at the healthy structure.
% Therefore b ~= J*d. The sign is determined by the FE response convention.
J = zeros(n_obs, n_elem);
for e = 1:n_elem
    d_pert = zeros(n_elem, 1);
    d_pert(e) = delta_d;
    EI_pert = EI * (1 - d_pert);

    Y_pert = compute_influence_lines(EI_pert, L, n_elem, x_load, x_sensor);
    J(:,e) = (Y_pert(:) - Y_healthy_clean(:)) / delta_d;
end

%% STEP 2: LINEARIZATION ERROR VS DAMAGE SEVERITY

test_severities = [0.05 0.10 0.15 0.20 0.30];

fprintf('\n--- STEP 2: SEVERITY SWEEP ---\n');
fprintf('Severity (%%) | Linearization error (%%)\n');
fprintf('-------------|-------------------------\n');

for k = 1:numel(test_severities)

    % Test damage at the same element, with a different severity
    d_test = zeros(n_elem, 1);
    d_test(damage_elem) = test_severities(k);

    EI_test = EI * (1 - d_test);

    % Actual noise-free response of this damaged beam
    Y_test = compute_influence_lines( ...
        EI_test, L, n_elem, x_load, x_sensor);

    b_actual_test = Y_test(:) - Y_healthy_clean(:);

    % First-order sensitivity prediction
    b_linear_test = J * d_test;

    actual_norm_test = norm(b_actual_test, 2);

    if actual_norm_test > 0
        error_test = 100 * ...
            norm(b_linear_test - b_actual_test, 2) ...
            / actual_norm_test;
    else
        error_test = NaN;
    end

    fprintf('%12.1f | %23.4f\n', ...
        100*test_severities(k), error_test);
end


%% STEP 1: CHECK THE LINEARIZATION ERROR

% Actual response change, using noise-free FE results
b_clean = Y_damaged_clean(:) - Y_healthy_clean(:);

% Response change predicted by the sensitivity matrix
b_linear = J * d_true;

% Calculate the relative error
actual_norm = norm(b_clean, 2);

if actual_norm == 0
    error('No response change detected. Check the damage configuration.');
end

linearization_error_pct = ...
    100 * norm(b_linear - b_clean, 2) / actual_norm;

fprintf('\n--- STEP 1: LINEARIZATION CHECK ---\n');
fprintf('Actual response-change norm: %.6e\n', actual_norm);
fprintf('Predicted response-change norm: %.6e\n', norm(b_linear, 2));
fprintf('Relative linearization error: %.4f %%\n', ...
    linearization_error_pct);


% Basic numerical diagnostics.
singular_values = svd(J, 0);
rank_tol = max(size(J)) * eps(max(singular_values));
rank_J = sum(singular_values > rank_tol);
if singular_values(end) > 0
    cond_J = singular_values(1) / singular_values(end);
else
    cond_J = Inf;
end

fprintf('\n--- FE / sensitivity diagnostics ---\n');
fprintf('Observations: %d | Elements/unknowns: %d\n', n_obs, n_elem);
fprintf('Numerical rank(J): %d of %d\n', rank_J, n_elem);
fprintf('cond(J): %.4e\n', cond_J);
fprintf('Assumed SD of difference noise: %.4e m/N\n', sigma_difference);
fprintf('Morozov discrepancy target: %.4e m/N\n', morozov_target);

%% 5. CHOOSE TIKHONOV REGULARIZATION USING A DISCREPANCY TARGET
% Solve min_d ||J*d-b||_2^2 + lambda*||d||_2^2 subject to 0 <= d <= 1.
% Projected FISTA is used so no Optimization Toolbox is needed.
smax = singular_values(1);
lambda_ref = max(smax^2, eps);
lambda_grid = lambda_ref * logspace(-8, 2, 100);
residual_norm = zeros(size(lambda_grid));
d_grid = zeros(n_elem, numel(lambda_grid));

for k = 1:numel(lambda_grid)
    d_grid(:,k) = solve_box_tikhonov(J, b, lambda_grid(k), 10000, 1e-9);
    residual_norm(k) = norm(J*d_grid(:,k) - b, 2);
end

cross_idx = find(residual_norm >= morozov_target, 1, 'first');
if ~isempty(cross_idx)
    chosen_idx = cross_idx;
    lambda_status = 'first discrepancy crossing found';
elseif all(residual_norm < morozov_target)
    % Residual never reaches target over this lambda range.
    chosen_idx = numel(lambda_grid);
    lambda_status = ['target not reached; selected largest tested lambda. ' ...
        'Inspect the residual curve and noise/model assumptions.'];
else
    % Even the least-regularized solution is above target, likely due to
    % linearization error, data mismatch, or a target that is too small.
    chosen_idx = 1;
    lambda_status = ['residual already above target at smallest tested lambda. ' ...
        'Inspect linearization error, noise assumptions, and lambda range.'];
end

d_est = d_grid(:, chosen_idx);
lambda_selected = lambda_grid(chosen_idx);
final_residual = residual_norm(chosen_idx);

%% STEP 3: COMPARE NOISY AND NOISE-FREE INVERSE RESULTS

% Use the same regularization strength for both comparisons
d_est_noise_free = solve_box_tikhonov( ...
    J, b_clean, lambda_selected, 10000, 1e-9);

fprintf('\n--- STEP 3: EFFECT OF MEASUREMENT NOISE ---\n');
fprintf('Element | True (%%) | Noisy estimate (%%) | Noise-free estimate (%%)\n');
fprintf('--------|----------|--------------------|------------------------\n');

for e = 1:n_elem
    fprintf('%7d | %8.2f | %18.2f | %22.2f\n', ...
        e, 100*d_true(e), ...
        100*d_est(e), ...
        100*d_est_noise_free(e));
end



%% STEP 4: NONLINEAR SEVERITY ESTIMATION
% Controlled test: the damaged element is assumed to be known.

severity_grid = 0:0.005:0.80;

fit_residual_noisy = zeros(size(severity_grid));
fit_residual_clean = zeros(size(severity_grid));

for k = 1:numel(severity_grid)

    % Candidate fractional stiffness loss
    d_try = zeros(n_elem, 1);
    d_try(damage_elem) = severity_grid(k);

    EI_try = EI * (1 - d_try);

    % Full nonlinear FE forward model
    Y_try = compute_influence_lines( ...
        EI_try, L, n_elem, x_load, x_sensor);

    % Predicted response change relative to the healthy structure
    b_try = Y_try(:) - Y_healthy_clean(:);

    % Compare with noisy and noise-free observations
    fit_residual_noisy(k) = norm(b_try - b, 2);
    fit_residual_clean(k) = norm(b_try - b_clean, 2);
end

% Best-fitting severity for each dataset
[best_residual_noisy, idx_noisy] = min(fit_residual_noisy);
[best_residual_clean, idx_clean] = min(fit_residual_clean);

best_severity_noisy = severity_grid(idx_noisy);
best_severity_clean = severity_grid(idx_clean);

fprintf('\n--- STEP 4: NONLINEAR SEVERITY FIT ---\n');
fprintf('Known damaged element: %d\n', damage_elem);
fprintf('True simulated severity: %.2f %%\n', ...
    100 * damage_severity);

fprintf('\nNoise-free fit:\n');
fprintf('Estimated severity: %.2f %%\n', ...
    100 * best_severity_clean);
fprintf('Residual: %.6e m/N\n', best_residual_clean);

fprintf('\nNoisy fit:\n');
fprintf('Estimated severity: %.2f %%\n', ...
    100 * best_severity_noisy);
fprintf('Residual: %.6e m/N\n', best_residual_noisy);

figure('Name', 'Nonlinear severity fit', 'Color', 'w');

plot(100 * severity_grid, fit_residual_clean, ...
    'LineWidth', 1.5);
hold on;

plot(100 * severity_grid, fit_residual_noisy, ...
    'LineWidth', 1.5);

plot(100 * best_severity_clean, best_residual_clean, ...
    'o', 'MarkerSize', 7);

plot(100 * best_severity_noisy, best_residual_noisy, ...
    's', 'MarkerSize', 7);

grid on;
xlabel('Candidate stiffness loss (%)');
ylabel('Response-fit residual (m/N)');
title('Nonlinear severity estimation with known damage location');

legend('Noise-free residual', 'Noisy residual', ...
    'Best noise-free fit', 'Best noisy fit', ...
    'Location', 'best');


%% STEP 5: NONLINEAR DAMAGE LOCALIZATION
% Assume there is exactly one damaged element, but its location is unknown.
% Search every element and estimate the best severity using the full FE model.

coarse_severity_grid = 0:0.02:0.80;

best_severity_location = zeros(n_elem, 1);
best_residual_location = inf(n_elem, 1);

% First, perform a coarse search for each possible damaged element.
for e = 1:n_elem

    for k = 1:numel(coarse_severity_grid)

        d_try = zeros(n_elem, 1);
        d_try(e) = coarse_severity_grid(k);

        EI_try = EI * (1 - d_try);

        Y_try = compute_influence_lines( ...
            EI_try, L, n_elem, x_load, x_sensor);

        b_try = Y_try(:) - Y_healthy_clean(:);
        residual_try = norm(b_try - b, 2);

        if residual_try < best_residual_location(e)
            best_residual_location(e) = residual_try;
            best_severity_location(e) = coarse_severity_grid(k);
        end
    end
end

% Refine the severity estimate around the best coarse result for each
% possible damaged element.
for e = 1:n_elem

    lower_severity = max(0, ...
        best_severity_location(e) - 0.02);

    upper_severity = min(0.80, ...
        best_severity_location(e) + 0.02);

    fine_severity_grid = linspace( ...
        lower_severity, upper_severity, 41);

    for k = 1:numel(fine_severity_grid)

        d_try = zeros(n_elem, 1);
        d_try(e) = fine_severity_grid(k);

        EI_try = EI * (1 - d_try);

        Y_try = compute_influence_lines( ...
            EI_try, L, n_elem, x_load, x_sensor);

        b_try = Y_try(:) - Y_healthy_clean(:);
        residual_try = norm(b_try - b, 2);

        if residual_try < best_residual_location(e)
            best_residual_location(e) = residual_try;
            best_severity_location(e) = fine_severity_grid(k);
        end
    end
end

% Rank candidate locations by their best response-fit residual.
[sorted_residuals, sorted_elements] = ...
    sort(best_residual_location, 'ascend');

fprintf('\n--- STEP 5: UNKNOWN-LOCATION DAMAGE IDENTIFICATION ---\n');
fprintf('Assumption: exactly one damaged element\n');
fprintf('Morozov discrepancy target: %.6e m/N\n', ...
    morozov_target);

fprintf('\nRank | Element | Best damage estimate (%%) | Residual (m/N)\n');
fprintf('-----|---------|--------------------------|---------------\n');

for r = 1:n_elem
    e = sorted_elements(r);

    fprintf('%4d | %7d | %24.2f | %.6e\n', ...
        r, e, 100*best_severity_location(e), ...
        sorted_residuals(r));
end

best_element = sorted_elements(1);
best_damage_severity = best_severity_location(best_element);

fprintf('\nBest candidate element: %d\n', best_element);
fprintf('Estimated damage location: %.3f m (element midpoint)\n', ...
    (best_element - 0.5) * L/n_elem);
fprintf('Estimated stiffness loss: %.2f %%\n', ...
    100*best_damage_severity);
fprintf('Best response-fit residual: %.6e m/N\n', ...
    sorted_residuals(1));


%% STEP 6: MONTE CARLO ROBUSTNESS TEST

n_mc = 30;
mc_severity_grid = 0.02:0.02:0.80;

% Precompute the nonlinear response for each candidate location/severity.
% This avoids rerunning the FE model for every noise realization.
n_candidates = n_elem * numel(mc_severity_grid);
candidate_responses = zeros(n_obs, n_candidates);
candidate_elements = zeros(1, n_candidates);
candidate_severities = zeros(1, n_candidates);

c = 0;

for e = 1:n_elem
    for k = 1:numel(mc_severity_grid)

        c = c + 1;

        d_candidate = zeros(n_elem, 1);
        d_candidate(e) = mc_severity_grid(k);

        EI_candidate = EI * (1 - d_candidate);

        Y_candidate = compute_influence_lines( ...
            EI_candidate, L, n_elem, x_load, x_sensor);

        candidate_responses(:,c) = ...
            Y_candidate(:) - Y_healthy_clean(:);

        candidate_elements(c) = e;
        candidate_severities(c) = mc_severity_grid(k);
    end
end

% Results from repeated independent noise realizations
mc_best_element = zeros(n_mc, 1);
mc_best_severity = zeros(n_mc, 1);
mc_best_residual = zeros(n_mc, 1);

for r = 1:n_mc

    % Reproducible but different noise for each trial
    rng(1000 + r, 'twister');

    Y_h_mc = Y_healthy_clean + ...
        sigma_measurement * randn(size(Y_healthy_clean));

    Y_d_mc = Y_damaged_clean + ...
        sigma_measurement * randn(size(Y_damaged_clean));

    b_mc = Y_d_mc(:) - Y_h_mc(:);

    % Compare all precomputed candidates to this noisy observation
    residuals_mc = sqrt(sum( ...
        (candidate_responses - b_mc).^2, 1));

    [mc_best_residual(r), idx] = min(residuals_mc);

    mc_best_element(r) = candidate_elements(idx);
    mc_best_severity(r) = candidate_severities(idx);
end

% Summarize performance
correct_location = (mc_best_element == damage_elem);
localization_rate = 100 * mean(correct_location);

fprintf('\n--- STEP 6: MONTE CARLO RESULTS ---\n');
fprintf('Number of trials: %d\n', n_mc);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('True severity: %.2f %%\n', 100*damage_severity);
fprintf('Correct localization: %d / %d (%.1f %%)\n', ...
    sum(correct_location), n_mc, localization_rate);

fprintf('\nLocalization frequency by element:\n');
for e = 1:n_elem
    count_e = sum(mc_best_element == e);
    if count_e > 0
        fprintf('Element %2d: %2d / %d trials\n', ...
            e, count_e, n_mc);
    end
end

if any(correct_location)
    correct_severities = 100 * mc_best_severity(correct_location);

    fprintf('\nSeverity, for correctly localized trials:\n');
    fprintf('Mean: %.2f %%\n', mean(correct_severities));
    fprintf('Standard deviation: %.2f percentage points\n', ...
        std(correct_severities));
else
    fprintf('\nNo trials correctly localized the damaged element.\n');
end

fprintf('\nMean estimated severity across all trials: %.2f %%\n', ...
    100 * mean(mc_best_severity));

% Plot localization frequency
figure('Name', 'Monte Carlo localization robustness', 'Color', 'w');
counts = arrayfun(@(e) sum(mc_best_element == e), 1:n_elem);
bar(1:n_elem, counts);
grid on;
xlabel('Estimated damaged element');
ylabel('Number of trials');
title(sprintf('Single-damage localization over %d noise realizations', n_mc));
xticks(1:n_elem);


%% STEP 7: TWO-DAMAGE NONLINEAR IDENTIFICATION

% Controlled test: exactly two elements are damaged.
% Their locations are unknown to the search algorithm.

multi_true_elements = [4, 7];
multi_true_severities = [0.20, 0.35];

d_multi_true = zeros(n_elem, 1);
d_multi_true(multi_true_elements) = multi_true_severities;

EI_multi_true = EI * (1 - d_multi_true);

% Generate the actual noise-free response for the two-damage case.
Y_multi_clean = compute_influence_lines( ...
    EI_multi_true, L, n_elem, x_load, x_sensor);

% Generate independent synthetic noise for healthy and damaged responses.
rng(2026, 'twister');

Y_multi_healthy_obs = Y_healthy_clean + ...
    sigma_measurement * randn(size(Y_healthy_clean));

Y_multi_damaged_obs = Y_multi_clean + ...
    sigma_measurement * randn(size(Y_multi_clean));

b_multi_obs = Y_multi_damaged_obs(:) - Y_multi_healthy_obs(:);

% Enumerate every unordered pair of distinct elements.
multi_pairs = nchoosek(1:n_elem, 2);
n_multi_pairs = size(multi_pairs, 1);

% Coarse severity search: 10%, 20%, ..., 60%.
% The 10% lower bound enforces the assumption of exactly two
% nonzero damaged elements in this controlled test.
multi_coarse_severities = 0.05:0.05:0.60;

best_coarse_residual = inf(n_multi_pairs, 1);
best_coarse_s1 = zeros(n_multi_pairs, 1);
best_coarse_s2 = zeros(n_multi_pairs, 1);

fprintf('\n--- STEP 7: COARSE TWO-DAMAGE SEARCH ---\n');

for p = 1:n_multi_pairs

    e1 = multi_pairs(p, 1);
    e2 = multi_pairs(p, 2);

    for a = 1:numel(multi_coarse_severities)
        for sev2_idx = 1:numel(multi_coarse_severities)

            s1 = multi_coarse_severities(a);
            s2 = multi_coarse_severities(sev2_idx);

            d_candidate = zeros(n_elem, 1);
            d_candidate(e1) = s1;
            d_candidate(e2) = s2;

            EI_candidate = EI * (1 - d_candidate);

            Y_candidate = compute_influence_lines( ...
                EI_candidate, L, n_elem, x_load, x_sensor);

            b_candidate = Y_candidate(:) - Y_healthy_clean(:);

            residual_candidate = norm( ...
                b_candidate - b_multi_obs, 2);

            if residual_candidate < best_coarse_residual(p)
                best_coarse_residual(p) = residual_candidate;
                best_coarse_s1(p) = s1;
                best_coarse_s2(p) = s2;
            end
        end
    end
end

% Refine the severity estimates for the three best coarse element pairs.
[~, coarse_order] = sort(best_coarse_residual, 'ascend');
n_refine_pairs = min(5, n_multi_pairs);

refined_results = zeros(n_refine_pairs, 5);

for r = 1:n_refine_pairs

    p = coarse_order(r);
    e1 = multi_pairs(p, 1);
    e2 = multi_pairs(p, 2);

    % Search +/-10 percentage points around the coarse estimates,
    % using 1-percentage-point increments.
    low1 = max(1, round((best_coarse_s1(p) - 0.10) * 100));
    high1 = min(80, round((best_coarse_s1(p) + 0.10) * 100));

    low2 = max(1, round((best_coarse_s2(p) - 0.10) * 100));
    high2 = min(80, round((best_coarse_s2(p) + 0.10) * 100));

    fine_s1_grid = (low1:high1) / 100;
    fine_s2_grid = (low2:high2) / 100;

    local_best_residual = Inf;
    local_best_s1 = NaN;
    local_best_s2 = NaN;

    for a = 1:numel(fine_s1_grid)
        for fine_s2_idx = 1:numel(fine_s2_grid)

            s1 = fine_s1_grid(a);
            s2 = fine_s2_grid(fine_s2_idx);

            d_candidate = zeros(n_elem, 1);
            d_candidate(e1) = s1;
            d_candidate(e2) = s2;

            EI_candidate = EI * (1 - d_candidate);

            Y_candidate = compute_influence_lines( ...
                EI_candidate, L, n_elem, x_load, x_sensor);

            b_candidate = Y_candidate(:) - Y_healthy_clean(:);

            residual_candidate = norm( ...
                b_candidate - b_multi_obs, 2);

            if residual_candidate < local_best_residual
                local_best_residual = residual_candidate;
                local_best_s1 = s1;
                local_best_s2 = s2;
            end
        end
    end

    refined_results(r, :) = [ ...
        e1, e2, local_best_s1, local_best_s2, local_best_residual];
end

% Rank the refined candidates by their response-fit residual.
[~, refined_order] = sort(refined_results(:, 5), 'ascend');

fprintf('\n--- STEP 7: REFINED TWO-DAMAGE RESULTS ---\n');
fprintf('True damaged elements: %d and %d\n', ...
    multi_true_elements(1), multi_true_elements(2));

fprintf('True stiffness losses: %.1f%% and %.1f%%\n', ...
    100*multi_true_severities(1), ...
    100*multi_true_severities(2));

fprintf('\nRank | Elements | Estimated losses (%%) | Residual (m/N)\n');
fprintf('-----|----------|-----------------------|---------------\n');

for r = 1:n_refine_pairs
    row = refined_results(refined_order(r), :);

    fprintf('%4d | %d and %d | %7.2f, %7.2f       | %.6e\n', ...
        r, round(row(1)), round(row(2)), ...
        100*row(3), 100*row(4), row(5));
end

best_row = refined_results(refined_order(1), :);

fprintf('\nBest candidate elements: %d and %d\n', ...
    round(best_row(1)), round(best_row(2)));

fprintf('Estimated stiffness losses: %.2f%% and %.2f%%\n', ...
    100*best_row(3), 100*best_row(4));

fprintf('Best response-fit residual: %.6e m/N\n', best_row(5));
fprintf('Single-damage noise discrepancy target: %.6e m/N\n', ...
    morozov_target);


%% STEP 8: MONTE CARLO ROBUSTNESS FOR TWO-DAMAGE IDENTIFICATION

n_mc_multi = 30;

% Candidate severities: 5% increments from 5% to 60%.
% This is a discrete search; severity estimates have 5% resolution.
mc_severity_grid = 0.05:0.05:0.60;

mc_pairs = nchoosek(1:n_elem, 2);
n_pairs_mc = size(mc_pairs, 1);
n_sev_mc = numel(mc_severity_grid);

n_candidates_mc = n_pairs_mc * n_sev_mc^2;

% Precompute every candidate response.
response_bank_mc = zeros(n_obs, n_candidates_mc);
candidate_e1_mc = zeros(n_candidates_mc, 1);
candidate_e2_mc = zeros(n_candidates_mc, 1);
candidate_s1_mc = zeros(n_candidates_mc, 1);
candidate_s2_mc = zeros(n_candidates_mc, 1);

idx = 0;

fprintf('\nPrecomputing two-damage candidate responses...\n');

for p = 1:n_pairs_mc

    e1 = mc_pairs(p, 1);
    e2 = mc_pairs(p, 2);

    for a = 1:n_sev_mc
        for c = 1:n_sev_mc

            idx = idx + 1;

            s1 = mc_severity_grid(a);
            s2 = mc_severity_grid(c);

            d_candidate = zeros(n_elem, 1);
            d_candidate(e1) = s1;
            d_candidate(e2) = s2;

            EI_candidate = EI * (1 - d_candidate);

            Y_candidate = compute_influence_lines( ...
                EI_candidate, L, n_elem, x_load, x_sensor);

            response_bank_mc(:, idx) = ...
                Y_candidate(:) - Y_healthy_clean(:);

            candidate_e1_mc(idx) = e1;
            candidate_e2_mc(idx) = e2;
            candidate_s1_mc(idx) = s1;
            candidate_s2_mc(idx) = s2;
        end
    end
end

% Store the best candidate for each independent noise realization.
mc_pair_e1 = zeros(n_mc_multi, 1);
mc_pair_e2 = zeros(n_mc_multi, 1);
mc_pair_s1 = zeros(n_mc_multi, 1);
mc_pair_s2 = zeros(n_mc_multi, 1);
mc_pair_residual = zeros(n_mc_multi, 1);

for r = 1:n_mc_multi

    rng(5000 + r, 'twister');

    % Independent measurement noise on healthy and damaged observations
    Yh_trial = Y_healthy_clean + ...
        sigma_measurement * randn(size(Y_healthy_clean));

    Yd_trial = Y_multi_clean + ...
        sigma_measurement * randn(size(Y_multi_clean));

    b_trial = Yd_trial(:) - Yh_trial(:);

    % Fit every precomputed two-damage candidate
    residuals_trial = sqrt(sum( ...
        (response_bank_mc - b_trial).^2, 1));

    [mc_pair_residual(r), best_idx] = min(residuals_trial);

    mc_pair_e1(r) = candidate_e1_mc(best_idx);
    mc_pair_e2(r) = candidate_e2_mc(best_idx);
    mc_pair_s1(r) = candidate_s1_mc(best_idx);
    mc_pair_s2(r) = candidate_s2_mc(best_idx);
end

% A pair is correct only if both damaged elements are identified.
correct_pair_mc = ...
    (mc_pair_e1 == multi_true_elements(1)) & ...
    (mc_pair_e2 == multi_true_elements(2));

pair_success_rate = 100 * mean(correct_pair_mc);

fprintf('\n--- STEP 8: TWO-DAMAGE MONTE CARLO RESULTS ---\n');
fprintf('Trials: %d\n', n_mc_multi);
fprintf('True elements: %d and %d\n', ...
    multi_true_elements(1), multi_true_elements(2));
fprintf('True severities: %.1f%% and %.1f%%\n', ...
    100*multi_true_severities(1), ...
    100*multi_true_severities(2));

fprintf('Correct pair identification: %d/%d (%.1f%%)\n', ...
    sum(correct_pair_mc), n_mc_multi, pair_success_rate);

fprintf('\nCandidate-pair frequency:\n');
for p = 1:n_pairs_mc

    e1 = mc_pairs(p, 1);
    e2 = mc_pairs(p, 2);

    count_pair = sum( ...
        (mc_pair_e1 == e1) & (mc_pair_e2 == e2));

    if count_pair > 0
        fprintf('Elements %d and %d: %d/%d trials\n', ...
            e1, e2, count_pair, n_mc_multi);
    end
end

if any(correct_pair_mc)

    s1_correct_pct = 100 * mc_pair_s1(correct_pair_mc);
    s2_correct_pct = 100 * mc_pair_s2(correct_pair_mc);

    fprintf('\nSeverity estimates when the pair is correct:\n');

    fprintf('Element %d: mean %.2f%%, SD %.2f percentage points\n', ...
        multi_true_elements(1), ...
        mean(s1_correct_pct), std(s1_correct_pct));

    fprintf('Element %d: mean %.2f%%, SD %.2f percentage points\n', ...
        multi_true_elements(2), ...
        mean(s2_correct_pct), std(s2_correct_pct));

else
    fprintf('\nNo trial identified both elements correctly.\n');
end

% Plot the frequency of identified pairs.
figure('Name', 'Two-damage Monte Carlo robustness', 'Color', 'w');

pair_counts = zeros(n_pairs_mc, 1);
for p = 1:n_pairs_mc
    pair_counts(p) = sum( ...
        (mc_pair_e1 == mc_pairs(p,1)) & ...
        (mc_pair_e2 == mc_pairs(p,2)));
end

bar(pair_counts);
grid on;
xlabel('Candidate element-pair index');
ylabel('Number of trials');
title(sprintf('Two-damage localization over %d noise realizations', ...
    n_mc_multi));
yline(n_mc_multi, '--');


%% STEP 9: MULTIPLE-DAMAGE LOCALIZATION VS NOISE LEVEL

noise_levels_stress = [0.01, 0.02, 0.05];
n_stress_levels = numel(noise_levels_stress);

stress_best_e1 = zeros(n_stress_levels, n_mc_multi);
stress_best_e2 = zeros(n_stress_levels, n_mc_multi);
stress_best_s1 = zeros(n_stress_levels, n_mc_multi);
stress_best_s2 = zeros(n_stress_levels, n_mc_multi);

fprintf('\n--- STEP 9: NOISE-LEVEL STRESS TEST ---\n');
fprintf('Trials per noise level: %d\n', n_mc_multi);
fprintf('True damaged elements: %d and %d\n', ...
    multi_true_elements(1), multi_true_elements(2));

for q = 1:n_stress_levels

    current_noise_level = noise_levels_stress(q);
    sigma_trial = current_noise_level * reference_deflection;

    for r = 1:n_mc_multi

        % Reproducible, independent noise for every trial and noise level.
        rng(9000 + 1000*q + r, 'twister');

        Yh_trial = Y_healthy_clean + ...
            sigma_trial * randn(size(Y_healthy_clean));

        Yd_trial = Y_multi_clean + ...
            sigma_trial * randn(size(Y_multi_clean));

        b_trial = Yd_trial(:) - Yh_trial(:);

        % Fit the existing candidate bank to this noisy observation.
        residuals_trial = sqrt(sum( ...
            (response_bank_mc - b_trial).^2, 1));

        [~, best_idx] = min(residuals_trial);

        stress_best_e1(q,r) = candidate_e1_mc(best_idx);
        stress_best_e2(q,r) = candidate_e2_mc(best_idx);
        stress_best_s1(q,r) = candidate_s1_mc(best_idx);
        stress_best_s2(q,r) = candidate_s2_mc(best_idx);

    end
end

fprintf('\nNoise level | Correct pairs | Localization rate\n');
fprintf('------------|---------------|------------------\n');

stress_success_rate = zeros(n_stress_levels, 1);

for q = 1:n_stress_levels

    correct_pair = ...
        (stress_best_e1(q,:) == multi_true_elements(1)) & ...
        (stress_best_e2(q,:) == multi_true_elements(2));

    stress_success_rate(q) = 100 * mean(correct_pair);

    fprintf('%10.1f%% | %5d / %d     | %15.1f%%\n', ...
        100*noise_levels_stress(q), ...
        sum(correct_pair), n_mc_multi, ...
        stress_success_rate(q));

    if any(correct_pair)
        fprintf('  Mean severities when correctly localized: %.1f%%, %.1f%%\n', ...
            100*mean(stress_best_s1(q,correct_pair)), ...
            100*mean(stress_best_s2(q,correct_pair)));
    end
end

figure('Name', 'Localization vs measurement noise', 'Color', 'w');
plot(100*noise_levels_stress, stress_success_rate, ...
    '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('Measurement noise level (% of reference deflection)');
ylabel('Correct pair identification (%)');
title('Two-damage localization robustness');
ylim([0 105]);


%% STEP 10: SENSOR-PLACEMENT OPTIMIZATION

% Test three sensors at distinct candidate positions, excluding supports.
sensor_positions_test = 0.5:0.5:(L - 0.5);
n_sensors_test = 3;
n_candidate_sensors = numel(sensor_positions_test);

sensor_combinations = nchoosek( ...
    1:n_candidate_sensors, n_sensors_test);

n_layouts = size(sensor_combinations, 1);

fprintf('\n--- STEP 10: SENSOR-PLACEMENT OPTIMIZATION ---\n');
fprintf('Candidate sensor positions: %d\n', n_candidate_sensors);
fprintf('Three-sensor layouts tested: %d\n', n_layouts);

% Compute healthy influence lines at every candidate sensor position.
Y_healthy_all = compute_influence_lines( ...
    EI_healthy, L, n_elem, x_load, sensor_positions_test);

% Build one sensitivity matrix containing all candidate sensor responses.
J_all = zeros(n_candidate_sensors*n_load_pts, n_elem);

for e = 1:n_elem

    d_pert = zeros(n_elem, 1);
    d_pert(e) = delta_d;

    EI_pert = EI * (1 - d_pert);

    Y_pert_all = compute_influence_lines( ...
        EI_pert, L, n_elem, x_load, sensor_positions_test);

    J_all(:,e) = ...
        (Y_pert_all(:) - Y_healthy_all(:)) / delta_d;
end

% Evaluate conditioning and sensitivity for each sensor layout.
layout_condition = inf(n_layouts, 1);
layout_sigma_min = zeros(n_layouts, 1);
layout_rank = zeros(n_layouts, 1);

for q = 1:n_layouts

    selected_sensors = sensor_combinations(q,:);

    % Stack each selected sensor's complete influence line.

    selected_rows = [];

    for s = selected_sensors
        % Y_all(:) stores all sensors at each load position.
        % Select sensor s at every load position.
        sensor_rows = s:n_candidate_sensors: ...
            (s + (n_load_pts - 1)*n_candidate_sensors);

        selected_rows = [selected_rows, sensor_rows];
    end


    J_layout = J_all(selected_rows, :);
    sv = svd(J_layout, 0);

    tol_rank = max(size(J_layout)) * eps(sv(1));
    layout_rank(q) = sum(sv > tol_rank);
    layout_sigma_min(q) = sv(end);

    if layout_rank(q) == n_elem && sv(end) > 0
        layout_condition(q) = sv(1) / sv(end);
    end
end

% Find the current layout's position in the candidate set.
current_sensor_indices = zeros(1, n_sensors_test);

for s = 1:n_sensors_test
    [~, current_sensor_indices(s)] = ...
        min(abs(sensor_positions_test - x_sensor(s)));
end

[baseline_found, baseline_idx] = ismember( ...
    sort(current_sensor_indices), sensor_combinations, 'rows');

fprintf('\nCurrent layout: [%.2f, %.2f, %.2f] m\n', ...
    x_sensor(1), x_sensor(2), x_sensor(3));

if baseline_found
    fprintf('Current layout rank: %d/%d\n', ...
        layout_rank(baseline_idx), n_elem);
    fprintf('Current condition number: %.4e\n', ...
        layout_condition(baseline_idx));
    fprintf('Current minimum singular value: %.4e\n', ...
        layout_sigma_min(baseline_idx));
end

% Rank layouts by minimum condition number.
[~, order_condition] = sort(layout_condition, 'ascend');

fprintf('\n--- TOP 10: LOWEST CONDITION NUMBER ---\n');
fprintf('Rank | Sensor positions (m) | cond(J) | sigma_min(J)\n');

for r = 1:min(10, n_layouts)

    q = order_condition(r);
    positions = sensor_positions_test(sensor_combinations(q,:));

    fprintf('%4d | [%4.1f, %4.1f, %4.1f] | %.4e | %.4e\n', ...
        r, positions(1), positions(2), positions(3), ...
        layout_condition(q), layout_sigma_min(q));
end

% Rank layouts by largest minimum singular value.
[~, order_sigma] = sort(layout_sigma_min, 'descend');

fprintf('\n--- TOP 10: LARGEST MINIMUM SINGULAR VALUE ---\n');
fprintf('Rank | Sensor positions (m) | sigma_min(J) | cond(J)\n');

for r = 1:min(10, n_layouts)

    q = order_sigma(r);
    positions = sensor_positions_test(sensor_combinations(q,:));

    fprintf('%4d | [%4.1f, %4.1f, %4.1f] | %.4e | %.4e\n', ...
        r, positions(1), positions(2), positions(3), ...
        layout_sigma_min(q), layout_condition(q));
end


%% STEP 11: VALIDATE THE ALTERNATIVE SENSOR LAYOUT

% Compare baseline and the candidate with the largest minimum singular value.
layouts_step11 = [
    2.5, 5.0, 7.5
    0.5, 5.0, 9.5
];

layout_labels = {
    'Baseline [2.5, 5.0, 7.5]'
    'Candidate [0.5, 5.0, 9.5]'
};

n_layouts_step11 = size(layouts_step11, 1);

healthy_step11 = cell(n_layouts_step11, 1);
multi_step11 = cell(n_layouts_step11, 1);
bank_step11 = cell(n_layouts_step11, 1);

% Baseline: reuse the existing model responses and candidate bank.
healthy_step11{1} = Y_healthy_clean;
multi_step11{1} = Y_multi_clean;
bank_step11{1} = response_bank_mc;

% Candidate sensor layout.
x_sensor_candidate = layouts_step11(2,:);

healthy_step11{2} = compute_influence_lines( ...
    EI_healthy, L, n_elem, x_load, x_sensor_candidate);

EI_multi_true_step11 = EI * (1 - d_multi_true);

multi_step11{2} = compute_influence_lines( ...
    EI_multi_true_step11, L, n_elem, x_load, x_sensor_candidate);

% Precompute every two-damage candidate for the alternative layout.
bank_step11{2} = zeros(n_obs, n_candidates_mc);

fprintf('\nPrecomputing candidate-layout responses...\n');

for c = 1:n_candidates_mc

    d_candidate = zeros(n_elem, 1);
    d_candidate(candidate_e1_mc(c)) = candidate_s1_mc(c);
    d_candidate(candidate_e2_mc(c)) = candidate_s2_mc(c);

    EI_candidate = EI * (1 - d_candidate);

    Y_candidate = compute_influence_lines( ...
        EI_candidate, L, n_elem, x_load, x_sensor_candidate);

    bank_step11{2}(:,c) = ...
        Y_candidate(:) - healthy_step11{2}(:);
end

% Compare both layouts under identical noise realizations.
correct_rate_step11 = zeros(n_layouts_step11, ...
    numel(noise_levels_stress));

fprintf('\n--- STEP 11: SENSOR LAYOUT VALIDATION ---\n');
fprintf('True damaged elements: %d and %d\n', ...
    multi_true_elements(1), multi_true_elements(2));

fprintf('\nLayout | Noise | Correct pairs | Success rate\n');
fprintf('-------|-------|---------------|-------------\n');

for layout_id = 1:n_layouts_step11

    for q = 1:numel(noise_levels_stress)

        sigma_trial = noise_levels_stress(q) * reference_deflection;
        correct_count = 0;

        for r = 1:n_mc_multi

            % Match Step 9 seeds to enable a fair baseline comparison.
            rng(9000 + 1000*q + r, 'twister');

            Yh_trial = healthy_step11{layout_id} + ...
                sigma_trial * randn(size(healthy_step11{layout_id}));

            Yd_trial = multi_step11{layout_id} + ...
                sigma_trial * randn(size(multi_step11{layout_id}));

            b_trial = Yd_trial(:) - Yh_trial(:);

            residuals_trial = sqrt(sum( ...
                (bank_step11{layout_id} - b_trial).^2, 1));

            [~, best_idx] = min(residuals_trial);

            pair_is_correct = ...
                candidate_e1_mc(best_idx) == multi_true_elements(1) && ...
                candidate_e2_mc(best_idx) == multi_true_elements(2);

            correct_count = correct_count + pair_is_correct;
        end

        correct_rate_step11(layout_id,q) = ...
            100 * correct_count / n_mc_multi;

        fprintf('%5d | %4.1f%% | %5d / %d     | %10.1f%%\n', ...
            layout_id, 100*noise_levels_stress(q), ...
            correct_count, n_mc_multi, ...
            correct_rate_step11(layout_id,q));
    end
end

% Plot the measured localization rates.
figure('Name', 'Sensor layout validation', 'Color', 'w');

plot(100*noise_levels_stress, correct_rate_step11(1,:), ...
    '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
hold on;

plot(100*noise_levels_stress, correct_rate_step11(2,:), ...
    '-s', 'LineWidth', 1.5, 'MarkerSize', 7);

grid on;
xlabel('Measurement noise level (% of reference deflection)');
ylabel('Correct two-element localization (%)');
title('Baseline versus alternative sensor placement');
legend(layout_labels, 'Location', 'best');
ylim([0 105]);


%% STEP 12: DIRECT SENSOR-LAYOUT COMPARISON

layouts_step12 = [
    2.5, 5.0, 7.5;   % Baseline
    0.5, 5.0, 9.5;   % Highest sigma_min from Step 10
    1.0, 9.0, 9.5;   % Lowest condition number from Step 10
    1.5, 4.5, 9.5    % Second-highest sigma_min from Step 10
];

layout_names_step12 = {
    'Baseline'
    'Highest sigma_min'
    'Lowest condition number'
    'Second-highest sigma_min'
};

n_layouts_step12 = size(layouts_step12, 1);

% Confirm all requested positions belong to the candidate grid.
sensor_indices_step12 = zeros(n_layouts_step12, 3);

for q = 1:n_layouts_step12
    [found, sensor_idx] = ismember( ...
        layouts_step12(q,:), sensor_positions_test);

    if ~all(found)
        error('A sensor position is missing from sensor_positions_test.');
    end

    sensor_indices_step12(q,:) = sensor_idx;
end

% Generate true two-damage responses at all candidate sensor positions.
EI_multi_step12 = EI * (1 - d_multi_true);

Y_multi_all_step12 = compute_influence_lines( ...
    EI_multi_step12, L, n_elem, x_load, sensor_positions_test);

% Precompute each possible two-damage candidate once, at all 19 positions.
% The response vector uses MATLAB column-major ordering.
fprintf('\nPrecomputing shared response bank for Step 12...\n');

response_bank_all_step12 = zeros( ...
    n_candidate_sensors*n_load_pts, n_candidates_mc);

for c = 1:n_candidates_mc

    d_candidate = zeros(n_elem, 1);
    d_candidate(candidate_e1_mc(c)) = candidate_s1_mc(c);
    d_candidate(candidate_e2_mc(c)) = candidate_s2_mc(c);

    EI_candidate = EI * (1 - d_candidate);

    Y_candidate_all = compute_influence_lines( ...
        EI_candidate, L, n_elem, x_load, sensor_positions_test);

    response_bank_all_step12(:,c) = ...
        Y_candidate_all(:) - Y_healthy_all(:);
end

% Results: each row is a layout, each column a noise level.
success_rate_step12 = zeros( ...
    n_layouts_step12, numel(noise_levels_stress));

fprintf('\n--- STEP 12: DIRECT LAYOUT COMPARISON ---\n');
fprintf('Trials per layout and noise level: %d\n', n_mc_multi);
fprintf('True damaged elements: %d and %d\n', ...
    multi_true_elements(1), multi_true_elements(2));

for q = 1:n_layouts_step12

    sensor_idx = sensor_indices_step12(q,:);

    % Select rows in the same order as Y_selected(:).
    selected_rows = reshape( ...
        sensor_idx(:) + ...
        (0:n_load_pts-1)*n_candidate_sensors, [], 1);

    candidate_bank_layout = ...
        response_bank_all_step12(selected_rows,:);

    Yh_layout = Y_healthy_all(sensor_idx,:);
    Yd_layout = Y_multi_all_step12(sensor_idx,:);

    for n = 1:numel(noise_levels_stress)

        sigma_trial = ...
            noise_levels_stress(n)*reference_deflection;

        correct_count = 0;

        for r = 1:n_mc_multi

            % Use the same random seed for every layout comparison.
            rng(9000 + 1000*n + r, 'twister');

            Yh_obs = Yh_layout + ...
                sigma_trial*randn(size(Yh_layout));

            Yd_obs = Yd_layout + ...
                sigma_trial*randn(size(Yd_layout));

            b_trial = Yd_obs(:) - Yh_obs(:);

            residuals_trial = sqrt(sum( ...
                (candidate_bank_layout - b_trial).^2, 1));

            [~, best_idx] = min(residuals_trial);

            pair_correct = ...
                candidate_e1_mc(best_idx) == multi_true_elements(1) && ...
                candidate_e2_mc(best_idx) == multi_true_elements(2);

            correct_count = correct_count + pair_correct;
        end

        success_rate_step12(q,n) = ...
            100*correct_count/n_mc_multi;
    end
end

fprintf('\nLayout | 1%% noise | 2%% noise | 5%% noise\n');
fprintf('-------|----------|----------|----------\n');

for q = 1:n_layouts_step12
    fprintf('%5d  | %7.1f%% | %7.1f%% | %7.1f%%\n', ...
        q, success_rate_step12(q,1), ...
        success_rate_step12(q,2), success_rate_step12(q,3));

    fprintf('  %s\n', layout_names_step12{q});
end

% Plot direct localization performance.
figure('Name', 'Direct sensor-layout comparison', 'Color', 'w');

bar(100*noise_levels_stress, success_rate_step12.');

grid on;
xlabel('Measurement noise level (% of reference deflection)');
ylabel('Correct two-damage localization (%)');
title('Sensor-layout performance using nonlinear identification');
legend(layout_names_step12, 'Location', 'best');
ylim([0 105]);


%% STEP 13: FILTERING THE INFLUENCE-LINE CURVATURE INDICATOR

% The CDI here is based on the second derivative with respect to
% moving-load position, NOT physical curvature along the bridge span.

window_sg = 9;       % Odd window length in samples
polynomial_order = 3;

dx_cdi = x_load(2) - x_load(1);

% Clean reference CDI: no measurement noise
curv_h_clean_13 = second_derivative_rows( ...
    Y_healthy_clean, dx_cdi);

curv_d_clean_13 = second_derivative_rows( ...
    Y_damaged_clean, dx_cdi);

CDI_clean_13 = sqrt(mean( ...
    (curv_d_clean_13 - curv_h_clean_13).^2, 1));

% Raw noisy CDI
curv_h_raw_13 = second_derivative_rows( ...
    Y_healthy_obs, dx_cdi);

curv_d_raw_13 = second_derivative_rows( ...
    Y_damaged_obs, dx_cdi);

CDI_raw_13 = sqrt(mean( ...
    (curv_d_raw_13 - curv_h_raw_13).^2, 1));

% Apply the same local polynomial filter to healthy and damaged data.
Y_healthy_sg_13 = sgolay_smooth_rows( ...
    Y_healthy_obs, window_sg, polynomial_order);

Y_damaged_sg_13 = sgolay_smooth_rows( ...
    Y_damaged_obs, window_sg, polynomial_order);

% Recalculate curvature after filtering
curv_h_sg_13 = second_derivative_rows( ...
    Y_healthy_sg_13, dx_cdi);

curv_d_sg_13 = second_derivative_rows( ...
    Y_damaged_sg_13, dx_cdi);

CDI_filtered_13 = sqrt(mean( ...
    (curv_d_sg_13 - curv_h_sg_13).^2, 1));

% Exclude endpoints from the comparison because numerical derivatives
% and local polynomial fits can be less reliable there.
edge_margin = floor(window_sg/2) + 2;
valid_idx = (edge_margin + 1):(n_load_pts - edge_margin);

rmse_raw_cdi = sqrt(mean( ...
    (CDI_raw_13(valid_idx) - CDI_clean_13(valid_idx)).^2));

rmse_filtered_cdi = sqrt(mean( ...
    (CDI_filtered_13(valid_idx) - CDI_clean_13(valid_idx)).^2));

reference_rms_cdi = sqrt(mean(CDI_clean_13(valid_idx).^2));

relative_rmse_raw = ...
    100 * rmse_raw_cdi / max(reference_rms_cdi, eps);

relative_rmse_filtered = ...
    100 * rmse_filtered_cdi / max(reference_rms_cdi, eps);

fprintf('\n--- STEP 13: CDI FILTERING TEST ---\n');
fprintf('Savitzky-Golay window: %d samples\n', window_sg);
fprintf('Polynomial order: %d\n', polynomial_order);

fprintf('Raw CDI RMSE: %.6e\n', rmse_raw_cdi);
fprintf('Filtered CDI RMSE: %.6e\n', rmse_filtered_cdi);

fprintf('Raw CDI relative RMSE: %.2f %%\n', relative_rmse_raw);
fprintf('Filtered CDI relative RMSE: %.2f %%\n', ...
    relative_rmse_filtered);

if rmse_filtered_cdi < rmse_raw_cdi
    fprintf('Result: filtering reduced CDI error in this realization.\n');
else
    fprintf('Result: filtering did not reduce CDI error in this realization.\n');
end

figure('Name', 'CDI filtering comparison', 'Color', 'w');

plot(x_load(valid_idx), CDI_clean_13(valid_idx), ...
    'LineWidth', 1.8);
hold on;

plot(x_load(valid_idx), CDI_raw_13(valid_idx), ...
    '--', 'LineWidth', 1.2);

plot(x_load(valid_idx), CDI_filtered_13(valid_idx), ...
    'LineWidth', 1.5);

grid on;
xlabel('Moving-load position (m)');
ylabel('RMS influence-line second-derivative difference');
title('Raw and filtered CDI versus clean reference');

legend('Clean reference', 'Raw noisy CDI', ...
    'Filtered CDI', 'Location', 'best');


%% STEP 14: MONTE CARLO SELECTION OF CDI FILTER WINDOW

filter_windows = [5 7 9 11 13 15];
filter_order = 3;
n_filter_trials = 30;

n_windows_test = numel(filter_windows);

% Use the same interior load positions for every window length.
max_half_window = floor(max(filter_windows)/2);
edge_margin_mc = max_half_window + 2;

valid_idx_mc = ...
    (edge_margin_mc + 1):(n_load_pts - edge_margin_mc);

% Clean-reference CDI, computed without measurement noise.
dx_mc = x_load(2) - x_load(1);

curv_h_reference = second_derivative_rows( ...
    Y_healthy_clean, dx_mc);

curv_d_reference = second_derivative_rows( ...
    Y_damaged_clean, dx_mc);

CDI_reference_mc = sqrt(mean( ...
    (curv_d_reference - curv_h_reference).^2, 1));

reference_rms_mc = sqrt(mean( ...
    CDI_reference_mc(valid_idx_mc).^2));

if reference_rms_mc <= eps
    error('Clean-reference CDI is too small for relative RMSE comparison.');
end

% Store relative RMSE for each trial and filter window.
relative_rmse_windows = zeros(n_windows_test, n_filter_trials);
relative_rmse_raw_mc = zeros(1, n_filter_trials);

for r = 1:n_filter_trials

    % Use reproducible, independent noise realizations.
    rng(14000 + r, 'twister');

    Yh_trial = Y_healthy_clean + ...
        sigma_measurement * randn(size(Y_healthy_clean));

    Yd_trial = Y_damaged_clean + ...
        sigma_measurement * randn(size(Y_damaged_clean));

    % Raw CDI for this trial.
    curv_h_raw = second_derivative_rows(Yh_trial, dx_mc);
    curv_d_raw = second_derivative_rows(Yd_trial, dx_mc);

    CDI_raw_trial = sqrt(mean( ...
        (curv_d_raw - curv_h_raw).^2, 1));

    rmse_raw = sqrt(mean( ...
        (CDI_raw_trial(valid_idx_mc) - ...
         CDI_reference_mc(valid_idx_mc)).^2));

    relative_rmse_raw_mc(r) = ...
        100 * rmse_raw / reference_rms_mc;

    % Apply each candidate filter window.
    for w = 1:n_windows_test

        win = filter_windows(w);

        Yh_filtered = sgolay_smooth_rows( ...
            Yh_trial, win, filter_order);

        Yd_filtered = sgolay_smooth_rows( ...
            Yd_trial, win, filter_order);

        curv_h_filtered = second_derivative_rows( ...
            Yh_filtered, dx_mc);

        curv_d_filtered = second_derivative_rows( ...
            Yd_filtered, dx_mc);

        CDI_filtered_trial = sqrt(mean( ...
            (curv_d_filtered - curv_h_filtered).^2, 1));

        rmse_filtered = sqrt(mean( ...
            (CDI_filtered_trial(valid_idx_mc) - ...
             CDI_reference_mc(valid_idx_mc)).^2));

        relative_rmse_windows(w,r) = ...
            100 * rmse_filtered / reference_rms_mc;
    end
end

% Summarize average error and variability across trials.
mean_relative_rmse = mean(relative_rmse_windows, 2);
std_relative_rmse = std(relative_rmse_windows, 0, 2);

[best_mean_rmse, best_window_idx] = min(mean_relative_rmse);
best_filter_window = filter_windows(best_window_idx);

fprintf('\n--- STEP 14: CDI FILTER-WINDOW ROBUSTNESS ---\n');
fprintf('Monte Carlo trials: %d\n', n_filter_trials);
fprintf('Polynomial order: %d\n', filter_order);
fprintf('Raw CDI mean relative RMSE: %.2f %% (SD %.2f)\n', ...
    mean(relative_rmse_raw_mc), std(relative_rmse_raw_mc));

fprintf('\nWindow | Mean relative RMSE (%%) | SD across trials (%%)\n');
fprintf('-------|------------------------|---------------------\n');

for w = 1:n_windows_test
    fprintf('%6d | %22.2f | %20.2f\n', ...
        filter_windows(w), ...
        mean_relative_rmse(w), ...
        std_relative_rmse(w));
end

fprintf('\nBest tested window: %d samples\n', best_filter_window);
fprintf('Mean relative RMSE at best window: %.2f %%\n', ...
    best_mean_rmse);

figure('Name', 'CDI filter-window robustness', 'Color', 'w');

errorbar(filter_windows, mean_relative_rmse, ...
    std_relative_rmse, '-o', 'LineWidth', 1.5);

hold on;
yline(mean(relative_rmse_raw_mc), '--', 'Raw CDI mean');

grid on;
xlabel('Savitzky-Golay window length (samples)');
ylabel('Relative CDI RMSE (%)');
title('CDI filter-window comparison over noise realizations');
legend('Filtered CDI (mean \pm SD)', ...
    'Raw CDI mean', 'Location', 'best');


%% STEP 15: DIRECT LOCAL-POLYNOMIAL CURVATURE ESTIMATION

windows_direct = [5 7 9 11 13 15];
poly_order_direct = 3;
n_trials_direct = 30;

% Use identical interior load positions for all window sizes.
max_half_direct = floor(max(windows_direct)/2);
valid_direct = (max_half_direct+1):(n_load_pts-max_half_direct);
dx_direct = x_load(2) - x_load(1);

n_windows_direct = numel(windows_direct);

% Build a clean reference CDI using the same derivative estimator
% as the corresponding noisy CDI.
CDI_reference_direct = zeros(n_windows_direct, n_load_pts);
reference_rms_direct = zeros(n_windows_direct, 1);

for w = 1:n_windows_direct

    win = windows_direct(w);

    curv_h_clean = sgolay_second_derivative_rows( ...
        Y_healthy_clean, dx_direct, win, poly_order_direct);

    curv_d_clean = sgolay_second_derivative_rows( ...
        Y_damaged_clean, dx_direct, win, poly_order_direct);

    CDI_reference_direct(w,:) = sqrt(mean( ...
        (curv_d_clean - curv_h_clean).^2, 1));

    reference_rms_direct(w) = sqrt(mean( ...
        CDI_reference_direct(w,valid_direct).^2));
end

% Compare raw finite differences, smooth-then-differentiate,
% and direct local-polynomial derivative estimates.
rmse_raw_direct = zeros(n_trials_direct, 1);
rmse_smoothed_direct = zeros(n_trials_direct, 1);
rmse_polynomial_direct = zeros(n_windows_direct, n_trials_direct);

for r = 1:n_trials_direct

    rng(18000 + r, 'twister');

    Yh_trial = Y_healthy_clean + ...
        sigma_measurement * randn(size(Y_healthy_clean));

    Yd_trial = Y_damaged_clean + ...
        sigma_measurement * randn(size(Y_damaged_clean));

    % Method A: raw second finite differences
    curv_h_raw = second_derivative_rows(Yh_trial, dx_direct);
    curv_d_raw = second_derivative_rows(Yd_trial, dx_direct);

    CDI_raw_trial = sqrt(mean( ...
        (curv_d_raw - curv_h_raw).^2, 1));

    clean_raw_h = second_derivative_rows(Y_healthy_clean, dx_direct);
    clean_raw_d = second_derivative_rows(Y_damaged_clean, dx_direct);

    CDI_clean_raw = sqrt(mean( ...
        (clean_raw_d - clean_raw_h).^2, 1));

    rmse_raw_direct(r) = 100 * sqrt(mean( ...
        (CDI_raw_trial(valid_direct) - ...
         CDI_clean_raw(valid_direct)).^2)) / ...
        max(sqrt(mean(CDI_clean_raw(valid_direct).^2)), eps);

    % Method B: smooth the lines, then differentiate twice
    Yh_sm = sgolay_smooth_rows(Yh_trial, 15, poly_order_direct);
    Yd_sm = sgolay_smooth_rows(Yd_trial, 15, poly_order_direct);

    curv_h_sm = second_derivative_rows(Yh_sm, dx_direct);
    curv_d_sm = second_derivative_rows(Yd_sm, dx_direct);

    CDI_sm_trial = sqrt(mean( ...
        (curv_d_sm - curv_h_sm).^2, 1));

    rmse_smoothed_direct(r) = 100 * sqrt(mean( ...
        (CDI_sm_trial(valid_direct) - ...
         CDI_clean_raw(valid_direct)).^2)) / ...
        max(sqrt(mean(CDI_clean_raw(valid_direct).^2)), eps);

    % Method C: estimate second derivative directly from local polynomials
    for w = 1:n_windows_direct

        win = windows_direct(w);

        curv_h_poly = sgolay_second_derivative_rows( ...
            Yh_trial, dx_direct, win, poly_order_direct);

        curv_d_poly = sgolay_second_derivative_rows( ...
            Yd_trial, dx_direct, win, poly_order_direct);

        CDI_poly_trial = sqrt(mean( ...
            (curv_d_poly - curv_h_poly).^2, 1));

        rmse_polynomial_direct(w,r) = 100 * sqrt(mean( ...
            (CDI_poly_trial(valid_direct) - ...
             CDI_reference_direct(w,valid_direct)).^2)) / ...
            max(reference_rms_direct(w), eps);
    end
end

fprintf('\n--- STEP 15: DIRECT DERIVATIVE COMPARISON ---\n');
fprintf('Trials: %d\n', n_trials_direct);
fprintf('Raw finite-difference CDI mean relative RMSE: %.2f%%\n', ...
    mean(rmse_raw_direct));
fprintf('Smooth-then-differentiate CDI mean relative RMSE: %.2f%%\n', ...
    mean(rmse_smoothed_direct));

fprintf('\nWindow | Direct derivative mean RMSE (%%) | SD (%%)\n');
fprintf('-------|----------------------------------|--------\n');

mean_poly_rmse = mean(rmse_polynomial_direct, 2);
std_poly_rmse = std(rmse_polynomial_direct, 0, 2);

for w = 1:n_windows_direct
    fprintf('%6d | %32.2f | %7.2f\n', ...
        windows_direct(w), mean_poly_rmse(w), ...
        std_poly_rmse(w));
end

[best_poly_rmse, best_poly_idx] = min(mean_poly_rmse);

fprintf('\nBest direct-derivative window: %d samples\n', ...
    windows_direct(best_poly_idx));
fprintf('Mean relative RMSE: %.2f%%\n', best_poly_rmse);

figure('Name', 'Direct curvature estimation comparison', 'Color', 'w');

errorbar(windows_direct, mean_poly_rmse, ...
    std_poly_rmse, '-o', 'LineWidth', 1.5);
hold on;

yline(mean(rmse_raw_direct), '--', 'Raw finite-difference reference');
yline(mean(rmse_smoothed_direct), ':', ...
    'Smooth-then-differentiate reference');

grid on;
xlabel('Window length (samples)');
ylabel('Relative CDI RMSE (%)');
title('Direct local-polynomial curvature estimation');
legend('Direct derivative (mean \pm SD)', ...
    'Raw finite differences', ...
    'Smooth then differentiate', ...
    'Location', 'best');

%% STEP 16: TEST CDI DAMAGE LOCALIZATION

n_trials_step16 = 30;
windows_step16 = [9 11 13 15];
poly_order_step16 = 3;
smooth_window_step16 = 15;
true_damage_elem_step16 = damage_elem;
dx_step16 = x_load(2) - x_load(1);

% ----- Clean reference: raw finite differences -----
curv_h_raw_clean16 = second_derivative_rows( ...
    Y_healthy_clean, dx_step16);
curv_d_raw_clean16 = second_derivative_rows( ...
    Y_damaged_clean, dx_step16);

CDI_raw_clean16 = sqrt(mean( ...
    (curv_d_raw_clean16 - curv_h_raw_clean16).^2, 1));

% ----- Clean reference: smooth, then differentiate -----
Yh_sm_clean16 = sgolay_smooth_rows( ...
    Y_healthy_clean, smooth_window_step16, poly_order_step16);
Yd_sm_clean16 = sgolay_smooth_rows( ...
    Y_damaged_clean, smooth_window_step16, poly_order_step16);

curv_h_sm_clean16 = second_derivative_rows(Yh_sm_clean16, dx_step16);
curv_d_sm_clean16 = second_derivative_rows(Yd_sm_clean16, dx_step16);

CDI_sm_clean16 = sqrt(mean( ...
    (curv_d_sm_clean16 - curv_h_sm_clean16).^2, 1));

% ----- Clean references: direct polynomial derivatives -----
CDI_direct_clean16 = zeros(numel(windows_step16), n_load_pts);

for w16 = 1:numel(windows_step16)
    win16 = windows_step16(w16);

    curv_h16 = sgolay_second_derivative_rows( ...
        Y_healthy_clean, dx_step16, win16, poly_order_step16);
    curv_d16 = sgolay_second_derivative_rows( ...
        Y_damaged_clean, dx_step16, win16, poly_order_step16);

    CDI_direct_clean16(w16,:) = sqrt(mean( ...
        (curv_d16 - curv_h16).^2, 1));
end

% Convert CDI values along the load path into element-level scores.
% Each element uses load positions in its interval [left, right).
clean_scores_raw16 = cdi_element_scores_step16( ...
    CDI_raw_clean16, x_load, L, n_elem);
clean_scores_sm16 = cdi_element_scores_step16( ...
    CDI_sm_clean16, x_load, L, n_elem);

clean_scores_direct16 = zeros(numel(windows_step16), n_elem);

for w16 = 1:numel(windows_step16)
    clean_scores_direct16(w16,:) = cdi_element_scores_step16( ...
        CDI_direct_clean16(w16,:), x_load, L, n_elem);
end

[~, clean_peak_raw16] = max(clean_scores_raw16);
[~, clean_peak_sm16] = max(clean_scores_sm16);

fprintf('\n--- STEP 16: CDI LOCALIZATION TEST ---\n');
fprintf('True damaged element: %d\n', true_damage_elem_step16);
fprintf('Clean CDI peak, raw method: element %d\n', clean_peak_raw16);
fprintf('Clean CDI peak, smooth method: element %d\n', clean_peak_sm16);

for w16 = 1:numel(windows_step16)
    [~, peak16] = max(clean_scores_direct16(w16,:));
    fprintf('Clean CDI peak, direct window %d: element %d\n', ...
        windows_step16(w16), peak16);
end

% ----- Monte Carlo localization tests -----
success_raw16 = 0;
success_sm16 = 0;
success_direct16 = zeros(numel(windows_step16), 1);

for r16 = 1:n_trials_step16
    rng(19000 + r16, 'twister');

    Yh_trial16 = Y_healthy_clean + ...
        sigma_measurement * randn(size(Y_healthy_clean));
    Yd_trial16 = Y_damaged_clean + ...
        sigma_measurement * randn(size(Y_damaged_clean));

    % A. Raw finite differences
    curv_h16 = second_derivative_rows(Yh_trial16, dx_step16);
    curv_d16 = second_derivative_rows(Yd_trial16, dx_step16);

    CDI16 = sqrt(mean((curv_d16 - curv_h16).^2, 1));
    scores16 = cdi_element_scores_step16(CDI16, x_load, L, n_elem);

    [~, predicted16] = max(scores16);
    success_raw16 = success_raw16 + ...
        (predicted16 == true_damage_elem_step16);

    % B. Smooth, then differentiate
    Yh_sm16 = sgolay_smooth_rows( ...
        Yh_trial16, smooth_window_step16, poly_order_step16);
    Yd_sm16 = sgolay_smooth_rows( ...
        Yd_trial16, smooth_window_step16, poly_order_step16);

    curv_h16 = second_derivative_rows(Yh_sm16, dx_step16);
    curv_d16 = second_derivative_rows(Yd_sm16, dx_step16);

    CDI16 = sqrt(mean((curv_d16 - curv_h16).^2, 1));
    scores16 = cdi_element_scores_step16(CDI16, x_load, L, n_elem);

    [~, predicted16] = max(scores16);
    success_sm16 = success_sm16 + ...
        (predicted16 == true_damage_elem_step16);

    % C. Direct local-polynomial differentiation
    for w16 = 1:numel(windows_step16)
        win16 = windows_step16(w16);

        curv_h16 = sgolay_second_derivative_rows( ...
            Yh_trial16, dx_step16, win16, poly_order_step16);
        curv_d16 = sgolay_second_derivative_rows( ...
            Yd_trial16, dx_step16, win16, poly_order_step16);

        CDI16 = sqrt(mean((curv_d16 - curv_h16).^2, 1));
        scores16 = cdi_element_scores_step16( ...
            CDI16, x_load, L, n_elem);

        [~, predicted16] = max(scores16);

        success_direct16(w16) = success_direct16(w16) + ...
            (predicted16 == true_damage_elem_step16);
    end
end

fprintf('\nMethod | Correct localizations | Success rate\n');
fprintf('-------|-----------------------|-------------\n');
fprintf('Raw finite differences | %2d/%2d | %6.2f%%\n', ...
    success_raw16, n_trials_step16, ...
    100*success_raw16/n_trials_step16);

fprintf('Smooth then differentiate | %2d/%2d | %6.2f%%\n', ...
    success_sm16, n_trials_step16, ...
    100*success_sm16/n_trials_step16);

for w16 = 1:numel(windows_step16)
    fprintf('Direct window %2d | %2d/%2d | %6.2f%%\n', ...
        windows_step16(w16), success_direct16(w16), ...
        n_trials_step16, ...
        100*success_direct16(w16)/n_trials_step16);
end

%% STEP 17: CDI ROBUSTNESS WITH REPEATED MEASUREMENTS

repeat_counts_step17 = [1 4 16 64];
n_trials_step17 = 30;
window_step17 = 15;
poly_order_step17 = 3;

success_step17 = zeros(size(repeat_counts_step17));

for k_idx17 = 1:numel(repeat_counts_step17)

    K17 = repeat_counts_step17(k_idx17);
    correct17 = 0;

    for trial17 = 1:n_trials_step17

        % Average K independent noisy measurements for each condition.
        Yh_avg17 = zeros(size(Y_healthy_clean));
        Yd_avg17 = zeros(size(Y_damaged_clean));

        for rep17 = 1:K17

            % Reuse the first few noise realizations across K values
            % to make the comparison more consistent.
            rng(22000 + 100*trial17 + rep17, 'twister');

            Yh_rep17 = Y_healthy_clean + ...
                sigma_measurement * randn(size(Y_healthy_clean));

            Yd_rep17 = Y_damaged_clean + ...
                sigma_measurement * randn(size(Y_damaged_clean));

            Yh_avg17 = Yh_avg17 + Yh_rep17;
            Yd_avg17 = Yd_avg17 + Yd_rep17;
        end

        Yh_avg17 = Yh_avg17 / K17;
        Yd_avg17 = Yd_avg17 / K17;

        % Estimate second derivatives directly using local polynomials.
        curv_h17 = sgolay_second_derivative_rows( ...
            Yh_avg17, dx_step16, window_step17, poly_order_step17);

        curv_d17 = sgolay_second_derivative_rows( ...
            Yd_avg17, dx_step16, window_step17, poly_order_step17);

        % Compute the CDI along the moving-load path.
        CDI17 = sqrt(mean((curv_d17 - curv_h17).^2, 1));

        % Convert the load-path CDI into element-level scores.
        scores17 = cdi_element_scores_step16( ...
            CDI17, x_load, L, n_elem);

        [~, predicted17] = max(scores17);

        if predicted17 == damage_elem
            correct17 = correct17 + 1;
        end
    end

    success_step17(k_idx17) = correct17;

end

fprintf('\n--- STEP 17: REPEATED-MEASUREMENT CDI TEST ---\n');
fprintf('Trials per repeat count: %d\n', n_trials_step17);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('Direct polynomial derivative window: %d samples\n\n', ...
    window_step17);

fprintf('Repeats | Correct localizations | Success rate\n');
fprintf('--------|-----------------------|-------------\n');

for k_idx17 = 1:numel(repeat_counts_step17)
    K17 = repeat_counts_step17(k_idx17);
    correct17 = success_step17(k_idx17);

    fprintf('%7d | %2d/%2d                 | %6.2f%%\n', ...
        K17, correct17, n_trials_step17, ...
        100 * correct17 / n_trials_step17);
end

% Plot localization success against number of repeated measurements.
figure('Name', 'Step 17: CDI Repeated-Measurement Test');

success_rate17 = 100 * success_step17 / n_trials_step17;

semilogx(repeat_counts_step17, success_rate17, ...
    '-o', 'LineWidth', 1.5, 'MarkerSize', 7);

xlabel('Number of repeated measurements');
ylabel('Correct localization rate (%)');
title('CDI Localization vs. Repeated Measurements');
ylim([0 105]);
xticks(repeat_counts_step17);
grid on;

%% STEP 18: MONTE CARLO VALIDATION OF REPEATED-MEASUREMENT CDI

repeat_counts_step18 = [16 32 64 128];
n_trials_step18 = 100;
window_step18 = 15;
poly_order_step18 = 3;
z_step18 = 1.96;

success_step18 = zeros(size(repeat_counts_step18));
ci_low_step18 = zeros(size(repeat_counts_step18));
ci_high_step18 = zeros(size(repeat_counts_step18));

for k_idx18 = 1:numel(repeat_counts_step18)

    K18 = repeat_counts_step18(k_idx18);
    correct18 = 0;

    for trial18 = 1:n_trials_step18

        Yh_avg18 = zeros(size(Y_healthy_clean));
        Yd_avg18 = zeros(size(Y_damaged_clean));

        for rep18 = 1:K18

            % Reuse identical noise realizations across K values.
            % The 1000-seed spacing prevents overlap between trials.
            rng(100000 + 1000*trial18 + rep18, 'twister');

            Yh_rep18 = Y_healthy_clean + ...
                sigma_measurement * randn(size(Y_healthy_clean));

            Yd_rep18 = Y_damaged_clean + ...
                sigma_measurement * randn(size(Y_damaged_clean));

            Yh_avg18 = Yh_avg18 + Yh_rep18;
            Yd_avg18 = Yd_avg18 + Yd_rep18;
        end

        Yh_avg18 = Yh_avg18 / K18;
        Yd_avg18 = Yd_avg18 / K18;

        curv_h18 = sgolay_second_derivative_rows( ...
            Yh_avg18, dx_step16, window_step18, poly_order_step18);

        curv_d18 = sgolay_second_derivative_rows( ...
            Yd_avg18, dx_step16, window_step18, poly_order_step18);

        CDI18 = sqrt(mean((curv_d18 - curv_h18).^2, 1));

        scores18 = cdi_element_scores_step16( ...
            CDI18, x_load, L, n_elem);

        [~, predicted18] = max(scores18);

        if predicted18 == damage_elem
            correct18 = correct18 + 1;
        end
    end

    success_step18(k_idx18) = correct18;

    % Wilson 95% confidence interval for the success proportion.
    p18 = correct18 / n_trials_step18;
    denom18 = 1 + z_step18^2 / n_trials_step18;

    center18 = (p18 + z_step18^2 / ...
        (2*n_trials_step18)) / denom18;

    half18 = z_step18 * sqrt( ...
        p18*(1-p18)/n_trials_step18 + ...
        z_step18^2/(4*n_trials_step18^2)) / denom18;

    ci_low_step18(k_idx18) = max(0, center18-half18);
    ci_high_step18(k_idx18) = min(1, center18+half18);
end

success_rate_step18 = ...
    100 * success_step18 / n_trials_step18;

fprintf('\n--- STEP 18: CDI REPEATABILITY VALIDATION ---\n');
fprintf('Monte Carlo trials per setting: %d\n', n_trials_step18);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('Direct derivative window: %d samples\n\n', window_step18);

fprintf('Repeats | Correct | Success rate | 95%% Wilson CI\n');
fprintf('--------|---------|--------------|----------------\n');

for k_idx18 = 1:numel(repeat_counts_step18)
    fprintf('%7d | %2d/%-4d | %10.2f%% | [%5.1f%%, %5.1f%%]\n', ...
        repeat_counts_step18(k_idx18), ...
        success_step18(k_idx18), n_trials_step18, ...
        success_rate_step18(k_idx18), ...
        100*ci_low_step18(k_idx18), ...
        100*ci_high_step18(k_idx18));
end

% Plot localization rates with confidence intervals.
figure('Name', 'Step 18: CDI Repeatability Validation');

lower_error18 = success_rate_step18 - 100*ci_low_step18;
upper_error18 = 100*ci_high_step18 - success_rate_step18;

errorbar(repeat_counts_step18, success_rate_step18, ...
    lower_error18, upper_error18, '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

set(gca, 'XScale', 'log');
xticks(repeat_counts_step18);
xlabel('Number of repeated measurements averaged');
ylabel('Correct localization rate (%)');
title('CDI Localization Reliability with Repeated Measurements');
ylim([0 105]);
grid on;

%% STEP 19: FAIR COMPARISON OF CDI AND NONLINEAR INVERSE FITTING

repeat_counts_step19 = [1 16 32 64 128];
n_trials_step19 = 100;
window_step19 = 15;
poly_order_step19 = 3;
z_step19 = 1.96;

success_cdi_step19 = zeros(size(repeat_counts_step19));
success_inverse_step19 = zeros(size(repeat_counts_step19));

for k_idx19 = 1:numel(repeat_counts_step19)

    K19 = repeat_counts_step19(k_idx19);

    correct_cdi19 = 0;
    correct_inverse19 = 0;

    for trial19 = 1:n_trials_step19

        % Average independent healthy and damaged measurements.
        Yh_avg19 = zeros(size(Y_healthy_clean));
        Yd_avg19 = zeros(size(Y_damaged_clean));

        for rep19 = 1:K19

            % Reuse identical noise realizations across repeat counts.
            rng(100000 + 1000*trial19 + rep19, 'twister');

            Yh_rep19 = Y_healthy_clean + ...
                sigma_measurement * randn(size(Y_healthy_clean));

            Yd_rep19 = Y_damaged_clean + ...
                sigma_measurement * randn(size(Y_damaged_clean));

            Yh_avg19 = Yh_avg19 + Yh_rep19;
            Yd_avg19 = Yd_avg19 + Yd_rep19;
        end

        Yh_avg19 = Yh_avg19 / K19;
        Yd_avg19 = Yd_avg19 / K19;

        % ---------------------------------------------------------
        % METHOD A: CDI WITH DIRECT POLYNOMIAL DIFFERENTIATION
        % ---------------------------------------------------------
        curv_h19 = sgolay_second_derivative_rows( ...
            Yh_avg19, dx_step16, window_step19, poly_order_step19);

        curv_d19 = sgolay_second_derivative_rows( ...
            Yd_avg19, dx_step16, window_step19, poly_order_step19);

        CDI19 = sqrt(mean((curv_d19 - curv_h19).^2, 1));

        scores_cdi19 = cdi_element_scores_step16( ...
            CDI19, x_load, L, n_elem);

        [~, predicted_cdi19] = max(scores_cdi19);

        if predicted_cdi19 == damage_elem
            correct_cdi19 = correct_cdi19 + 1;
        end

        % ---------------------------------------------------------
        % METHOD B: NONLINEAR RESPONSE-BASED INVERSE FITTING
        % The candidate bank was precomputed in Step 6.
        % ---------------------------------------------------------
        b19 = Yd_avg19(:) - Yh_avg19(:);

        residuals_inverse19 = sqrt(sum( ...
            (candidate_responses - b19).^2, 1));

        [~, best_candidate19] = min(residuals_inverse19);

        predicted_inverse19 = candidate_elements(best_candidate19);

        if predicted_inverse19 == damage_elem
            correct_inverse19 = correct_inverse19 + 1;
        end

    end

    success_cdi_step19(k_idx19) = correct_cdi19;
    success_inverse_step19(k_idx19) = correct_inverse19;
end

% -------------------------------------------------------------
% REPORT SUCCESS RATES AND 95% WILSON CONFIDENCE INTERVALS
% -------------------------------------------------------------
fprintf('\n--- STEP 19: CDI VS NONLINEAR INVERSE FITTING ---\n');
fprintf('Trials per repeat count: %d\n', n_trials_step19);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('CDI direct derivative window: %d samples\n\n', window_step19);

fprintf(['Repeats | CDI success (95%% CI)       | ', ...
         'Inverse success (95%% CI)\n']);
fprintf(['--------|----------------------------|', ...
         '----------------------------\n']);

rate_cdi19 = 100 * success_cdi_step19 / n_trials_step19;
rate_inverse19 = 100 * success_inverse_step19 / n_trials_step19;

ci_cdi_low19 = zeros(size(repeat_counts_step19));
ci_cdi_high19 = zeros(size(repeat_counts_step19));
ci_inv_low19 = zeros(size(repeat_counts_step19));
ci_inv_high19 = zeros(size(repeat_counts_step19));

for k_idx19 = 1:numel(repeat_counts_step19)

    counts19 = [success_cdi_step19(k_idx19), ...
                success_inverse_step19(k_idx19)];

    for method19 = 1:2

        p19 = counts19(method19) / n_trials_step19;
        denom19 = 1 + z_step19^2 / n_trials_step19;

        center19 = (p19 + z_step19^2 / ...
            (2*n_trials_step19)) / denom19;

        half19 = z_step19 * sqrt( ...
            p19*(1-p19)/n_trials_step19 + ...
            z_step19^2/(4*n_trials_step19^2)) / denom19;

        low19 = max(0, center19 - half19);
        high19 = min(1, center19 + half19);

        if method19 == 1
            ci_cdi_low19(k_idx19) = low19;
            ci_cdi_high19(k_idx19) = high19;
        else
            ci_inv_low19(k_idx19) = low19;
            ci_inv_high19(k_idx19) = high19;
        end
    end

    fprintf('%7d | %3d/%3d = %5.1f%% [%4.1f,%4.1f] | ', ...
        repeat_counts_step19(k_idx19), ...
        success_cdi_step19(k_idx19), n_trials_step19, ...
        rate_cdi19(k_idx19), ...
        100*ci_cdi_low19(k_idx19), ...
        100*ci_cdi_high19(k_idx19));

    fprintf('%3d/%3d = %5.1f%% [%4.1f,%4.1f]\n', ...
        success_inverse_step19(k_idx19), n_trials_step19, ...
        rate_inverse19(k_idx19), ...
        100*ci_inv_low19(k_idx19), ...
        100*ci_inv_high19(k_idx19));
end

% Plot the comparison.
figure('Name', 'Step 19: CDI vs Nonlinear Inverse Fitting');

semilogx(repeat_counts_step19, rate_cdi19, '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);
hold on;

semilogx(repeat_counts_step19, rate_inverse19, '-s', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

xlabel('Number of repeated measurements averaged');
ylabel('Correct localization rate (%)');
title('CDI vs Nonlinear Inverse Fitting');
legend('CDI', 'Nonlinear inverse fitting', 'Location', 'best');
xticks(repeat_counts_step19);
ylim([0 105]);
grid on;
hold off;

%% STEP 20: SENSOR-POSITION MODEL-MISMATCH VALIDATION

sensor_offsets_step20 = [0 0.01 0.025 0.05 0.10];
repeat_counts_step20 = [1 64];
n_trials_step20 = 100;

success_step20 = zeros( ...
    numel(sensor_offsets_step20), numel(repeat_counts_step20));

% Three-sensor offset pattern
offset_pattern_step20 = [1 -1 1];
for off_idx20 = 1:numel(sensor_offsets_step20)

    offset20 = sensor_offsets_step20(off_idx20);
    % Actual sensor coordination
    x_sensor_true20 = x_sensor + ...
        offset20 * offset_pattern_step20;

    if any(x_sensor_true20 <= 0) || any(x_sensor_true20 >= L)
        error('Step 20: sensor offset places a sensor outside the span.');
    end
% Generate clean responses using the actual, shifted coordinates.

Yh_true20 = compute_influence_lines( ...
    EI_healthy, L, n_elem, x_load, x_sensor_true20);

Yd_true20 = compute_influence_lines( ...
    EI_damaged, L, n_elem, x_load, x_sensor_true20);

    for rep_idx20 = 1:numel(repeat_counts_step20)

        K20 = repeat_counts_step20(rep_idx20);
        correct20 = 0;

        for trial20 = 1:n_trials_step20

            Yh_avg20 = zeros(size(Yh_true20));
            Yd_avg20 = zeros(size(Yd_true20));

            for measurement20 = 1:K20

                % Same standardized random noise across offset
                % conditions for the same trial and repeat index.
                rng(300000 + 1000*trial20 + measurement20, 'twister');

                Yh_noisy20 = Yh_true20 + ...
                    sigma_measurement * randn(size(Yh_true20));

                Yd_noisy20 = Yd_true20 + ...
                    sigma_measurement * randn(size(Yd_true20));

                Yh_avg20 = Yh_avg20 + Yh_noisy20;
                Yd_avg20 = Yd_avg20 + Yd_noisy20;
            end

            Yh_avg20 = Yh_avg20 / K20;
            Yd_avg20 = Yd_avg20 / K20;

            % Differential measurements to compare with the
            % nominal candidate response bank.
            b20 = Yd_avg20(:) - Yh_avg20(:);

            residuals20 = sqrt(sum( ...
                (candidate_responses - b20).^2, 1));

            [~, best_candidate20] = min(residuals20);
            predicted_element20 = candidate_elements(best_candidate20);

            if predicted_element20 == damage_elem
                correct20 = correct20 + 1;
            end
        end

        success_step20(off_idx20, rep_idx20) = correct20;
    end
end

fprintf('\n--- STEP 20: SENSOR-POSITION MISMATCH TEST ---\n');
fprintf('Trials per condition: %d\n', n_trials_step20);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('Measurement noise level: 1%%\n\n');

fprintf('Offset (cm) | K=1 success | K=64 success\n');
fprintf('------------|-------------|-------------\n');

for off_idx20 = 1:numel(sensor_offsets_step20)

    fprintf('%11.1f | %3d/100 (%5.1f%%) | %3d/100 (%5.1f%%)\n', ...
        100*sensor_offsets_step20(off_idx20), ...
        success_step20(off_idx20,1), ...
        100*success_step20(off_idx20,1)/n_trials_step20, ...
        success_step20(off_idx20,2), ...
        100*success_step20(off_idx20,2)/n_trials_step20);
end

% Plot localization success against sensor-position offset.
figure('Name', 'Step 20: Sensor-Position Model Mismatch');

plot(100*sensor_offsets_step20, ...
    100*success_step20(:,1)/n_trials_step20, '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);
hold on;

plot(100*sensor_offsets_step20, ...
    100*success_step20(:,2)/n_trials_step20, '-s', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

xlabel('Sensor-position offset magnitude (cm)');
ylabel('Correct localization rate (%)');
title('Effect of Sensor-Position Model Mismatch');
legend('One measurement', 'Average of 64 measurements', ...
    'Location', 'best');
ylim([0 105]);
grid on;
hold off;

%% STEP 21: SEVERITY AND RESIDUAL UNDER SENSOR-POSITION MISMATCH

offsets_step21 = [0 0.01 0.025 0.05 0.10];
repeats_step21 = [1 64];
trials_step21 = 100;

n_offsets21 = numel(offsets_step21);
n_repeats21 = numel(repeats_step21);

success21 = zeros(n_offsets21, n_repeats21);
mean_severity21 = zeros(n_offsets21, n_repeats21);
mae_all21 = zeros(n_offsets21, n_repeats21);
mae_correct21 = nan(n_offsets21, n_repeats21);
mean_residual21 = zeros(n_offsets21, n_repeats21);

offset_pattern21 = [1 -1 1];

for oi21 = 1:n_offsets21

    offset21 = offsets_step21(oi21);

    % Actual sensor positions, different from the nominal
    % positions used to precompute candidate_responses.
    x_sensor_actual21 = x_sensor + ...
        offset21 * offset_pattern21;

    if any(x_sensor_actual21 <= 0) || ...
            any(x_sensor_actual21 >= L)
        error('Step 21: sensor position is outside the span.');
    end

    % Generate the true healthy and damaged responses using
    % the actual sensor positions.
    Yh_actual21 = compute_influence_lines( ...
        EI_healthy, L, n_elem, x_load, x_sensor_actual21);

    Yd_actual21 = compute_influence_lines( ...
        EI_damaged, L, n_elem, x_load, x_sensor_actual21);

    for ki21 = 1:n_repeats21

        K21 = repeats_step21(ki21);

        estimated_element21 = zeros(trials_step21, 1);
        estimated_severity21 = zeros(trials_step21, 1);
        best_residual21 = zeros(trials_step21, 1);

        for trial21 = 1:trials_step21

            Yh_avg21 = zeros(size(Yh_actual21));
            Yd_avg21 = zeros(size(Yd_actual21));

            for measurement21 = 1:K21

                % Matched random-noise realizations across offset
                % and repeat-count conditions.
                rng(400000 + 1000*trial21 + measurement21, ...
                    'twister');

                Yh_noisy21 = Yh_actual21 + ...
                    sigma_measurement * randn(size(Yh_actual21));

                Yd_noisy21 = Yd_actual21 + ...
                    sigma_measurement * randn(size(Yd_actual21));

                Yh_avg21 = Yh_avg21 + Yh_noisy21;
                Yd_avg21 = Yd_avg21 + Yd_noisy21;
            end

            Yh_avg21 = Yh_avg21 / K21;
            Yd_avg21 = Yd_avg21 / K21;

            % Observed differential influence lines
            b21 = Yd_avg21(:) - Yh_avg21(:);

            % Fit against the original nominal-position candidate bank
            residuals21 = sqrt(sum( ...
                (candidate_responses - b21).^2, 1));

            [best_residual21(trial21), best_idx21] = ...
                min(residuals21);

            estimated_element21(trial21) = ...
                candidate_elements(best_idx21);

            estimated_severity21(trial21) = ...
                candidate_severities(best_idx21);
        end

        % Localization success
        correct21 = estimated_element21 == damage_elem;

        success21(oi21, ki21) = ...
            100 * mean(correct21);

        % Mean estimated severity, as a percentage
        mean_severity21(oi21, ki21) = ...
            100 * mean(estimated_severity21);

        % Severity error across all trials, in percentage points
        mae_all21(oi21, ki21) = ...
            100 * mean(abs(estimated_severity21 - damage_severity));

        % Severity error only when the location is correct
        if any(correct21)
            mae_correct21(oi21, ki21) = ...
                100 * mean(abs( ...
                estimated_severity21(correct21) - damage_severity));
        end

        % Mean best-fit residual
        mean_residual21(oi21, ki21) = ...
            mean(best_residual21);
    end
end

fprintf('\n--- STEP 21: SEVERITY AND RESIDUAL ANALYSIS ---\n');
fprintf('Trials per condition: %d\n', trials_step21);
fprintf('True damage location: element %d\n', damage_elem);
fprintf('True stiffness loss: %.1f%%\n\n', ...
    100 * damage_severity);

fprintf(['Offset(cm) | Repeats | Localization(%%) | ', ...
         'Mean loss(%%) | MAE all(pp) | MAE correct(pp) | ', ...
         'Mean residual\n']);

fprintf(['-----------|---------|-----------------|', ...
         '-------------|-------------|-----------------|', ...
         '--------------\n']);

for oi21 = 1:n_offsets21
    for ki21 = 1:n_repeats21

        fprintf('%10.1f | %7d | %15.1f | %11.2f | %11.2f |', ...
            100 * offsets_step21(oi21), ...
            repeats_step21(ki21), ...
            success21(oi21,ki21), ...
            mean_severity21(oi21,ki21), ...
            mae_all21(oi21,ki21));

        if isnan(mae_correct21(oi21,ki21))
            fprintf(' %15s |', 'N/A');
        else
            fprintf(' %15.2f |', mae_correct21(oi21,ki21));
        end

        fprintf(' %.6e\n', mean_residual21(oi21,ki21));
    end
end

% Compare severity error across sensor-position offsets.
figure('Name', 'Step 21: Severity Error Under Sensor Mismatch');

plot(100*offsets_step21, mae_all21(:,1), '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);
hold on;

plot(100*offsets_step21, mae_all21(:,2), '-s', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

xlabel('Sensor-position offset magnitude (cm)');
ylabel('Mean absolute severity error (percentage points)');
title('Severity Estimation Under Sensor-Position Mismatch');
legend('One measurement', 'Average of 64 measurements', ...
    'Location', 'best');
grid on;
hold off;

%% STEP 22: GLOBAL EI MODEL-MISMATCH TEST

EI_factors_step22 = [0.80 0.90 0.95 1.00 1.05 1.10 1.20];
repeats_step22 = [1 64];
trials_step22 = 100;

n_factors22 = numel(EI_factors_step22);
n_repeat_settings22 = numel(repeats_step22);

success22 = zeros(n_factors22, n_repeat_settings22);
mean_severity22 = zeros(n_factors22, n_repeat_settings22);
mae_severity22 = zeros(n_factors22, n_repeat_settings22);
mean_residual22 = zeros(n_factors22, n_repeat_settings22);

for fi22 = 1:n_factors22

    factor22 = EI_factors_step22(fi22);

    % The actual beam has globally scaled EI.
    % The candidate bank remains based on nominal EI.
    EI_healthy_actual22 = factor22 * EI_healthy;
    EI_damaged_actual22 = factor22 * EI_damaged;

    Yh_actual22 = compute_influence_lines( ...
        EI_healthy_actual22, L, n_elem, x_load, x_sensor);

    Yd_actual22 = compute_influence_lines( ...
        EI_damaged_actual22, L, n_elem, x_load, x_sensor);

    for ki22 = 1:n_repeat_settings22

        K22 = repeats_step22(ki22);

        estimated_element22 = zeros(trials_step22, 1);
        estimated_severity22 = zeros(trials_step22, 1);
        residual22 = zeros(trials_step22, 1);

        for trial22 = 1:trials_step22

            Yh_avg22 = zeros(size(Yh_actual22));
            Yd_avg22 = zeros(size(Yd_actual22));

            for measurement22 = 1:K22

                % Paired random seeds across stiffness factors.
                rng(500000 + 1000*trial22 + measurement22, ...
                    'twister');

                Yh_noisy22 = Yh_actual22 + ...
                    sigma_measurement * randn(size(Yh_actual22));

                Yd_noisy22 = Yd_actual22 + ...
                    sigma_measurement * randn(size(Yd_actual22));

                Yh_avg22 = Yh_avg22 + Yh_noisy22;
                Yd_avg22 = Yd_avg22 + Yd_noisy22;
            end

            Yh_avg22 = Yh_avg22 / K22;
            Yd_avg22 = Yd_avg22 / K22;

            % Observed differential influence lines.
            b22 = Yd_avg22(:) - Yh_avg22(:);

            % Fit against the nominal-EI candidate response bank.
            residuals22 = sqrt(sum( ...
                (candidate_responses - b22).^2, 1));

            [residual22(trial22), best_idx22] = min(residuals22);

            estimated_element22(trial22) = ...
                candidate_elements(best_idx22);

            estimated_severity22(trial22) = ...
                candidate_severities(best_idx22);
        end

        % Localization accuracy.
        success22(fi22,ki22) = ...
            100 * mean(estimated_element22 == damage_elem);

        % Estimated damage severity and absolute error.
        mean_severity22(fi22,ki22) = ...
            100 * mean(estimated_severity22);

        mae_severity22(fi22,ki22) = ...
            100 * mean(abs(estimated_severity22 - damage_severity));

        % Mean norm of the best-fit residual.
        mean_residual22(fi22,ki22) = mean(residual22);
    end
end

fprintf('\n--- STEP 22: GLOBAL EI MISMATCH TEST ---\n');
fprintf('Trials per condition: %d\n', trials_step22);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('True stiffness loss: %.1f%%\n', 100*damage_severity);
fprintf('Sensor positions: nominal and correct\n\n');

fprintf(['Actual/nominal EI | Repeats | Localization(%%) | ', ...
         'Mean loss(%%) | MAE(pp) | Mean residual\n']);

fprintf(['------------------|---------|-----------------|', ...
         '-------------|---------|--------------\n']);

for fi22 = 1:n_factors22
    for ki22 = 1:n_repeat_settings22

        fprintf('%17.2f | %7d | %15.1f | %11.2f | %7.2f | %.6e\n', ...
            EI_factors_step22(fi22), ...
            repeats_step22(ki22), ...
            success22(fi22,ki22), ...
            mean_severity22(fi22,ki22), ...
            mae_severity22(fi22,ki22), ...
            mean_residual22(fi22,ki22));
    end
end

% Plot severity estimation error versus actual global EI factor.
figure('Name', 'Step 22: Global EI Mismatch');

plot(EI_factors_step22, mae_severity22(:,1), '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);
hold on;

plot(EI_factors_step22, mae_severity22(:,2), '-s', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

xlabel('Actual EI / nominal EI');
ylabel('Mean absolute severity error (percentage points)');
title('Damage-Severity Error Under Global EI Mismatch');
legend('One measurement', 'Average of 64 measurements', ...
    'Location', 'best');
grid on;
hold off;

%% STEP 23: CALIBRATE GLOBAL EI BEFORE DAMAGE-SEVERITY FITTING

EI_factors_step23 = [0.80 0.90 0.95 1.00 1.05 1.10 1.20];
repeats_step23 = [1 64];
trials_step23 = 100;

n_factors23 = numel(EI_factors_step23);
n_repeat_settings23 = numel(repeats_step23);

% Trial-level results
factor_estimates23 = zeros( ...
    n_factors23, n_repeat_settings23, trials_step23);

severity_raw23 = zeros( ...
    n_factors23, n_repeat_settings23, trials_step23);

severity_calibrated23 = zeros( ...
    n_factors23, n_repeat_settings23, trials_step23);

element_raw23 = zeros( ...
    n_factors23, n_repeat_settings23, trials_step23);

element_calibrated23 = zeros( ...
    n_factors23, n_repeat_settings23, trials_step23);

% Nominal healthy response used to estimate the global EI scale.
healthy_reference23 = Y_healthy_clean(:);
healthy_reference_norm23 = sum(healthy_reference23.^2);

if healthy_reference_norm23 <= eps
    error('Step 23: healthy reference response has zero norm.');
end

for fi23 = 1:n_factors23

    factor23 = EI_factors_step23(fi23);

    % Actual beam: globally scaled healthy and damaged stiffness.
    EI_healthy_actual23 = factor23 * EI_healthy;
    EI_damaged_actual23 = factor23 * EI_damaged;

    % Use the nominal sensor positions.
    Yh_actual23 = compute_influence_lines( ...
        EI_healthy_actual23, L, n_elem, ...
        x_load, x_sensor);

    Yd_actual23 = compute_influence_lines( ...
        EI_damaged_actual23, L, n_elem, ...
        x_load, x_sensor);

    for ki23 = 1:n_repeat_settings23

        K23 = repeats_step23(ki23);

        for trial23 = 1:trials_step23

            Yh_avg23 = zeros(size(Yh_actual23));
            Yd_avg23 = zeros(size(Yd_actual23));

            for measurement23 = 1:K23

                % Use the same noise-seed convention as Step 22.
                rng(500000 + 1000*trial23 + measurement23, ...
                    'twister');

                Yh_noisy23 = Yh_actual23 + ...
                    sigma_measurement * randn(size(Yh_actual23));

                Yd_noisy23 = Yd_actual23 + ...
                    sigma_measurement * randn(size(Yd_actual23));

                Yh_avg23 = Yh_avg23 + Yh_noisy23;
                Yd_avg23 = Yd_avg23 + Yd_noisy23;
            end

            Yh_avg23 = Yh_avg23 / K23;
            Yd_avg23 = Yd_avg23 / K23;

            % -----------------------------------------------------
            % A. Estimate the global EI factor from healthy data.
            %
            % Yh_actual ~= beta * Yh_nominal, where beta = 1/f.
            % Therefore, estimated f = 1/beta.
            % -----------------------------------------------------
            healthy_observed23 = Yh_avg23(:);

            beta23 = ...
                (healthy_reference23' * healthy_observed23) / ...
                healthy_reference_norm23;

            if ~isfinite(beta23) || beta23 <= 0
                error(['Step 23: invalid global stiffness estimate. ', ...
                       'Check the healthy response data.']);
            end

            factor_hat23 = 1 / beta23;

            factor_estimates23(fi23,ki23,trial23) = factor_hat23;

            % Observed differential influence lines.
            b23 = Yd_avg23(:) - Yh_avg23(:);

            % -----------------------------------------------------
            % B. Uncorrected inverse fit.
            % -----------------------------------------------------
            residual_raw23 = sum( ...
                (candidate_responses - b23).^2, 1);

            [~, best_raw23] = min(residual_raw23);

            element_raw23(fi23,ki23,trial23) = ...
                candidate_elements(best_raw23);

            severity_raw23(fi23,ki23,trial23) = ...
                candidate_severities(best_raw23);

            % -----------------------------------------------------
            % C. Calibrated inverse fit.
            %
            % Scale the measured differential response to the
            % nominal global EI level before candidate matching.
            % -----------------------------------------------------
            b_calibrated23 = factor_hat23 * b23;

            residual_calibrated23 = sum( ...
                (candidate_responses - b_calibrated23).^2, 1);

            [~, best_calibrated23] = min(residual_calibrated23);

            element_calibrated23(fi23,ki23,trial23) = ...
                candidate_elements(best_calibrated23);

            severity_calibrated23(fi23,ki23,trial23) = ...
                candidate_severities(best_calibrated23);
        end
    end
end

% -------------------------------------------------------------
% REPORT COMPARISON
% -------------------------------------------------------------
fprintf('\n--- STEP 23: GLOBAL EI CALIBRATION VALIDATION ---\n');
fprintf('Trials per condition: %d\n', trials_step23);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('True stiffness loss: %.1f%%\n', 100*damage_severity);
fprintf('Calibration uses healthy influence-line measurements.\n\n');

fprintf(['Actual EI | Repeats | Estimated EI factor | ', ...
         'Raw loc(%%) | Cal loc(%%) | Raw loss(%%) | ', ...
         'Cal loss(%%) | Raw MAE(pp) | Cal MAE(pp)\n']);

fprintf(['----------|---------|--------------------|', ...
         '------------|------------|-------------|', ...
         '-------------|-------------|------------\n']);

for fi23 = 1:n_factors23
    for ki23 = 1:n_repeat_settings23

        f_estimates23 = squeeze( ...
            factor_estimates23(fi23,ki23,:));

        raw_sev23 = squeeze(severity_raw23(fi23,ki23,:));
        cal_sev23 = squeeze(severity_calibrated23(fi23,ki23,:));

        raw_elem23 = squeeze(element_raw23(fi23,ki23,:));
        cal_elem23 = squeeze(element_calibrated23(fi23,ki23,:));

        mean_f23 = mean(f_estimates23);
        raw_loc23 = 100 * mean(raw_elem23 == damage_elem);
        cal_loc23 = 100 * mean(cal_elem23 == damage_elem);

        mean_raw_loss23 = 100 * mean(raw_sev23);
        mean_cal_loss23 = 100 * mean(cal_sev23);

        raw_mae23 = 100 * mean(abs(raw_sev23 - damage_severity));
        cal_mae23 = 100 * mean(abs(cal_sev23 - damage_severity));

        fprintf(['%9.2f | %7d | %19.4f | %10.1f | ', ...
                 '%10.1f | %11.2f | %11.2f | ', ...
                 '%11.2f | %10.2f\n'], ...
            EI_factors_step23(fi23), ...
            repeats_step23(ki23), ...
            mean_f23, ...
            raw_loc23, ...
            cal_loc23, ...
            mean_raw_loss23, ...
            mean_cal_loss23, ...
            raw_mae23, ...
            cal_mae23);
    end
end

% Plot severity error before and after calibration.
figure('Name', 'Step 23: Global EI Calibration');

raw_mae_plot23 = zeros(n_factors23, n_repeat_settings23);
cal_mae_plot23 = zeros(n_factors23, n_repeat_settings23);

for fi23 = 1:n_factors23
    for ki23 = 1:n_repeat_settings23

        raw_values23 = squeeze(severity_raw23(fi23,ki23,:));
        cal_values23 = squeeze(severity_calibrated23(fi23,ki23,:));

        raw_mae_plot23(fi23,ki23) = ...
            100 * mean(abs(raw_values23 - damage_severity));

        cal_mae_plot23(fi23,ki23) = ...
            100 * mean(abs(cal_values23 - damage_severity));
    end
end

plot(EI_factors_step23, raw_mae_plot23(:,1), '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);
hold on;

plot(EI_factors_step23, cal_mae_plot23(:,1), '-s', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

plot(EI_factors_step23, raw_mae_plot23(:,2), '--o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

plot(EI_factors_step23, cal_mae_plot23(:,2), '--s', ...
    'LineWidth', 1.5, 'MarkerSize', 7);

xlabel('Actual EI / nominal EI');
ylabel('Mean absolute severity error (percentage points)');
title('Damage-Severity Error Before and After EI Calibration');

legend('Raw, 1 measurement', ...
       'Calibrated, 1 measurement', ...
       'Raw, 64 measurements', ...
       'Calibrated, 64 measurements', ...
       'Location', 'best');

grid on;
hold off;

%% STEP 24: COMBINED EI AND SENSOR-POSITION MISMATCH

factors24 = [0.80 1.00 1.20];
offsets24 = [0 0.05 0.10];   % metres
repeats24 = [1 64];
trials24 = 50;

pattern24 = [1 -1 1];

nF24 = numel(factors24);
nO24 = numel(offsets24);
nK24 = numel(repeats24);

loc_raw24 = zeros(nF24,nO24,nK24);
loc_cal24 = zeros(nF24,nO24,nK24);
loss_raw24 = zeros(nF24,nO24,nK24);
loss_cal24 = zeros(nF24,nO24,nK24);
mae_raw24 = zeros(nF24,nO24,nK24);
mae_cal24 = zeros(nF24,nO24,nK24);
factor_hat_mean24 = zeros(nF24,nO24,nK24);

% Nominal healthy reference used for global EI calibration.
healthy_ref24 = Y_healthy_clean(:);
healthy_ref_norm24 = sum(healthy_ref24.^2);

for fi24 = 1:nF24

    f24 = factors24(fi24);

    EIh_actual24 = f24 * EI_healthy;
    EId_actual24 = f24 * EI_damaged;

    for oi24 = 1:nO24

        offset24 = offsets24(oi24);

        % Simulated actual sensor positions.
        x_sensor_actual24 = x_sensor + offset24 * pattern24;

        if any(x_sensor_actual24 <= 0) || ...
                any(x_sensor_actual24 >= L)
            error('Step 24: sensor position lies outside the span.');
        end

        Yh_actual24 = compute_influence_lines( ...
            EIh_actual24, L, n_elem, x_load, x_sensor_actual24);

        Yd_actual24 = compute_influence_lines( ...
            EId_actual24, L, n_elem, x_load, x_sensor_actual24);

        for ki24 = 1:nK24

            K24 = repeats24(ki24);

            elem_raw_trials24 = zeros(trials24,1);
            elem_cal_trials24 = zeros(trials24,1);
            sev_raw_trials24 = zeros(trials24,1);
            sev_cal_trials24 = zeros(trials24,1);
            factor_trials24 = zeros(trials24,1);

            for trial24 = 1:trials24

                Yh_sum24 = zeros(size(Yh_actual24));
                Yd_sum24 = zeros(size(Yd_actual24));

                for rep24 = 1:K24

                    rng(600000 + 1000*trial24 + rep24,'twister');

                    Yh_sum24 = Yh_sum24 + Yh_actual24 + ...
                        sigma_measurement * randn(size(Yh_actual24));

                    Yd_sum24 = Yd_sum24 + Yd_actual24 + ...
                        sigma_measurement * randn(size(Yd_actual24));
                end

                Yh_avg24 = Yh_sum24 / K24;
                Yd_avg24 = Yd_sum24 / K24;

                % Differential observation.
                b24 = Yd_avg24(:) - Yh_avg24(:);

                % Estimate global EI factor from healthy observations.
                beta24 = (healthy_ref24' * Yh_avg24(:)) ...
                    / healthy_ref_norm24;

                if ~isfinite(beta24) || beta24 <= 0
                    error('Step 24: invalid global EI calibration.');
                end

                f_hat24 = 1 / beta24;
                factor_trials24(trial24) = f_hat24;

                % Uncalibrated candidate fit.
                residual_raw24 = sum( ...
                    (candidate_responses - b24).^2, 1);

                [~, idx_raw24] = min(residual_raw24);

                elem_raw_trials24(trial24) = ...
                    candidate_elements(idx_raw24);

                sev_raw_trials24(trial24) = ...
                    candidate_severities(idx_raw24);

                % Calibrated candidate fit.
                b_cal24 = f_hat24 * b24;

                residual_cal24 = sum( ...
                    (candidate_responses - b_cal24).^2, 1);

                [~, idx_cal24] = min(residual_cal24);

                elem_cal_trials24(trial24) = ...
                    candidate_elements(idx_cal24);

                sev_cal_trials24(trial24) = ...
                    candidate_severities(idx_cal24);
            end

            % Summarize all trials for this condition.
            loc_raw24(fi24,oi24,ki24) = ...
                100 * mean(elem_raw_trials24 == damage_elem);

            loc_cal24(fi24,oi24,ki24) = ...
                100 * mean(elem_cal_trials24 == damage_elem);

            loss_raw24(fi24,oi24,ki24) = ...
                100 * mean(sev_raw_trials24);

            loss_cal24(fi24,oi24,ki24) = ...
                100 * mean(sev_cal_trials24);

            mae_raw24(fi24,oi24,ki24) = ...
                100 * mean(abs(sev_raw_trials24 - damage_severity));

            mae_cal24(fi24,oi24,ki24) = ...
                100 * mean(abs(sev_cal_trials24 - damage_severity));

            factor_hat_mean24(fi24,oi24,ki24) = ...
                mean(factor_trials24);
        end
    end
end

fprintf('\n--- STEP 24: COMBINED MODEL-MISMATCH TEST ---\n');
fprintf('Trials per condition: %d\n', trials24);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('True severity: %.1f%%\n\n',100*damage_severity);

fprintf(['EI factor | Offset(cm) | Repeats | EI estimate | ', ...
    'Raw loc(%%) | Cal loc(%%) | Raw loss(%%) | ', ...
    'Cal loss(%%) | Raw MAE(pp) | Cal MAE(pp)\n']);

fprintf(['----------|------------|---------|-------------|', ...
    '------------|------------|-------------|', ...
    '------------|-------------|------------\n']);

for fi24 = 1:nF24
    for oi24 = 1:nO24
        for ki24 = 1:nK24

            fprintf('%9.2f | %10.1f | %7d | %11.4f | %10.1f | %10.1f | %11.2f | %10.2f | %11.2f | %10.2f\n', ...
                factors24(fi24), ...
                100*offsets24(oi24), ...
                repeats24(ki24), ...
                factor_hat_mean24(fi24,oi24,ki24), ...
                loc_raw24(fi24,oi24,ki24), ...
                loc_cal24(fi24,oi24,ki24), ...
                loss_raw24(fi24,oi24,ki24), ...
                loss_cal24(fi24,oi24,ki24), ...
                mae_raw24(fi24,oi24,ki24), ...
                mae_cal24(fi24,oi24,ki24));
        end
    end
end

%% STEP 25: COMBINED MODEL MISMATCH AND NOISE STRESS TEST

factors25 = [0.80 1.00 1.20];
offsets25 = [0 0.10];          % metres
noise_levels25 = [1 2 5];      % multiples of original 1% noise SD
repeats25 = [1 64];
trials25 = 50;

pattern25 = [1 -1 1];

nF25 = numel(factors25);
nO25 = numel(offsets25);
nN25 = numel(noise_levels25);
nK25 = numel(repeats25);

% Results indexed as:
% stiffness factor x sensor offset x noise level x repeat count
loc_raw25 = zeros(nF25,nO25,nN25,nK25);
loc_cal25 = zeros(nF25,nO25,nN25,nK25);
mae_raw25 = zeros(nF25,nO25,nN25,nK25);
mae_cal25 = zeros(nF25,nO25,nN25,nK25);
loss_raw25 = zeros(nF25,nO25,nN25,nK25);
loss_cal25 = zeros(nF25,nO25,nN25,nK25);
factor_hat_mean25 = zeros(nF25,nO25,nN25,nK25);

% Exact nominal-model reference, as used in Steps 23 and 24.
healthy_ref25 = Y_healthy_clean(:);
healthy_ref_norm25 = sum(healthy_ref25.^2);

if healthy_ref_norm25 <= eps
    error('Step 25: healthy reference response has zero norm.');
end

for fi25 = 1:nF25

    f25 = factors25(fi25);

    % Actual global stiffness differs from the nominal model.
    EIh_actual25 = f25 * EI_healthy;
    EId_actual25 = f25 * EI_damaged;

    for oi25 = 1:nO25

        offset25 = offsets25(oi25);

        % Actual sensor positions differ from the nominal positions.
        x_sensor_actual25 = x_sensor + offset25 * pattern25;

        if any(x_sensor_actual25 <= 0) || ...
                any(x_sensor_actual25 >= L)
            error('Step 25: a sensor lies outside the span.');
        end

        % Generate responses at actual sensor positions.
        Yh_actual25 = compute_influence_lines( ...
            EIh_actual25, L, n_elem, ...
            x_load, x_sensor_actual25);

        Yd_actual25 = compute_influence_lines( ...
            EId_actual25, L, n_elem, ...
            x_load, x_sensor_actual25);

        for ni25 = 1:nN25

            % Original sigma_measurement represents the 1% noise SD.
            sigma25 = sigma_measurement * noise_levels25(ni25);

            for ki25 = 1:nK25

                K25 = repeats25(ki25);

                elem_raw_trials25 = zeros(trials25,1);
                elem_cal_trials25 = zeros(trials25,1);
                sev_raw_trials25 = zeros(trials25,1);
                sev_cal_trials25 = zeros(trials25,1);
                factor_trials25 = zeros(trials25,1);

                for trial25 = 1:trials25

                    Yh_sum25 = zeros(size(Yh_actual25));
                    Yd_sum25 = zeros(size(Yd_actual25));

                    for rep25 = 1:K25

                        % Use matched random draws across conditions.
                        rng(700000 + 1000*trial25 + rep25, 'twister');

                        Yh_sum25 = Yh_sum25 + Yh_actual25 + ...
                            sigma25 * randn(size(Yh_actual25));

                        Yd_sum25 = Yd_sum25 + Yd_actual25 + ...
                            sigma25 * randn(size(Yd_actual25));
                    end

                    Yh_avg25 = Yh_sum25 / K25;
                    Yd_avg25 = Yd_sum25 / K25;

                    % Differential observation.
                    b25 = Yd_avg25(:) - Yh_avg25(:);

                    % Estimate global EI factor from the healthy response.
                    beta25 = (healthy_ref25' * Yh_avg25(:)) ...
                        / healthy_ref_norm25;

                    if ~isfinite(beta25) || beta25 <= 0
                        error('Step 25: invalid global stiffness estimate.');
                    end

                    factor_hat25 = 1 / beta25;
                    factor_trials25(trial25) = factor_hat25;

                    % A. Uncalibrated inverse fit.
                    residual_raw25 = sum( ...
                        (candidate_responses - b25).^2, 1);

                    [~, idx_raw25] = min(residual_raw25);

                    elem_raw_trials25(trial25) = ...
                        candidate_elements(idx_raw25);

                    sev_raw_trials25(trial25) = ...
                        candidate_severities(idx_raw25);

                    % B. Calibrated inverse fit.
                    b_cal25 = factor_hat25 * b25;

                    residual_cal25 = sum( ...
                        (candidate_responses - b_cal25).^2, 1);

                    [~, idx_cal25] = min(residual_cal25);

                    elem_cal_trials25(trial25) = ...
                        candidate_elements(idx_cal25);

                    sev_cal_trials25(trial25) = ...
                        candidate_severities(idx_cal25);
                end

                % Localization success rates.
                loc_raw25(fi25,oi25,ni25,ki25) = ...
                    100 * mean(elem_raw_trials25 == damage_elem);

                loc_cal25(fi25,oi25,ni25,ki25) = ...
                    100 * mean(elem_cal_trials25 == damage_elem);

                % Mean estimated severities, expressed as percentages.
                loss_raw25(fi25,oi25,ni25,ki25) = ...
                    100 * mean(sev_raw_trials25);

                loss_cal25(fi25,oi25,ni25,ki25) = ...
                    100 * mean(sev_cal_trials25);

                % Mean absolute error in percentage points.
                mae_raw25(fi25,oi25,ni25,ki25) = ...
                    100 * mean(abs(sev_raw_trials25 - damage_severity));

                mae_cal25(fi25,oi25,ni25,ki25) = ...
                    100 * mean(abs(sev_cal_trials25 - damage_severity));

                factor_hat_mean25(fi25,oi25,ni25,ki25) = ...
                    mean(factor_trials25);
            end
        end
    end
end

%% PRINT STEP 25 RESULTS

fprintf('\n--- STEP 25: COMBINED MISMATCH AND NOISE TEST ---\n');
fprintf('Trials per condition: %d\n', trials25);
fprintf('True damaged element: %d\n', damage_elem);
fprintf('True damage severity: %.1f%%\n', 100*damage_severity);
fprintf('Noise levels are multiples of the original 1%% noise SD.\n\n');

fprintf(['EI factor | Offset(cm) | Noise(%%) | Repeats | ', ...
    'EI estimate | Raw loc(%%) | Cal loc(%%) | ', ...
    'Raw loss(%%) | Cal loss(%%) | Raw MAE | Cal MAE\n']);

fprintf(['----------|------------|----------|---------|', ...
    '-------------|------------|------------|', ...
    '-------------|-------------|----------|---------\n']);

for fi25 = 1:nF25
    for oi25 = 1:nO25
        for ni25 = 1:nN25
            for ki25 = 1:nK25

                fprintf('%9.2f | %10.1f | %8.1f | %7d | %11.4f | %10.1f | %10.1f | %11.2f | %11.2f | %8.2f | %8.2f\n', ...
                    factors25(fi25), ...
                    100*offsets25(oi25), ...
                    noise_levels25(ni25), ...
                    repeats25(ki25), ...
                    factor_hat_mean25(fi25,oi25,ni25,ki25), ...
                    loc_raw25(fi25,oi25,ni25,ki25), ...
                    loc_cal25(fi25,oi25,ni25,ki25), ...
                    loss_raw25(fi25,oi25,ni25,ki25), ...
                    loss_cal25(fi25,oi25,ni25,ki25), ...
                    mae_raw25(fi25,oi25,ni25,ki25), ...
                    mae_cal25(fi25,oi25,ni25,ki25));
            end
        end
    end
end

%% PLOT CALIBRATION BENEFIT

figure('Name', 'Step 25: Calibration Under Combined Uncertainty');

% Compare the most difficult tested condition:
% 20% stiffness mismatch and 10 cm sensor offset.
idx_f25 = find(factors25 == 0.80 | factors25 == 1.20);
idx_o25 = find(offsets25 == 0.10);

hold on;

for jj25 = 1:numel(idx_f25)

    ff25 = idx_f25(jj25);

    raw_curve25 = squeeze( ...
        mae_raw25(ff25,idx_o25,:,2));

    cal_curve25 = squeeze( ...
        mae_cal25(ff25,idx_o25,:,2));

    plot(noise_levels25, raw_curve25, '-o', ...
        'LineWidth', 1.5, 'MarkerSize', 7);

    plot(noise_levels25, cal_curve25, '--s', ...
        'LineWidth', 1.5, 'MarkerSize', 7);
end

xlabel('Noise level (multiple of original 1% noise SD)');
ylabel('Mean absolute severity error (percentage points)');
title('Calibration Under Combined Model Uncertainty');

legend('EI factor 0.80, raw', 'EI factor 0.80, calibrated', ...
       'EI factor 1.20, raw', 'EI factor 1.20, calibrated', ...
       'Location', 'best');

grid on;
hold off;

fprintf('\n--- Inverse analysis ---\n');
fprintf('Lambda: %.4e\n', lambda_selected);
fprintf('Residual norm: %.4e m/N\n', final_residual);
fprintf('Morozov target: %.4e m/N\n', morozov_target);
fprintf('Selection status: %s\n', lambda_status);
fprintf('\nElement | True EI loss (%%) | Estimated EI loss (%%)\n');
fprintf('--------|------------------|----------------------\n');
for e = 1:n_elem
    fprintf('%7d | %16.3f | %20.3f\n', ...
        e, 100*d_true(e), 100*d_est(e));
end

%% 6. DAMAGE INDICATORS (FOR DETECTION, NOT DIRECT LOCALIZATION)
% Difference indicator: aggregate the observed influence-line differences
% over sensors at each LOAD position. Smoothing is for display only.
delta_Y_obs = Y_damaged_obs - Y_healthy_obs;
DI = sqrt(mean(delta_Y_obs.^2, 1)) / max(abs(Y_healthy_clean(:)));
DI_display = movmean(DI, 3);

% Influence-line curvature indicator: second derivative with respect to
% LOAD POSITION. This is not the physical beam curvature d^2w/dx^2 along
% the bridge span. Numerical second derivatives amplify measurement noise.
dx_load = x_load(2) - x_load(1);
curv_healthy = second_derivative_rows(Y_healthy_obs, dx_load);
curv_damaged = second_derivative_rows(Y_damaged_obs, dx_load);
delta_curv = curv_damaged - curv_healthy;
CDI = sqrt(mean(delta_curv.^2, 1));
CDI_display = movmean(CDI, 5);
CDI_scale = max(sqrt(mean(curv_healthy.^2, 1)));
if CDI_scale > 0
    CDI_display = CDI_display / CDI_scale;
end

%% 7. PLOTS
figure('Name', 'Influence lines', 'Color', 'w');
for s = 1:numel(x_sensor)
    subplot(numel(x_sensor), 1, s);
    plot(x_load, Y_healthy_clean(s,:), 'LineWidth', 1.5); hold on;
    plot(x_load, Y_damaged_clean(s,:), '--', 'LineWidth', 1.5);
    grid on;
    ylabel('Deflection (m/N)');
    title(sprintf('%s sensor (x = %.2f m)', sensor_names{s}, x_sensor(s)));
    if s == 1
        legend('Healthy', 'Damaged (noise-free)', 'Location', 'best');
    end
end
xlabel('Moving-load position (m)');

figure('Name', 'Influence-line change indicators', 'Color', 'w');
subplot(2,1,1);
plot(x_load, DI_display, 'LineWidth', 1.5);
grid on; xlabel('Moving-load position (m)');
ylabel('Normalized DI');
title('Influence-line displacement-change indicator');

subplot(2,1,2);
plot(x_load, CDI_display, 'LineWidth', 1.5);
grid on; xlabel('Moving-load position (m)');
ylabel('Normalized CDI');
title('Influence-line curvature-change indicator (second derivative vs load position)');

figure('Name', 'Estimated element damage', 'Color', 'w');
element_midpoints = ((1:n_elem) - 0.5) * (L/n_elem);
bar(element_midpoints, 100*d_est, 0.8); hold on;
plot(element_midpoints, 100*d_true, 'ro', ...
    'MarkerFaceColor', 'r', 'MarkerSize', 6);
grid on; xlabel('Element midpoint (m)');
ylabel('Estimated fractional EI loss (%)');
title('Bounded Tikhonov damage estimate');
legend('Estimated', 'True simulated damage', 'Location', 'best');
ylim([0, max(10, 1.1*100*max([d_true; d_est]))]);

figure('Name', 'Morozov discrepancy curve', 'Color', 'w');
semilogx(lambda_grid, residual_norm, 'LineWidth', 1.5); hold on;
yline(morozov_target, '--', 'Noise discrepancy target');
xline(lambda_selected, ':', 'Selected lambda');
grid on; xlabel('\lambda');
ylabel('||Jd_\lambda - b||_2 (m/N)');
title('Regularization selection by discrepancy principle');

%% ----------------------- LOCAL FUNCTIONS -------------------------------

function Y = compute_influence_lines(EI_e, L, n_elem, x_load, x_sensor)
% Compute static influence lines for vertical displacement at arbitrary
% sensor coordinates under a unit downward point load moving over the span.

    n_nodes = n_elem + 1;
    ndof = 2*n_nodes;
    le = L/n_elem;
    K = zeros(ndof, ndof);

    for e = 1:n_elem
        ke = EI_e(e)/le^3 * ...
            [ 12,       6*le, -12,       6*le;
              6*le, 4*le^2,  -6*le, 2*le^2;
             -12,      -6*le,  12,      -6*le;
              6*le, 2*le^2,  -6*le, 4*le^2 ];
        dofs = [2*e-1, 2*e, 2*e+1, 2*e+2];
        K(dofs, dofs) = K(dofs, dofs) + ke;
    end

    % Simply supported: vertical displacement fixed at both supports;
    % rotations are free.
    fixed_dofs = [1, 2*n_nodes-1];
    free_dofs = setdiff(1:ndof, fixed_dofs).';
    Kff = K(free_dofs, free_dofs);

    Y = zeros(numel(x_sensor), numel(x_load));
    for j = 1:numel(x_load)
        x = x_load(j);
        [e_load, xi_load] = locate_element(x, L, n_elem);
        Nload = hermite_shape(xi_load, le);
        load_dofs = [2*e_load-1; 2*e_load; 2*e_load+1; 2*e_load+2];

        F = zeros(ndof, 1);
        % Downward unit point load. The Hermite shape functions give the
        % consistent equivalent nodal force vector at the load position.
        F(load_dofs) = F(load_dofs) - Nload;

        q = zeros(ndof, 1);
        q(free_dofs) = Kff \ F(free_dofs);

        for s = 1:numel(x_sensor)
            [e_s, xi_s] = locate_element(x_sensor(s), L, n_elem);
            Ns = hermite_shape(xi_s, le);
            sensor_dofs = [2*e_s-1; 2*e_s; 2*e_s+1; 2*e_s+2];
            Y(s,j) = Ns.' * q(sensor_dofs);
        end
    end
end

function [e, xi] = locate_element(x, L, n_elem)
% Return element number and local coordinate xi in [0,1].
    le = L/n_elem;
    x_clamped = min(max(x, 0), L);
    e = min(floor(x_clamped/le) + 1, n_elem);
    xi = (x_clamped - (e-1)*le)/le;
    xi = min(max(xi, 0), 1);
end

function N = hermite_shape(xi, le)
% Cubic Hermite displacement interpolation shape functions.
% DOF order: [v1, theta1, v2, theta2].
    N1 = 1 - 3*xi^2 + 2*xi^3;
    N2 = le * (xi - 2*xi^2 + xi^3);
    N3 = 3*xi^2 - 2*xi^3;
    N4 = le * (-xi^2 + xi^3);
    N = [N1; N2; N3; N4];
end

function d = solve_box_tikhonov(J, b, lambda, max_iter, tol)
% Solve min 0.5*||J*d-b||^2 + 0.5*lambda*||d||^2 subject to 0<=d<=1
% using projected FISTA. This avoids a dependency on lsqlin/lsqnonneg.
    n = size(J, 2);
    d = zeros(n, 1);
    z = d;
    t = 1;
    lipschitz = norm(J, 2)^2 + lambda;
    if ~(isfinite(lipschitz) && lipschitz > 0)
        error('Invalid Lipschitz constant in solve_box_tikhonov.');
    end
    step = 1/lipschitz;

    for iter = 1:max_iter
        grad = J.'*(J*z - b) + lambda*z;
        d_new = min(max(z - step*grad, 0), 1);

        t_new = (1 + sqrt(1 + 4*t^2))/2;
        z = d_new + ((t - 1)/t_new)*(d_new - d);

        if norm(d_new - d, 2) <= tol * max(1, norm(d, 2))
            d = d_new;
            break;
        end
        d = d_new;
        t = t_new;
    end
end

function d2Y = second_derivative_rows(Y, dx)
% Second numerical derivative along columns (load-position coordinate).
    d2Y = zeros(size(Y));
    for r = 1:size(Y,1)
        first = gradient(Y(r,:), dx);
        d2Y(r,:) = gradient(first, dx);
    end
end



function Y_smooth = sgolay_smooth_rows(Y, window_length, polynomial_order)
% Local polynomial smoothing without requiring a toolbox.
% Each row is smoothed along its load-position samples.

    if mod(window_length, 2) ~= 1
        error('window_length must be odd.');
    end

    if polynomial_order < 0 || ...
            polynomial_order >= window_length
        error('Require 0 <= polynomial_order < window_length.');
    end

    n_samples = size(Y, 2);

    if window_length > n_samples
        error('window_length cannot exceed the number of samples.');
    end

    half_window = floor(window_length / 2);
    Y_smooth = zeros(size(Y));

    for r = 1:size(Y, 1)
        for j = 1:n_samples

            idx = max(1, j-half_window): ...
                min(n_samples, j+half_window);

            % Coordinates relative to the sample being estimated.
            t = (idx - j).';

            A = ones(numel(idx), polynomial_order + 1);

            for p = 1:polynomial_order
                A(:, p+1) = t.^p;
            end

            % Fit a local polynomial by least squares.
            coefficients = A \ Y(r, idx).';

            % The intercept is the fitted value at the target sample.
            Y_smooth(r, j) = coefficients(1);
        end
    end
end


function d2Y = sgolay_second_derivative_rows( ...
    Y, dx, window_length, polynomial_order)
% Estimate the second derivative using local polynomial least squares.
% This function does not require the Signal Processing Toolbox.

    if mod(window_length, 2) ~= 1
        error('window_length must be odd.');
    end

    if polynomial_order < 2 || ...
            polynomial_order >= window_length
        error('Require 2 <= polynomial_order < window_length.');
    end

    if dx <= 0
        error('dx must be positive.');
    end

    n_samples = size(Y, 2);

    if window_length > n_samples
        error('window_length exceeds the number of samples.');
    end

    half_window = floor(window_length/2);
    d2Y = zeros(size(Y));

    for r = 1:size(Y,1)
        for j = 1:n_samples

            idx = max(1, j-half_window): ...
                min(n_samples, j+half_window);

            % Coordinates relative to the target sample, in metres.
            t = ((idx-j)*dx).';

            A = ones(numel(idx), polynomial_order+1);

            for p = 1:polynomial_order
                A(:,p+1) = t.^p;
            end

            coefficients = A \ Y(r,idx).';

            % For polynomial coefficients in powers of t,
            % d^2Y/dx^2 at the target point is 2*coefficient 3.
            d2Y(r,j) = 2*coefficients(3);
        end
    end
end

function scores = cdi_element_scores_step16(CDI, x_load, L, n_elem)

    scores = zeros(1, n_elem);
    element_length = L / n_elem;

    for e = 1:n_elem
        left_edge = (e - 1) * element_length;
        right_edge = e * element_length;

        if e < n_elem
            idx = x_load >= left_edge & x_load < right_edge;
        else
            idx = x_load >= left_edge & x_load <= right_edge;
        end

        if any(idx)
            scores(e) = mean(CDI(idx));
        else
            scores(e) = NaN;
        end
    end
end