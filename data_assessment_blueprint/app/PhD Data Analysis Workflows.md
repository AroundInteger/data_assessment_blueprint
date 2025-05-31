# PhD Data Analysis Workflows: Detailed Examples

## Workflow 1: Sports Analytics - KPI Match Outcome Prediction

**Research Question:** *Is there a robust subset of KPIs that can predict match outcomes in team sports?*

### Data Context
- **Dataset:** Season KPI data (10-15 teams, home/away format)
- **Target:** Binary match outcome (win/loss)
- **Features:** 30-50 KPIs per match
- **Sample Size:** ~300-400 matches per season

### Phase 1: Data Exploration & Quality Assessment

**Step 1.1: Data Structure Validation**

# Essential checks for sports data
def validate_sports_data(df):
    # Check home/away balance
    home_away_balance = df['venue'].value_counts()
    
    # Verify each team plays expected number of games
    games_per_team = df.groupby('team').size()
    expected_games = (n_teams - 1) * 2  # Each team plays others home/away
    
    # Check for missing matches in season
    missing_fixtures = identify_missing_fixtures(df, expected_games)
    
    # Temporal consistency
    season_timeline = df['match_date'].sort_values()
    
    return validation_report


**Step 1.2: KPI Distribution Analysis**
- Document KPI calculation methods and validate against match videos (sample)
- Identify extreme values and verify against match reports
- Check for systematic differences between home/away performances
- Assess KPI correlation structure preliminary view

**Key Questions for Phase 1:**
- Are there recording inconsistencies across venues?
- Do KPI distributions make practical sense given the sport?
- Are there obvious data entry errors or missing values?

### Phase 2: Measurement Uncertainty & Signal Quality

**Step 2.1: KPI Reliability Assessment**

# Cross-validate KPIs against independent measures where possible
def assess_kpi_reliability(kpi_data, validation_data=None):
    # Test-retest reliability (if available)
    if validation_data:
        reliability_coeffs = []
        for kpi in kpi_columns:
            icc = calculate_icc(kpi_data[kpi], validation_data[kpi])
            reliability_coeffs.append(icc)
    
    # Internal consistency checks
    # E.g., possession % + opponent possession % should ≈ 100%
    consistency_checks = perform_consistency_checks(kpi_data)
    
    return reliability_report


**Step 2.2: Systematic Bias Detection**
- Home advantage effects in KPI measurements
- Referee or scorer effects (if identifiable)
- Seasonal drift in KPI values
- Equipment or rule changes mid-season

**Key Outputs:**
- KPI reliability coefficients
- Bias correction factors (if needed)
- Uncertainty estimates for each KPI

### Phase 3: Statistical Characterization

**Step 3.1: Distributional Analysis**

def comprehensive_distribution_analysis(df, kpi_columns):
    results = {}
    
    for kpi in kpi_columns:
        # Multiple normality tests
        shapiro_stat, shapiro_p = shapiro(df[kpi])
        anderson_stat, anderson_crit = anderson(df[kpi], dist='norm')
        jarque_bera_stat, jarque_bera_p = jarque_bera(df[kpi])
        
        # Practical significance of deviations
        effect_size = cohen_d_from_normal(df[kpi])
        
        # Alternative distributions
        best_fit_dist = find_best_distribution(df[kpi])
        
        results[kpi] = {
            'normality_tests': {
                'shapiro': (shapiro_stat, shapiro_p),
                'anderson': (anderson_stat, anderson_crit),
                'jarque_bera': (jarque_bera_stat, jarque_bera_p)
            },
            'deviation_effect_size': effect_size,
            'best_fit_distribution': best_fit_dist,
            'transformation_needed': effect_size > 0.5  # Practical threshold
        }
    
    return results


**Step 3.2: Correlation and Multicollinearity**

# Comprehensive correlation analysis
correlation_matrix = df[kpi_columns].corr()
high_correlations = find_pairs_above_threshold(correlation_matrix, 0.8)

# VIF analysis for multicollinearity
vif_scores = calculate_vif(df[kpi_columns])
problematic_features = vif_scores[vif_scores > 5]


**Step 3.3: Target Variable Analysis**
- Win/loss balance overall and by team
- Home advantage quantification
- Temporal clustering of results
- Class imbalance handling strategy

**Decision Point:** Based on distributional analysis, decide on:
- Transformation strategies (log, Box-Cox, rank-based)
- Correlation-based feature filtering thresholds
- Sampling strategies for imbalanced classes

### Phase 4: Feature Selection & Model Development

**Step 4.1: Iterative mRMR-Random Forest Pipeline**

def iterative_mrmr_rf_selection(X_train, y_train, max_k=20):
    """
    Iteratively apply mRMR and evaluate with Random Forest
    Returns optimal feature subset and performance metrics
    """
    performance_scores = {}
    feature_sets = {}
    
    # Stratified CV setup
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in range(1, min(max_k + 1, X_train.shape[1])):
        # mRMR feature selection
        selected_features = mrmr_classif(
            X=X_train, 
            y=y_train, 
            K=k,
            relevance='f_classif',
            redundancy='pearson'
        )
        
        # Random Forest evaluation with CV
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Initial, will be tuned later
            random_state=42,
            class_weight='balanced'  # Handle imbalance
        )
        
        # Cross-validation on training data only
        cv_scores = cross_val_score(
            rf_model, 
            X_train[selected_features], 
            y_train,
            cv=cv_strategy,
            scoring='roc_auc'  # Better for imbalanced classes
        )
        
        performance_scores[k] = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'features': selected_features
        }
        
        feature_sets[k] = selected_features
    
    # Find optimal k
    optimal_k = max(performance_scores.keys(), 
                   key=lambda k: performance_scores[k]['mean_cv_score'])
    
    return optimal_k, performance_scores, feature_sets


**Step 4.2: Hyperparameter Optimization**

def optimize_hyperparameters(X_train, y_train, selected_features):
    """
    Optimize RF hyperparameters on selected features
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf_model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    )
    
    # Stratified CV for hyperparameter tuning
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train[selected_features], y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


**Step 4.3: Multiple Pipeline Comparison**

# Handle different feature engineering approaches
pipelines = {
    'raw_features': {
        'features': raw_kpi_features,
        'preprocessing': StandardScaler()
    },
    'engineered_features': {
        'features': engineered_kpi_features,
        'preprocessing': RobustScaler()  # Less sensitive to outliers
    }
}

best_pipeline_results = {}
for pipeline_name, pipeline_config in pipelines.items():
    # Apply full mRMR-RF process
    optimal_k, scores, features = iterative_mrmr_rf_selection(
        pipeline_config['features'], y_train
    )
    
    # Hyperparameter optimization
    best_model, best_params = optimize_hyperparameters(
        pipeline_config['features'], y_train, features[optimal_k]
    )
    
    best_pipeline_results[pipeline_name] = {
        'optimal_k': optimal_k,
        'selected_features': features[optimal_k],
        'best_model': best_model,
        'cv_performance': scores[optimal_k]
    }


### Phase 5: Model Evaluation & Validation

**Step 5.1: Test Set Evaluation with Bootstrap CI**

def bootstrap_evaluation(model, X_test, y_test, n_bootstrap=1000):
    """
    Bootstrap confidence intervals for model performance
    """
    bootstrap_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_boot = X_test.iloc[indices]
        y_boot = y_test.iloc[indices]
        
        # Get predictions
        y_pred = model.predict(X_boot)
        y_pred_proba = model.predict_proba(X_boot)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_boot, y_pred)
        auc_score = roc_auc_score(y_boot, y_pred_proba)
        f1 = f1_score(y_boot, y_pred)
        
        bootstrap_scores.append({
            'accuracy': accuracy,
            'auc': auc_score,
            'f1': f1
        })
    
    # Calculate confidence intervals
    ci_results = {}
    for metric in ['accuracy', 'auc', 'f1']:
        scores = [score[metric] for score in bootstrap_scores]
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores, 97.5)
        ci_results[metric] = {
            'mean': np.mean(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }
    
    return ci_results, bootstrap_scores


**Step 5.2: Feature Importance Analysis**

def comprehensive_feature_importance(model, X_test, y_test, selected_features):
    """
    Multiple methods for feature importance
    """
    # Mean Decrease Accuracy (Permutation Importance)
    perm_importance = permutation_importance(
        model, X_test[selected_features], y_test,
        n_repeats=30,  # Multiple permutations for stability
        random_state=42,
        scoring='roc_auc'
    )
    
    # SHAP values for local interpretability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[selected_features])
    
    # Feature importance summary
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'mda_importance': perm_importance.importances_mean,
        'mda_std': perm_importance.importances_std,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('mda_importance', ascending=False)
    
    return importance_df, shap_values


**Key Validation Outputs:**
- 95% CI for accuracy, AUC, F1-score
- Feature importance rankings with confidence intervals
- Model interpretability analysis
- Performance stability assessment

### Phase 6: Limitation Documentation & Reporting

**Systematic Limitation Assessment:**

1. **Data Limitations**
   - Sample size constraints (single season)
   - League-specific generalizability
   - KPI measurement precision bounds
   - Missing confounding variables (player injuries, etc.)

2. **Methodological Constraints**
   - Feature selection stability across seasons
   - Model assumptions about KPI-outcome relationships
   - Bootstrap assumptions and limitations
   - Class imbalance handling effectiveness

3. **Practical Implementation Boundaries**
   - Real-time prediction feasibility
   - KPI availability timing constraints
   - Model update frequency requirements
   - Threshold selection for practical use

**Final Deliverables:**
- Robust feature subset with confidence intervals
- Model performance with uncertainty quantification
- Feature importance with interpretability analysis
- Comprehensive limitations documentation

---

## Workflow 2: Fluorescence Time-Series - Drug Dose-Response Analysis

**Research Question:** *What is the dose-response relationship for cellular drug treatments, and how does response timing vary across cell populations?*

### Data Context
- **Dataset:** Time-lapse fluorescence microscopy
- **Structure:** Cells nested within wells, wells within plates
- **Variables:** Treatment dose, time points, fluorescence intensity
- **Sample Size:** ~500-2000 cells across 6-8 dose levels

### Phase 1: Hierarchical Data Structure Assessment

**Step 1.1: Experimental Design Validation**

def validate_experimental_design(metadata):
    # Check treatment randomization
    randomization_check = assess_randomization(metadata['treatment_layout'])
    
    # Verify replicate structure
    replicates_per_dose = metadata.groupby('dose').size()
    
    # Check time point consistency
    time_point_coverage = assess_temporal_coverage(metadata)
    
    # Plate effects assessment
    plate_effects = test_plate_effects(metadata)
    
    return design_validation_report


**Step 1.2: Cell Tracking Quality**

def assess_tracking_quality(tracking_data):
    # Track continuity analysis
    track_lengths = tracking_data.groupby('cell_id')['time_point'].count()
    dropout_patterns = analyze_dropout_patterns(tracking_data)
    
    # Segmentation quality metrics
    segmentation_scores = calculate_segmentation_quality(tracking_data)
    
    # Edge effects identification
    edge_effects = identify_edge_effects(tracking_data, well_boundaries)
    
    return tracking_quality_report


### Phase 2: Signal Processing & Photophysical Corrections

**Step 2.1: Photobleaching Correction**

def photobleaching_correction(fluorescence_data):
    # Single exponential decay model for each cell
    corrected_data = []
    
    for cell_id in fluorescence_data['cell_id'].unique():
        cell_data = fluorescence_data[fluorescence_data['cell_id'] == cell_id]
        
        # Fit exponential decay to control wells
        if cell_data['treatment'].iloc[0] == 'control':
            # Fit: F(t) = F0 * exp(-k*t)
            decay_params = fit_exponential_decay(cell_data['fluorescence'], cell_data['time'])
            
            # Apply correction
            corrected_fluorescence = correct_photobleaching(
                cell_data['fluorescence'], 
                decay_params, 
                cell_data['time']
            )
        else:
            corrected_fluorescence = cell_data['fluorescence']
        
        corrected_data.append(corrected_fluorescence)
    
    return corrected_data


**Step 2.2: Background Subtraction & SNR Assessment**

def signal_quality_assessment(corrected_data):
    # Background fluorescence characterization
    background_stats = calculate_background_stats(corrected_data)
    
    # Signal-to-noise ratio calculation
    snr_per_cell = []
    for cell_data in corrected_data:
        signal_variance = np.var(cell_data['fluorescence'])
        noise_variance = estimate_noise_variance(cell_data)
        snr = signal_variance / noise_variance
        snr_per_cell.append(snr)
    
    # Quality filtering thresholds
    min_snr_threshold = np.percentile(snr_per_cell, 25)  # Bottom quartile
    
    return snr_per_cell, min_snr_threshold


### Phase 3: Hierarchical Statistical Modeling

**Step 3.1: Nested Data Structure Analysis**

def analyze_nested_structure(data):
    # Intraclass correlation coefficients
    # Level 1: Cells within wells
    icc_wells = calculate_icc(data, groupby='well_id', response='fluorescence')
    
    # Level 2: Wells within plates  
    icc_plates = calculate_icc(data, groupby='plate_id', response='fluorescence')
    
    # Variance partitioning
    variance_components = partition_variance(data, ['cell_id', 'well_id', 'plate_id'])
    
    return {
        'icc_wells': icc_wells,
        'icc_plates': icc_plates,
        'variance_components': variance_components
    }


**Step 3.2: Dose-Response Distribution Analysis**

def dose_response_distributions(data):
    distribution_analysis = {}
    
    for dose in data['dose'].unique():
        dose_data = data[data['dose'] == dose]['fluorescence']
        
        # Test multiple distributions
        distributions = ['normal', 'lognormal', 'gamma', 'beta']
        fit_results = {}
        
        for dist in distributions:
            params = fit_distribution(dose_data, dist)
            aic = calculate_aic(dose_data, dist, params)
            fit_results[dist] = {'params': params, 'aic': aic}
        
        # Select best fitting distribution
        best_dist = min(fit_results.keys(), key=lambda d: fit_results[d]['aic'])
        distribution_analysis[dose] = {
            'best_distribution': best_dist,
            'all_fits': fit_results
        }
    
    return distribution_analysis


### Phase 4: Mixed-Effects Dose-Response Modeling

**Step 4.1: Hierarchical Model Development**

def fit_dose_response_model(data):
    # Four-parameter logistic with random effects
    # Model: response ~ bottom + (top - bottom) / (1 + (dose/EC50)^hill_slope)
    # With random effects for wells and plates
    
    model_formula = """
    fluorescence ~ bottom + (top - bottom) / (1 + (dose/ec50)**hill_slope) + 
                   (1|well_id) + (1|plate_id) + (1|cell_id)
    """
    
    # Fit using nlme or similar
    dose_response_model = fit_nonlinear_mixed_effects(
        formula=model_formula,
        data=data,
        start_values={'bottom': 0, 'top': 100, 'ec50': 10, 'hill_slope': 1}
    )
    
    return dose_response_model


**Step 4.2: Bootstrap Confidence Intervals for Parameters**

def bootstrap_dose_response_ci(data, model, n_bootstrap=1000):
    """
    Bootstrap confidence intervals for dose-response parameters
    """
    bootstrap_params = []
    
    for i in range(n_bootstrap):
        # Resample at the well level to maintain structure
        wells = data['well_id'].unique()
        bootstrap_wells = np.random.choice(wells, size=len(wells), replace=True)
        
        bootstrap_data = []
        for well in bootstrap_wells:
            well_data = data[data['well_id'] == well].copy()
            bootstrap_data.append(well_data)
        
        bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)
        
        # Refit model
        try:
            bootstrap_model = fit_nonlinear_mixed_effects(
                formula=model.formula,
                data=bootstrap_df,
                start_values=model.params
            )
            bootstrap_params.append(bootstrap_model.params)
        except:
            continue  # Skip failed fits
    
    # Calculate confidence intervals
    param_ci = {}
    for param in ['bottom', 'top', 'ec50', 'hill_slope']:
        param_values = [p[param] for p in bootstrap_params]
        param_ci[param] = {
            'mean': np.mean(param_values),
            'ci_lower': np.percentile(param_values, 2.5),
            'ci_upper': np.percentile(param_values, 97.5)
        }
    
    return param_ci


### Phase 5: Temporal Dynamics Analysis

**Step 5.1: Time-to-Response Analysis**

def analyze_response_timing(data, threshold_response=50):
    """
    Analyze when cells reach response threshold
    """
    time_to_response = []
    
    for cell_id in data['cell_id'].unique():
        cell_data = data[data['cell_id'] == cell_id].sort_values('time_point')
        
        # Find first time point exceeding threshold
        response_times = cell_data[cell_data['fluorescence'] > threshold_response]
        
        if len(response_times) > 0:
            time_to_response.append({
                'cell_id': cell_id,
                'dose': cell_data['dose'].iloc[0],
                'time_to_response': response_times['time_point'].iloc[0]
            })
    
    return pd.DataFrame(time_to_response)


**Step 5.2: Population Heterogeneity Analysis**

def analyze_population_heterogeneity(data):
    # Mixture model to identify subpopulations
    heterogeneity_analysis = {}
    
    for dose in data['dose'].unique():
        dose_responses = data[data['dose'] == dose]['max_fluorescence']
        
        # Fit Gaussian mixture models with different components
        n_components_range = range(1, 5)
        mixture_results = {}
        
        for n_comp in n_components_range:
            gmm = GaussianMixture(n_components=n_comp, random_state=42)
            gmm.fit(dose_responses.values.reshape(-1, 1))
            
            aic = gmm.aic(dose_responses.values.reshape(-1, 1))
            bic = gmm.bic(dose_responses.values.reshape(-1, 1))
            
            mixture_results[n_comp] = {
                'model': gmm,
                'aic': aic,
                'bic': bic
            }
        
        # Select optimal number of components
        best_n_comp = min(mixture_results.keys(), 
                         key=lambda n: mixture_results[n]['bic'])
        
        heterogeneity_analysis[dose] = {
            'optimal_components': best_n_comp,
            'mixture_model': mixture_results[best_n_comp]['model'],
            'population_proportions': mixture_results[best_n_comp]['model'].weights_
        }
    
    return heterogeneity_analysis


### Phase 6: Biological Validation & Limitations

**Validation Steps:**
1. **Positive/Negative Control Validation**
   - Expected dose-response curves for known compounds
   - Vehicle control stability over time
   - Positive control EC50 comparison with literature

2. **Cross-Plate Reproducibility**
   - Inter-plate variability assessment
   - Batch effect correction validation
   - Technical replicate consistency

3. **Biological Plausibility**
   - Hill slope biological interpretation
   - EC50 values compared to known pharmacology
   - Time-course consistency with mechanism of action

**Documented Limitations:**
- **Technical Constraints**
  - Photobleaching correction accuracy
  - Cell tracking reliability bounds
  - Temporal resolution limitations
  - Fluorescence quantification precision

- **Biological Constraints**
  - Cell line representativeness
  - Drug delivery assumptions
  - Cellular heterogeneity interpretation
  - Translation to in vivo relevance

- **Statistical Constraints**
  - Mixed-effects model assumptions
  - Bootstrap sampling limitations
  - Multiple comparison adjustments
  - Dose range and spacing effects

---

## Workflow 3: Athletic Race Performance - Time-Based Analysis

**Research Question:** *What are the optimal pacing strategies for different race distances, and can we predict performance outcomes from early race splits?*

### Data Context
- **Dataset:** Race timing data from chip-based systems
- **Structure:** Multiple races, multiple athletes, split times at regular intervals
- **Variables:** Split times, cumulative times, environmental conditions, athlete demographics
- **Sample Size:** 500-2000 race performances across different distances (5K, 10K, half-marathon, marathon)

### Phase 1: Race Data Quality and Consistency

**Step 1.1: Timing System Validation**

def validate_race_timing_data(race_data):
    validation_report = {}
    
    # Check for impossible split times
    impossible_splits = identify_impossible_splits(race_data)
    
    # Verify cumulative time consistency
    cumulative_consistency = check_cumulative_consistency(race_data)
    
    # DNF (Did Not Finish) pattern analysis
    dnf_patterns = analyze_dnf_patterns(race_data)
    
    # Course measurement validation
    course_distances = validate_course_distances(race_data)
    
    # Environmental data completeness
    weather_completeness = assess_weather_data_completeness(race_data)
    
    validation_report = {
        'impossible_splits': impossible_splits,
        'cumulative_consistency': cumulative_consistency,
        'dnf_patterns': dnf_patterns,
        'course_distances': course_distances,
        'weather_completeness': weather_completeness
    }
    
    return validation_report


**Step 1.2: Performance Standardization**

def standardize_race_performances(race_data):
    """
    Account for course difficulty and environmental conditions
    """
    standardized_data = race_data.copy()
    
    # Course difficulty adjustment (if multiple courses)
    if 'course_id' in race_data.columns:
        course_factors = calculate_course_difficulty_factors(race_data)
        standardized_data['adjusted_time'] = adjust_for_course_difficulty(
            race_data['finish_time'], 
            race_data['course_id'], 
            course_factors
        )
    
    # Environmental condition adjustments
    # Temperature, humidity, wind effects
    environmental_factors = calculate_environmental_impact(
        race_data[['temperature', 'humidity', 'wind_speed']]
    )
    
    standardized_data['weather_adjusted_time'] = adjust_for_weather(
        standardized_data['adjusted_time'],
        environmental_factors
    )
    
    # Age-grading (if demographic data available)
    if 'age' in race_data.columns:
        standardized_data['age_graded_performance'] = calculate_age_grading(
            standardized_data['weather_adjusted_time'],
            race_data['age'],
            race_data['gender'],
            race_data['distance']
        )
    
    return standardized_data


### Phase 2: Pacing Pattern Analysis

**Step 2.1: Split Time Quality Assessment**

def assess_split_time_quality(split_data):
    """
    Evaluate measurement uncertainty and systematic errors in split timing
    """
    quality_metrics = {}
    
    # Split time precision analysis
    split_precision = []
    for split_point in split_data.columns:
        if 'split' in split_point:
            # Compare with neighboring splits for consistency
            precision = assess_split_precision(split_data[split_point])
            split_precision.append(precision)
    
    # Identify timing mat malfunctions
    timing_anomalies = detect_timing_anomalies(split_data)
    
    # Calculate measurement uncertainty for each split
    split_uncertainty = estimate_split_uncertainty(split_data)
    
    quality_metrics = {
        'split_precision': split_precision,
        'timing_anomalies': timing_anomalies,
        'measurement_uncertainty': split_uncertainty
    }
    
    return quality_metrics


**Step 2.2: Pacing Strategy Classification**

def classify_pacing_strategies(split_data):
    """
    Identify different pacing patterns using clustering
    """
    # Calculate relative pace for each split
    relative_paces = calculate_relative_paces(split_data)
    
    # Smooth pacing curves to reduce noise
    smoothed_paces = smooth_pacing_curves(relative_paces)
    
    # K-means clustering to identify pacing strategies
    n_clusters_range = range(3, 8)
    clustering_results = {}
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(smoothed_paces)
        
        # Evaluate clustering quality
        silhouette_score = calculate_silhouette_score(smoothed_paces, cluster_labels)
        
        clustering_results[n_clusters] = {
            'labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'silhouette_score': silhouette_score
        }
    
    # Select optimal number of clusters
    optimal_n_clusters = max(clustering_results.keys(), 
                           key=lambda k: clustering_results[k]['silhouette_score'])
    
    return clustering_results[optimal_n_clusters], clustering_results


### Phase 3: Statistical Analysis of Race Performance

**Step 3.1: Pacing Pattern Statistical Characterization**

def analyze_pacing_distributions(pacing_data, performance_outcomes):
    """
    Statistical analysis of pacing patterns and their relationship to performance
    """
    statistical_analysis = {}
    
    # Test normality of pacing metrics
    pacing_metrics = ['pace_variability', 'positive_split_magnitude', 'fade_index']
    
    for metric in pacing_metrics:
        # Multiple normality tests
        shapiro_stat, shapiro_p = shapiro(pacing_data[metric])
        anderson_stat, anderson_crit = anderson(pacing_data[metric])
        
        # Test for appropriate transformations if non-normal
        if shapiro_p < 0.05:
            # Try log transformation
            log_transformed = np.log(pacing_data[metric] + 1)  # +1 to handle zeros
            log_shapiro_stat, log_shapiro_p = shapiro(log_transformed)
            
            # Try Box-Cox transformation
            boxcox_transformed, lambda_param = boxcox(pacing_data[metric] + 1)
            boxcox_shapiro_stat, boxcox_shapiro_p = shapiro(boxcox_transformed)
            
            best_transformation = select_best_transformation(
                [shapiro_p, log_shapiro_p, boxcox_shapiro_p],
                ['none', 'log', 'boxcox']
            )
        else:
            best_transformation = 'none'
        
        statistical_analysis[metric] = {
            'original_normality': (shapiro_stat, shapiro_p),
            'anderson_darling': (anderson_stat, anderson_crit),
            'best_transformation': best_transformation
        }
    
    return statistical_analysis


**Step 3.2: Performance Prediction Modeling**

def develop_performance_prediction_model(split_data, outcomes):
    """
    Build models to predict race outcomes from early splits
    """
    prediction_models = {}
    
    # Different prediction horizons (25%, 50%, 75% of race)
    prediction_points = [0.25, 0.50, 0.75]
    
    for prediction_point in prediction_points:
        # Extract features up to prediction point
        features = extract_early_race_features(split_data, prediction_point)
        
        # Feature engineering
        engineered_features = engineer_pacing_features(features)
        
        # Train/test split (stratified by performance quartiles)
        X_train, X_test, y_train, y_test = train_test_split(
            engineered_features, outcomes,
            test_size=0.2, 
            stratify=pd.qcut(outcomes, q=4, labels=False),
            random_state=42
        )
        
        # Model comparison
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'svr': SVR()
        }
        
        model_performance = {}
        for model_name, model in models.items():
            # Cross-validation on training set
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=5, scoring='neg_mean_absolute_error'
            )
            
            # Fit and evaluate on test set
            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)
            
            # Calculate multiple metrics
            mae = mean_absolute_error(y_test, test_predictions)
            rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            r2 = r2_score(y_test, test_predictions)
            
            model_performance[model_name] = {
                'cv_mae': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_mae': mae,
                'test_rmse': rmse,
                'test_r2': r2,
                'fitted_model': model
            }
        
        # Select best model
        best_model = min(model_performance.keys(), 
                        key=lambda k: model_performance[k]['test_mae'])
        
        prediction_models[prediction_point] = {
            'best_model': best_model,
            'model_performance': model_performance,
            'feature_importance': get_feature_importance(
                model_performance[best_model]['fitted_model'], 
                engineered_features.columns
            )
        }
    
    return prediction_models


### Phase 4: Bootstrap Validation and Uncertainty Quantification

**Step 4.1: Prediction Confidence Intervals**

def bootstrap_prediction_intervals(model, X_test, y_test, n_bootstrap=1000):
    """
    Bootstrap confidence intervals for predictions
    """
    bootstrap_predictions = []
    bootstrap_metrics = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_boot = X_test.iloc[indices]
        y_boot = y_test.iloc[indices]
        
        # Generate predictions
        predictions = model.predict(X_boot)
        
        # Calculate metrics
        mae = mean_absolute_error(y_boot, predictions)
        rmse = np.sqrt(mean_squared_error(y_boot, predictions))
        
        bootstrap_predictions.append(predictions)
        bootstrap_metrics.append({'mae': mae, 'rmse': rmse})
    
    # Calculate prediction intervals
    prediction_intervals = np.percentile(bootstrap_predictions, [2.5, 97.5], axis=0)
    
    # Calculate metric confidence intervals
    mae_ci = np.percentile([m['mae'] for m in bootstrap_metrics], [2.5, 97.5])
    rmse_ci = np.percentile([m['rmse'] for m in bootstrap_metrics], [2.5, 97.5])
    
    return {
        'prediction_intervals': prediction_intervals,
        'mae_ci': mae_ci,
        'rmse_ci': rmse_ci
    }


### Phase 5: Practical Application and Limitations

**Performance Insights:**
1. **Optimal Pacing Strategies** by race distance and athlete ability
2. **Early Warning Systems** for performance deterioration
3. **Strategic Decision Points** during races
4. **Environmental Impact Quantification**

**Documented Limitations:**
- **Timing System Precision** - typically ±1-2 seconds per split
- **Course Variability** - elevation, surface, turns affecting pace
- **Environmental Corrections** - model accuracy for extreme conditions
- **Individual Variability** - population models vs. individual optimization
- **Sample Bias** - competitive vs. recreational athlete populations

---

## Workflow 4: Cycling Performance - IMU/GPS Sensor Integration

**Research Question:** *How do biomechanical efficiency metrics from IMU sensors relate to GPS-derived performance outcomes in competitive cycling?*

### Data Context
- **Dataset:** Synchronized IMU and GPS data from cycling computers
- **Structure:** Multiple rides per cyclist, multiple cyclists across performance levels
- **Variables:** Power output, cadence, speed, elevation, accelerometer/gyroscope data
- **Sample Size:** 50-100 cyclists, 10-50 rides each, 1-10 Hz sampling depending on metric

### Phase 1: Multi-Sensor Data Integration and Synchronization

**Step 1.1: Sensor Data Quality Assessment**

def assess_cycling_sensor_quality(sensor_data):
    """
    Comprehensive quality assessment for cycling sensor data
    """
    quality_report = {}
    
    # GPS data quality
    gps_quality = assess_gps_quality(sensor_data['gps'])
    quality_report['gps'] = {
        'satellite_count': gps_quality['avg_satellites'],
        'hdop_values': gps_quality['hdop_distribution'],
        'speed_consistency': gps_quality['speed_consistency'],
        'elevation_smoothness': gps_quality['elevation_smoothness']
    }
    
    # IMU data quality
    imu_quality = assess_imu_quality(sensor_data['imu'])
    quality_report['imu'] = {
        'accelerometer_noise': imu_quality['accel_noise_levels'],
        'gyroscope_drift': imu_quality['gyro_drift_assessment'],
        'sampling_rate_consistency': imu_quality['sampling_consistency'],
        'calibration_status': imu_quality['calibration_check']
    }
    
    # Power meter data quality (if available)
    if 'power' in sensor_data:
        power_quality = assess_power_meter_quality(sensor_data['power'])
        quality_report['power'] = {
            'zero_offset_stability': power_quality['zero_offset'],
            'dropout_percentage': power_quality['data_completeness'],
            'spike_detection': power_quality['anomalous_values']
        }
    
    # Synchronization assessment
    sync_quality = assess_sensor_synchronization(sensor_data)
    quality_report['synchronization'] = {
        'time_alignment_accuracy': sync_quality['alignment_error'],
        'sampling_rate_compatibility': sync_quality['rate_matching']
    }
    
    return quality_report


**Step 1.2: Data Fusion and Alignment**

def fuse_cycling_sensor_data(gps_data, imu_data, power_data=None):
    """
    Temporal alignment and fusion of multi-rate sensor data
    """
    # Establish common time base (usually GPS time)
    reference_timestamps = gps_data['timestamp']
    
    # Interpolate high-frequency IMU data to GPS timestamps
    imu_interpolated = interpolate_imu_to_gps_time(imu_data, reference_timestamps)
    
    # Handle power data synchronization
    if power_data is not None:
        power_synchronized = synchronize_power_data(power_data, reference_timestamps)
    else:
        power_synchronized = None
    
    # Create unified dataset
    fused_data = pd.DataFrame({
        'timestamp': reference_timestamps,
        'latitude': gps_data['latitude'],
        'longitude': gps_data['longitude'],
        'speed': gps_data['speed'],
        'elevation': gps_data['elevation'],
        'accel_x': imu_interpolated['accel_x'],
        'accel_y': imu_interpolated['accel_y'],
        'accel_z': imu_interpolated['accel_z'],
        'gyro_x': imu_interpolated['gyro_x'],
        'gyro_y': imu_interpolated['gyro_y'],
        'gyro_z': imu_interpolated['gyro_z']
    })
    
    if power_synchronized is not None:
        fused_data['power'] = power_synchronized['power']
        fused_data['cadence'] = power_synchronized['cadence']
    
    return fused_data


### Phase 2: Signal Processing and Feature Engineering

**Step 2.1: GPS-Derived Performance Metrics**

def calculate_gps_performance_metrics(gps_data):
    """
    Extract performance metrics from GPS data
    """
    performance_metrics = {}
    
    # Speed analysis
    performance_metrics['speed_stats'] = {
        'avg_speed': np.mean(gps_data['speed']),
        'max_speed': np.max(gps_data['speed']),
        'speed_variability': np.std(gps_data['speed']),
        'speed_percentiles': np.percentile(gps_data['speed'], [25, 50, 75, 90, 95])
    }
    
    # Elevation analysis
    elevation_gain, elevation_loss = calculate_elevation_changes(gps_data['elevation'])
    performance_metrics['elevation_stats'] = {
        'total_elevation_gain': elevation_gain,
        'total_elevation_loss': elevation_loss,
        'max_gradient': calculate_max_gradient(gps_data),
        'climbing_speed': calculate_climbing_speed(gps_data)
    }
    
    # Acceleration patterns
    acceleration = calculate_acceleration_from_speed(gps_data['speed'], gps_data['timestamp'])
    performance_metrics['acceleration_stats'] = {
        'avg_acceleration': np.mean(acceleration),
        'acceleration_variability': np.std(acceleration),
        'sprint_efforts': identify_sprint_efforts(acceleration),
        'braking_events': identify_braking_events(acceleration)
    }
    
    # Segment analysis (climbs, flats, descents)
    segments = classify_terrain_segments(gps_data)
    performance_metrics['segment_performance'] = analyze_segment_performance(gps_data, segments)
    
    return performance_metrics


**Step 2.2: IMU-Derived Biomechanical Metrics**

def calculate_imu_biomechanical_metrics(imu_data):
    """
    Extract biomechanical efficiency metrics from IMU sensors
    """
    biomech_metrics = {}
    
    # Pedaling smoothness from accelerometer
    pedaling_smoothness = calculate_pedaling_smoothness(imu_data['accel_z'])
    biomech_metrics['pedaling_efficiency'] = {
        'smoothness_index': pedaling_smoothness,
        'pedal_stroke_consistency': calculate_pedal_stroke_consistency(imu_data),
        'power_application_efficiency': calculate_power_application_efficiency(imu_data)
    }
    
    # Body stability from accelerometer
    body_stability = calculate_body_stability(imu_data[['accel_x', 'accel_y', 'accel_z']])
    biomech_metrics['stability'] = {
        'lateral_stability': body_stability['lateral'],
        'vertical_stability': body_stability['vertical'],
        'overall_stability_score': body_stability['overall']
    }
    
    # Cadence analysis from gyroscope
    cadence_from_imu = extract_cadence_from_gyroscope(imu_data['gyro_z'])
    biomech_metrics['cadence_analysis'] = {
        'avg_cadence': np.mean(cadence_from_imu),
        'cadence_variability': np.std(cadence_from_imu),
        'cadence_consistency': calculate_cadence_consistency(cadence_from_imu)
    }
    
    # Bike handling metrics
    handling_metrics = calculate_bike_handling_metrics(imu_data)
    biomech_metrics['bike_handling'] = {
        'cornering_smoothness': handling_metrics['cornering'],
        'braking_smoothness': handling_metrics['braking'],
        'steering_stability': handling_metrics['steering']
    }
    
    return biomech_metrics


### Phase 3: Multi-Modal Statistical Analysis

**Step 3.1: Feature Selection for Multi-Modal Data**

def multi_modal_feature_selection(gps_features, imu_features, performance_outcome):
    """
    Feature selection across GPS and IMU modalities
    """
    # Combine all features
    all_features = pd.concat([gps_features, imu_features], axis=1)
    
    # Handle different correlation structures within modalities
    # GPS features may be highly correlated (speed, distance, etc.)
    # IMU features may have different correlation patterns
    
    # Hierarchical feature selection
    # Level 1: Within-modality feature selection
    gps_selected = mrmr_classif(gps_features, performance_outcome, K=10)
    imu_selected = mrmr_classif(imu_features, performance_outcome, K=10)
    
    # Level 2: Cross-modality integration
    combined_selected = list(gps_selected) + list(imu_selected)
    
    # Level 3: Final optimization across modalities
    final_features = mrmr_classif(
        all_features[combined_selected], 
        performance_outcome, 
        K=15
    )
    
    # Validate feature stability across cross-validation folds
    feature_stability = assess_feature_selection_stability(
        all_features, performance_outcome, final_features
    )
    
    return final_features, feature_stability


**Step 3.2: Hierarchical Modeling for Nested Data**

def hierarchical_performance_model(features, outcomes, cyclist_ids):
    """
    Account for repeated measures within cyclists
    """
    # Prepare data for mixed effects modeling
    model_data = pd.DataFrame(features)
    model_data['outcome'] = outcomes
    model_data['cyclist_id'] = cyclist_ids
    
    # Test for significant cyclist-level random effects
    icc_cyclist = calculate_icc(model_data, groupby='cyclist_id', response='outcome')
    
    if icc_cyclist > 0.1:  # Substantial clustering
        # Mixed effects model
        model_formula = f"outcome ~ {' + '.join(features.columns)} + (1|cyclist_id)"
        mixed_model = fit_mixed_effects_model(model_formula, model_data)
        
        return mixed_model, {'model_type': 'mixed_effects', 'icc': icc_cyclist}
    else:
        # Regular regression sufficient
        regular_model = fit_regular_regression(features, outcomes)
        return regular_model, {'model_type': 'regular', 'icc': icc_cyclist}


### Phase 4: Cross-Modal Validation and Performance Prediction

**Step 4.1: Modality-Specific Model Comparison**

def compare_modality_contributions(gps_features, imu_features, outcomes):
    """
    Compare predictive power of GPS vs IMU vs combined models
    """
    model_comparison = {}
    
    # GPS-only model
    gps_model = RandomForestRegressor(random_state=42)
    gps_cv_scores = cross_val_score(gps_model, gps_features, outcomes, cv=5, scoring='r2')
    
    # IMU-only model
    imu_model = RandomForestRegressor(random_state=42)
    imu_cv_scores = cross_val_score(imu_model, imu_features, outcomes, cv=5, scoring='r2')
    
    # Combined model
    combined_features = pd.concat([gps_features, imu_features], axis=1)
    combined_model = RandomForestRegressor(random_state=42)
    combined_cv_scores = cross_val_score(combined_model, combined_features, outcomes, cv=5, scoring='r2')
    
    model_comparison = {
        'gps_only': {
            'cv_r2_mean': gps_cv_scores.mean(),
            'cv_r2_std': gps_cv_scores.std(),
            'feature_count': gps_features.shape[1]
        },
        'imu_only': {
            'cv_r2_mean': imu_cv_scores.mean(),
            'cv_r2_std': imu_cv_scores.std(),
            'feature_count': imu_features.shape[1]
        },
        'combined': {
            'cv_r2_mean': combined_cv_scores.mean(),
            'cv_r2_std': combined_cv_scores.std(),
            'feature_count': combined_features.shape[1]
        }
    }
    
    # Statistical comparison of model performance
    gps_vs_combined = paired_ttest(gps_cv_scores, combined_cv_scores)
    imu_vs_combined = paired_ttest(imu_cv_scores, combined_cv_scores)
    
    model_comparison['statistical_tests'] = {
        'gps_vs_combined': gps_vs_combined,
        'imu_vs_combined': imu_vs_combined
    }
    
    return model_comparison


**Step 4.2: Real-Time Performance Monitoring**

def develop_real_time_monitoring_system(model, feature_extractors):
    """
    System for real-time performance assessment during rides
    """
    def real_time_predictor(streaming_data_window):
        # Extract features from current data window
        current_gps_features = feature_extractors['gps'](streaming_data_window['gps'])
        current_imu_features = feature_extractors['imu'](streaming_data_window['imu'])
        
        # Combine features
        current_features = np.concatenate([current_gps_features, current_imu_features])
        
        # Generate prediction with uncertainty
        prediction = model.predict([current_features])[0]
        
        # Calculate prediction uncertainty (if using ensemble methods)
        if hasattr(model, 'estimators_'):
            individual_predictions = [estimator.predict([current_features])[0] 
                                    for estimator in model.estimators_]
            prediction_std = np.std(individual_predictions)
        else:
            prediction_std = None
        
        return {
            'predicted_performance': prediction,
            'uncertainty': prediction_std,
            'timestamp': streaming_data_window['timestamp'][-1]
        }
    
    return real_time_predictor


### Phase 5: Technology Integration and Practical Limitations

**Validation Framework:**
1. **Cross-Device Validation** - Model performance across different sensor brands
2. **Environmental Robustness** - Performance in various weather/terrain conditions
3. **Real-Time Feasibility** - Computational requirements for live analysis
4. **User Acceptance** - Practical utility for cyclists and coaches

**Comprehensive Limitations Documentation:**

**Technical Limitations:**
- **GPS Accuracy** - Multipath errors in urban/forested areas, typical ±3-5m accuracy
- **IMU Sensor Drift** - Gyroscope bias drift over long rides, accelerometer noise
- **Synchronization Precision** - Inter-sensor timing accuracy ±10-50ms
- **Power Meter Variability** - ±1-3% accuracy, temperature sensitivity
- **Sampling Rate Heterogeneity** - Different sensors at different frequencies

**Methodological Constraints:**
- **Individual Calibration** - Generic models vs. personalized algorithms
- **Environmental Generalization** - Training data representativeness
- **Feature Engineering Stability** - Robustness across different ride types
- **Model Interpretability** - Complex multi-modal models vs. actionable insights

**Practical Implementation Boundaries:**
- **Battery Life** - High-frequency sampling impact on device longevity
- **Storage Requirements** - Data volume for multi-sensor recording
- **Processing Power** - Real-time analysis computational demands
- **User Interface** - Complexity of multi-metric feedback systems
- **Cost-Benefit** - Equipment cost vs. performance improvement magnitude

**Statistical Limitations:**
- **Sample Size** - Sufficient data across cyclist ability levels and conditions
- **Temporal Generalization** - Performance changes over training seasons
- **Cross-Population Validity** - Road vs. mountain vs. track cycling differences
- **Confounding Variables** - Equipment differences, training status, motivation

---

## Workflow 5: Longitudinal Biomechanical Analysis - Injury Risk Prediction

**Research Question:** *Can kinematic movement patterns predict injury risk in athletes, and which movement phases are most predictive?*

### Data Context
- **Dataset:** 3D motion capture during athletic movements
- **Structure:** Multiple trials per athlete, multiple athletes per sport
- **Variables:** Joint angles, velocities, accelerations over time
- **Sample Size:** 100-200 athletes, 5-10 trials each, 500-1000 Hz sampling

### Phase 1: Movement Data Quality Assessment

**Step 1.1: Motion Capture Quality Control**

def motion_capture_quality_control(motion_data):
    quality_metrics = {}
    
    # Marker occlusion analysis
    occlusion_rates = calculate_occlusion_rates(motion_data)
    
    # Gap filling quality assessment
    gap_filling_accuracy = assess_gap_filling_accuracy(motion_data)
    
    # Coordinate system consistency
    coordinate_consistency = check_coordinate_system_consistency(motion_data)
    
    # Calibration residuals
    calibration_quality = assess_calibration_quality(motion_data)
    
    quality_metrics = {
        'occlusion_rates': occlusion_rates,
        'gap_filling_accuracy': gap_filling_accuracy,
        'coordinate_consistency': coordinate_consistency,
        'calibration_quality': calibration_quality
    }
    
    return quality_metrics


**Step 1.2: Movement Trial Standardization**

def standardize_movement_trials(motion_data):
    standardized_trials = []
    
    for trial in motion_data:
        # Identify movement phases (e.g., loading, propulsion, landing)
        movement_phases = identify_movement_phases(trial)
        
        # Time normalization (0-100% of movement cycle)
        normalized_trial = time_normalize_trial(trial, movement_phases)
        
        # Spatial normalization (body dimensions)
        spatially_normalized = spatial_normalize_trial(normalized_trial, 
                                                     athlete_anthropometrics)
        
        standardized_trials.append(spatially_normalized)
    
    return standardized_trials


### Phase 2: Biomechanical Signal Processing

**Step 2.1: Filtering and Noise Reduction**

def optimize_filtering_parameters(kinematic_data):
    """
    Optimize filter cutoff using residual analysis
    """
    cutoff_frequencies = np.arange(5, 25, 1)  # Hz
    residual_scores = []
    
    for cutoff in cutoff_frequencies:
        # Apply Butterworth filter
        filtered_data = butterworth_filter(kinematic_data, cutoff=cutoff, order=4)
        
        # Calculate residuals between raw and filtered
        residuals = kinematic_data - filtered_data
        
        # Residual analysis score (balance smoothing vs. data retention)
        residual_score = calculate_residual_score(residuals)
        residual_scores.append(residual_score)
    
    # Find optimal cutoff (minimum residual score)
    optimal_cutoff = cutoff_frequencies[np.argmin(residual_scores)]
    
    return optimal_cutoff, residual_scores


**Step 2.2: Derived Variable Calculation**

def calculate_biomechanical_variables(filtered_data):
    """
    Calculate velocities, accelerations, and joint angles
    """
    biomech_variables = {}
    
    # Joint angle calculations
    joint_angles = calculate_joint_angles(filtered_data['positions'])
    
    # Velocity calculations (first derivative)
    velocities = calculate_velocities(filtered_data['positions'])
    
    # Acceleration calculations (second derivative)
    accelerations = calculate_accelerations(velocities)
    
    # Error propagation for derived variables
    velocity_uncertainty = propagate_uncertainty(
        filtered_data['position_uncertainty'], 
        derivative_order=1
    )
    
    acceleration_uncertainty = propagate_uncertainty(
        velocity_uncertainty, 
        derivative_order=1
    )
    
    biomech_variables = {
        'joint_angles': joint_angles,
        'velocities': velocities,
        'accelerations': accelerations,
        'velocity_uncertainty': velocity_uncertainty,
        'acceleration_uncertainty': acceleration_uncertainty
    }
    
    return biomech_variables


### Phase 3: Functional Data Analysis

**Step 3.1: Movement Pattern Classification**

def classify_movement_patterns(biomech_data):
    """
    Principal component analysis of movement curves
    """
    # Functional PCA for time-series data
    fpca_results = {}
    
    for joint in ['knee', 'hip', 'ankle']:
        joint_curves = extract_joint_curves(biomech_data, joint)
        
        # Functional PCA
        fpca = FunctionalPCA(n_components=5)
        fpca_scores = fpca.fit_transform(joint_curves)
        
        # Explained variance
        explained_variance = fpca.explained_variance_ratio_
        
        fpca_results[joint] = {
            'scores': fpca_scores,
            'components': fpca.components_,
            'explained_variance': explained_variance
        }
    
    return fpca_results


**Step 3.2: Statistical Parametric Mapping**

def statistical_parametric_mapping(injured_group, control_group):
    """
    Time-series comparison between injured and non-injured athletes
    """
    spm_results = {}
    
    for variable in ['knee_angle', 'hip_moment', 'ground_reaction_force']:
        # Extract time-normalized curves
        injured_curves = extract_curves(injured_group, variable)
        control_curves = extract_curves(control_group, variable)
        
        # SPM t-test at each time point
        t_stats = []
        p_values = []
        
        for time_point in range(101):  # 0-100% normalized time
            injured_values = injured_curves[:, time_point]
            control_values = control_curves[:, time_point]
            
            t_stat, p_val = ttest_ind(injured_values, control_values)
            t_stats.append(t_stat)
            p_values.append(p_val)
        
        # Multiple comparison correction using random field theory
        corrected_p_values = rft_correction(p_values, smoothness=10)
        
        # Identify significant time periods
        significant_periods = identify_significant_clusters(
            corrected_p_values, 
            alpha=0.05
        )
        
        spm_results[variable] = {
            't_statistics': t_stats,
            'p_values': p_values,
            'corrected_p_values': corrected_p_values,
            'significant_periods': significant_periods
        }
    
    return spm_results


### Phase 4: Injury Risk Prediction Modeling

**Step 4.1: Longitudinal Feature Engineering**

def engineer_injury_risk_features(biomech_data, injury_outcomes):
    """
    Create features that capture injury risk patterns
    """
    risk_features = {}
    
    # Asymmetry indices
    asymmetry_features = calculate_asymmetry_indices(biomech_data)
    risk_features['asymmetry'] = asymmetry_features
    
    # Movement variability measures
    variability_features = calculate_movement_variability(biomech_data)
    risk_features['variability'] = variability_features
    
    # Peak load indicators
    peak_load_features = identify_peak_loading_patterns(biomech_data)
    risk_features['peak_loads'] = peak_load_features
    
    # Fatigue-related changes
    fatigue_features = calculate_fatigue_indicators(biomech_data)
    risk_features['fatigue'] = fatigue_features
    
    # Temporal progression features
    temporal_features = calculate_temporal_progression(biomech_data)
    risk_features['temporal'] = temporal_features
    
    return risk_features


**Step 4.2: Survival Analysis for Time-to-Injury**
python
def injury_survival_analysis(features, injury_data):
    """
    Cox proportional hazards model for injury prediction
    """
    # Prepare survival data
    survival_data = prepare_survival_data(injury_data)
    
    # Cox regression with feature selection
    cox_model = CoxPHFitter()
    
    # Feature selection using concordance index
    selected_features = select_features_by_concordance(features, survival_data)
    
    # Fit Cox model
    cox_model.fit(
        survival_data[selected_features + ['duration', 'event']], 
        duration_col='duration', 
        event_col='event'
    )
    
    # Validate proportional hazards assumption
    ph_assumption_test = cox_model.check_assumptions(survival_data)
    
    return cox_model, ph_assumption_test, selected_features


### Phase 5: Clinical Translation and Validation

**Prospective Validation:**
1. **Forward-looking injury prediction** using baseline movement patterns
2. **Intervention effectiveness** assessment through movement correction
3. **Clinical decision thresholds** for screening programs
4. **Cost-effectiveness analysis** of screening vs. treatment

**Documented Limitations:**
- **Movement standardization** - laboratory vs. field movement differences
- **Injury definition consistency** - varying injury severity classifications
- **Prediction horizon** - short-term vs. long-term risk assessment
- **Population generalizability** - sport-specific vs. general movement patterns
- **Technology requirements** - motion capture accessibility and cost

---

## Common Methodological Threads Across All Workflows

### Universal Quality Assurance Framework

**Phase 1 Commonalities:**
- **Data provenance documentation** - source, collection methods, known limitations
- **Structure validation** - completeness, consistency, format verification
- **Temporal/spatial integrity** - alignment, synchronization, missing data patterns

**Phase 2 Universals:**
- **Measurement uncertainty quantification** - precision, accuracy, systematic bias
- **Signal quality assessment** - noise characterization, artifact identification
- **Calibration validation** - drift assessment, reference standard comparison

**Phase 3 Statistical Rigor:**
- **Multi-method assumption testing** - never rely on single normality test
- **Transformation evaluation** - Box-Cox, log, rank-based alternatives
- **Effect size consideration** - practical vs. statistical significance
- **Correlation structure analysis** - temporal, spatial, hierarchical dependencies

**Phase 4 Modeling Excellence:**
- **Feature selection validation** - stability across cross-validation folds
- **Hyperparameter optimization** - proper nested CV, no test set contamination
- **Bootstrap confidence intervals** - 1000+ iterations standard
- **Model interpretability** - feature importance with uncertainty quantification

**Phase 5 Comprehensive Validation:**
- **Multiple validation strategies** - internal, external, prospective when possible
- **Limitation documentation** - technical, methodological, practical constraints
- **Uncertainty communication** - confidence intervals, prediction intervals
- **Practical implementation boundaries** - real-world applicability assessment

### Domain-Specific Adaptation Strategies

**Sports Analytics Specializations:**
- **Temporal dependencies** - seasonal effects, opponent adjustments, fatigue
- **Hierarchical structures** - players within teams, games within seasons
- **Performance context** - home/away, competition level, stakes variation
- **Environmental factors** - weather, venue, equipment differences

**Biomedical Data Considerations:**
- **Nested experimental designs** - cells within wells, subjects within groups
- **Dose-response relationships** - non-linear modeling, threshold effects
- **Biological plausibility** - mechanistic interpretation requirements
- **Regulatory compliance** - validation standards, documentation requirements

**Sensor Integration Challenges:**
- **Multi-rate data fusion** - temporal alignment, interpolation strategies
- **Cross-modal validation** - sensor redundancy, failure detection
- **Real-time processing** - computational constraints, latency requirements
- **Calibration maintenance** - drift correction, reference updating

### Decision Trees for Methodological Choices

**Statistical Approach Selection:**

1. Sample Size Assessment
   ├─ n < 30 → Emphasize non-parametric methods, bootstrap CI
   ├─ 30 ≤ n < 100 → Mixed parametric/non-parametric, robust methods
   └─ n ≥ 100 → Full parametric testing with effect size evaluation

2. Data Structure Recognition
   ├─ Independent observations → Standard statistical methods
   ├─ Temporal correlation → Time series methods, autocorrelation correction
   ├─ Hierarchical nesting → Mixed effects models, ICC assessment
   └─ Spatial correlation → Spatial statistics, clustering adjustment

3. Distributional Characteristics
   ├─ Normal distribution → Parametric methods with assumption validation
   ├─ Skewed distributions → Transformation or robust alternatives
   ├─ Bounded variables → Beta regression, logistic transformation
   └─ Count data → Poisson, negative binomial modeling


**Validation Strategy Selection:**

1. Data Availability
   ├─ Single dataset → Cross-validation, bootstrap validation
   ├─ Multiple datasets → External validation, meta-analysis
   └─ Longitudinal data → Temporal validation, prospective testing

2. Prediction Horizon
   ├─ Real-time prediction → Online learning, streaming validation
   ├─ Short-term prediction → Rolling window validation
   └─ Long-term prediction → Time series cross-validation

3. Practical Application
   ├─ Research context → Academic rigor, comprehensive documentation
   ├─ Clinical application → Regulatory validation, safety assessment
   └─ Commercial deployment → A/B testing, user acceptance validation


This comprehensive framework ensures that regardless of domain - whether sports performance, biomedical research, or sensor analytics - students follow the same rigorous analytical principles while adapting appropriately to their specific research context and data characteristics.