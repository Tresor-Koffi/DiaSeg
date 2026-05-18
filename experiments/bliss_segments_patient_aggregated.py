"""
BLISS Diagonal Segments - Patient-Level Aggregation
====================================================
Instead of clustering 53k individual segments,
aggregate segment statistics per patient, then cluster 12 patients.

Key Innovation:
- Extract segments vs reference (as before)
- Compute AGGREGATE statistics per patient
- Cluster on patient-level features (12 patients)
- Should achieve 75-90% patient accuracy!

Author: Research Team
Date: 2026-03-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.svm import SVC
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

# ============================================================================
# DTW + DIAGONAL EXTRACTION (Same as before)
# ============================================================================

def compute_dtw_with_path(seq1, seq2):
    """Compute DTW with optimal path"""
    n, m = len(seq1), len(seq2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (seq1[i-1] - seq2[j-1]) ** 2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i, j))
        candidates = [(D[i-1, j-1], i-1, j-1), (D[i-1, j], i-1, j), (D[i, j-1], i, j-1)]
        _, i, j = min(candidates)
    path.append((1, 1))
    path.reverse()
    
    directions = []
    for k in range(1, len(path)):
        di, dj = path[k][0] - path[k-1][0], path[k][1] - path[k-1][1]
        if di == 1 and dj == 1:
            directions.append('diagonal')
        elif di == 1:
            directions.append('vertical')
        else:
            directions.append('horizontal')
    
    return path, directions, D[n, m]

def extract_diagonal_segments(path, directions, ell_max=3):
    """Extract diagonal segments"""
    segments = []
    K = len(path)
    
    for i in range(1, K - 1):
        is_boundary = (directions[i-1] == 'diagonal') and (i == 1 or directions[i-2] != 'diagonal')
        if not is_boundary:
            continue
        
        s, E, breaks = i - 1, {}, 0
        for j in range(i, K):
            if j - 1 < len(directions):
                if directions[j-1] == 'diagonal':
                    E[breaks] = j
                else:
                    breaks += 1
                    if breaks > ell_max:
                        break
        
        for ell, e in E.items():
            if i < len(directions) and e - 1 < len(directions):
                if directions[i-1] == 'diagonal' and directions[e-1] == 'diagonal':
                    L = (e - i + 2) - ell
                    if L > ell:
                        segments.append({
                            'start_idx': s,
                            'end_idx': e,
                            'length': L,
                            'breaks': ell,
                        })
    
    return segments

def extract_gait_cycles(accel, phases):
    """Extract complete gait cycles"""
    cycles = []
    in_cycle = False
    cycle_start = None
    current_phases_seen = set()
    
    for i in range(len(phases)):
        current_phase = phases[i]
        
        if current_phase == 1 and not in_cycle:
            cycle_start = i
            in_cycle = True
            current_phases_seen = {1}
        
        elif in_cycle:
            current_phases_seen.add(current_phase)
            
            if current_phase == 7:
                if len(current_phases_seen) == 7:
                    cycle_end = i
                    
                    cycle = {
                        'start_idx': cycle_start,
                        'end_idx': cycle_end,
                        'length': cycle_end - cycle_start + 1,
                        'accel': accel[cycle_start:cycle_end+1],
                        'phases': phases[cycle_start:cycle_end+1]
                    }
                    cycles.append(cycle)
                
                in_cycle = False
                current_phases_seen = set()
    
    return cycles

def load_all_trials(subject_id, bliss_dir):
    """Load ALL trials for a patient"""
    
    processed_dir = bliss_dir / subject_id / "Processed Data"
    
    if not processed_dir.exists():
        return None
    
    pattern = f"{subject_id}_Trial_*_Analog_processed.csv"
    all_files = list(processed_dir.glob(pattern))
    
    valid_files = []
    for f in all_files:
        parts = f.stem.split('_')
        try:
            trial_idx = parts.index('Trial')
            trial_num_str = parts[trial_idx + 1]
            trial_num = int(trial_num_str)
            
            if trial_num < 10:
                valid_files.append((f, trial_num))
        except:
            continue
    
    if not valid_files:
        return None
    
    valid_files.sort(key=lambda x: x[1])
    
    all_cycles = []
    trials_loaded = []
    
    for filepath, trial_num in valid_files:
        
        try:
            df = pd.read_csv(filepath)
            
            if 'R_Soleus_ACC Z' in df.columns:
                accel = df['R_Soleus_ACC Z'].values
            elif 'R_Soleus_ACC X' in df.columns:
                accel = df['R_Soleus_ACC X'].values
            else:
                continue
            
            if 'phase' not in df.columns:
                continue
            
            phases = df['phase'].values.astype(int)
            
            unique_phases = np.unique(phases)
            if len(unique_phases) < 7 or max(unique_phases) < 7:
                continue
            
            cycles = extract_gait_cycles(accel, phases)
            
            if len(cycles) > 0:
                all_cycles.extend(cycles)
                trials_loaded.append(trial_num)
            
        except Exception as e:
            continue
    
    if len(all_cycles) == 0:
        return None
    
    return {
        'cycles': all_cycles,
        'subject_id': subject_id,
        'trials_loaded': trials_loaded,
        'num_cycles': len(all_cycles)
    }

def get_patient_severity(subject_id):
    """Get patient severity classification"""
    
    HEALTHY = [
        'AB2930', 'AB2931', 'AB2933', 'AB2937', 'AB2938', 
        'AB2939', 'AB2940', 'AB2941'
    ]
    
    MILD = ['AB2932', 'AB2935']
    
    SEVERE = ['AB2934', 'AB2936']
    
    if subject_id in HEALTHY:
        return 'Healthy', 0
    elif subject_id in MILD:
        return 'Mild', 1
    elif subject_id in SEVERE:
        return 'Severe', 2
    else:
        return 'Unknown', -1

def create_reference_cycle(healthy_cycles, target_length=1000):
    """Create a reference cycle by averaging all healthy cycles"""
    
    print(f"\n→ Creating reference cycle from {len(healthy_cycles)} healthy cycles...")
    
    interpolated = []
    
    for cycle in healthy_cycles:
        accel = cycle['accel']
        
        if len(accel) < 10:
            continue
        
        x_old = np.linspace(0, 1, len(accel))
        x_new = np.linspace(0, 1, target_length)
        
        f_accel = interp1d(x_old, accel, kind='cubic', fill_value='extrapolate')
        accel_interp = f_accel(x_new)
        
        interpolated.append(accel_interp)
    
    all_accel = np.array(interpolated)
    ref_accel = np.mean(all_accel, axis=0)
    ref_accel_norm = (ref_accel - np.mean(ref_accel)) / (np.std(ref_accel) + 1e-6)
    
    print(f"  ✓ Reference cycle created from {len(interpolated)} cycles")
    
    return {
        'accel_norm': ref_accel_norm,
        'length': target_length
    }

# ============================================================================
# EXTRACT SEGMENTS VS REFERENCE
# ============================================================================

def extract_segments_vs_reference(patient_cycles, reference_cycle, max_cycles=20):
    """Extract diagonal segments by comparing patient cycles to reference"""
    
    all_segments = []
    
    if len(patient_cycles) > max_cycles:
        np.random.seed(42)
        indices = np.random.choice(len(patient_cycles), max_cycles, replace=False)
        patient_cycles = [patient_cycles[i] for i in indices]
    
    ref_accel = reference_cycle['accel_norm']
    ref_length = reference_cycle['length']
    
    for cycle in patient_cycles:
        
        accel = cycle['accel']
        
        if len(accel) < 10:
            continue
        
        # Interpolate to reference length
        x_old = np.linspace(0, 1, len(accel))
        x_new = np.linspace(0, 1, ref_length)
        
        f_accel = interp1d(x_old, accel, kind='cubic', fill_value='extrapolate')
        accel_interp = f_accel(x_new)
        
        # Normalize
        accel_norm = (accel_interp - np.mean(accel_interp)) / (np.std(accel_interp) + 1e-6)
        
        # DTW vs reference
        path, directions, dtw_dist = compute_dtw_with_path(accel_norm, ref_accel)
        
        # Extract segments
        segments = extract_diagonal_segments(path, directions, ell_max=3)
        
        # Store with DTW distance
        for seg in segments:
            seg['dtw_distance'] = dtw_dist
            all_segments.append(seg)
    
    return all_segments

# ============================================================================
# AGGREGATE PATIENT-LEVEL FEATURES
# ============================================================================

def aggregate_patient_features(segments, subject_id):
    """
    Compute aggregate statistics from patient's segments
    Returns patient-level feature vector
    """
    
    if len(segments) == 0:
        return None
    
    # Extract arrays
    L_values = np.array([s['length'] for s in segments])
    ell_values = np.array([s['breaks'] for s in segments])
    dtw_values = np.array([s['dtw_distance'] for s in segments])
    
    # COMPREHENSIVE AGGREGATE FEATURES
    features = {
        'subject_id': subject_id,
        
        # LENGTH STATISTICS (10 features)
        'L_mean': np.mean(L_values),
        'L_std': np.std(L_values),
        'L_median': np.median(L_values),
        'L_min': np.min(L_values),
        'L_max': np.max(L_values),
        'L_q25': np.percentile(L_values, 25),
        'L_q75': np.percentile(L_values, 75),
        'L_iqr': np.percentile(L_values, 75) - np.percentile(L_values, 25),
        'L_skewness': stats.skew(L_values),
        'L_kurtosis': stats.kurtosis(L_values),
        
        # BREAKS STATISTICS (10 features)
        'ell_mean': np.mean(ell_values),
        'ell_std': np.std(ell_values),
        'ell_median': np.median(ell_values),
        'ell_min': np.min(ell_values),
        'ell_max': np.max(ell_values),
        'ell_q25': np.percentile(ell_values, 25),
        'ell_q75': np.percentile(ell_values, 75),
        'ell_iqr': np.percentile(ell_values, 75) - np.percentile(ell_values, 25),
        'ell_skewness': stats.skew(ell_values),
        'ell_kurtosis': stats.kurtosis(ell_values),
        
        # DTW DISTANCE STATISTICS (10 features)
        'dtw_mean': np.mean(dtw_values),
        'dtw_std': np.std(dtw_values),
        'dtw_median': np.median(dtw_values),
        'dtw_min': np.min(dtw_values),
        'dtw_max': np.max(dtw_values),
        'dtw_q25': np.percentile(dtw_values, 25),
        'dtw_q75': np.percentile(dtw_values, 75),
        'dtw_iqr': np.percentile(dtw_values, 75) - np.percentile(dtw_values, 25),
        'dtw_skewness': stats.skew(dtw_values),
        'dtw_kurtosis': stats.kurtosis(dtw_values),
        
        # RATIO STATISTICS (5 features)
        'ratio_L_ell_mean': np.mean(L_values / (ell_values + 1)),
        'ratio_L_ell_std': np.std(L_values / (ell_values + 1)),
        'ratio_L_ell_median': np.median(L_values / (ell_values + 1)),
        'ratio_L_ell_max': np.max(L_values / (ell_values + 1)),
        'ratio_L_ell_min': np.min(L_values / (ell_values + 1)),
        
        # SEGMENT COUNTS (5 features)
        'num_segments': len(segments),
        'num_long_segments': np.sum(L_values > np.median(L_values)),
        'num_zero_breaks': np.sum(ell_values == 0),
        'pct_long_segments': 100 * np.sum(L_values > 50) / len(segments),
        'pct_zero_breaks': 100 * np.sum(ell_values == 0) / len(segments),
        
        # VARIABILITY (5 features)
        'cv_L': np.std(L_values) / (np.mean(L_values) + 1e-6),
        'cv_ell': np.std(ell_values) / (np.mean(ell_values) + 1e-6),
        'cv_dtw': np.std(dtw_values) / (np.mean(dtw_values) + 1e-6),
        'range_L': np.max(L_values) - np.min(L_values),
        'range_ell': np.max(ell_values) - np.min(ell_values),
    }
    
    return features

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run patient-level aggregated analysis"""
    
    bliss_dir = Path(r"C:\Users\treso\Desktop\SSNTEST\DATA\BLISS")
    
    print("\n" + "█"*70)
    print("█" + "  PATIENT-LEVEL AGGREGATED CLUSTERING".center(68) + "█")
    print("█"*70)
    
    print("\n" + "="*70)
    print("STEP 1: LOADING PATIENTS & CREATING REFERENCE")
    print("="*70)
    
    # Load all patients
    all_patient_data = []
    healthy_cycles_for_ref = []
    
    for i in range(12):
        subject_id = f"AB{2930 + i}"
        
        print(f"\n{subject_id}:")
        
        severity, severity_code = get_patient_severity(subject_id)
        
        result = load_all_trials(subject_id, bliss_dir)
        
        if result is None:
            print(f"  ❌ No valid data")
            continue
        
        cycles = result['cycles']
        
        print(f"  ✓ Loaded {len(cycles)} cycles ({severity})")
        
        all_patient_data.append({
            'subject_id': subject_id,
            'severity': severity,
            'severity_code': severity_code,
            'cycles': cycles
        })
        
        if severity == 'Healthy':
            healthy_cycles_for_ref.extend(cycles)
    
    print(f"\n✓ Loaded {len(all_patient_data)} patients")
    print(f"✓ Healthy cycles for reference: {len(healthy_cycles_for_ref)}")
    
    # Create reference
    reference_cycle = create_reference_cycle(healthy_cycles_for_ref, target_length=1000)
    
    # Extract segments and aggregate
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING SEGMENTS & AGGREGATING")
    print("="*70)
    
    patient_features = []
    
    for patient in all_patient_data:
        
        subject_id = patient['subject_id']
        severity = patient['severity']
        severity_code = patient['severity_code']
        cycles = patient['cycles']
        
        print(f"\n{subject_id} ({severity}):")
        print(f"  → Extracting segments vs reference...")
        
        segments = extract_segments_vs_reference(cycles, reference_cycle, max_cycles=20)
        
        print(f"  ✓ Extracted {len(segments)} segments")
        print(f"  → Computing aggregate features...")
        
        agg_features = aggregate_patient_features(segments, subject_id)
        
        if agg_features is None:
            print(f"  ❌ No features computed")
            continue
        
        # Add labels
        agg_features['severity'] = severity
        agg_features['severity_code'] = severity_code
        agg_features['binary_label'] = 0 if severity == 'Healthy' else 1
        
        patient_features.append(agg_features)
        
        print(f"  ✓ Features: L_mean={agg_features['L_mean']:.1f}, "
              f"ell_mean={agg_features['ell_mean']:.2f}, "
              f"dtw_mean={agg_features['dtw_mean']:.1f}")
    
    print("\n" + "="*70)
    print(f"TOTAL: {len(patient_features)} patients with aggregated features")
    print("="*70)
    
    # Convert to DataFrame
    df = pd.DataFrame(patient_features)
    
    print("\n" + "="*70)
    print("STEP 3: PATIENT-LEVEL DATASET")
    print("="*70)
    
    print(f"\nPatients: {len(df)}")
    print(f"\nDistribution:")
    print(df['severity'].value_counts())
    
    # Prepare features
    feature_cols = [col for col in df.columns 
                   if col not in ['subject_id', 'severity', 'severity_code', 'binary_label']]
    
    print(f"\nTotal features: {len(feature_cols)}")
    
    X = df[feature_cols].values
    y_true = df['binary_label'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: Healthy={np.sum(y_true==0)}, Pathological={np.sum(y_true==1)}")
    
    # Scaling
    print("\n" + "="*70)
    print("STEP 4: SCALING")
    print("="*70)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✓ Scaled with RobustScaler")
    
    # K-means
    print("\n" + "="*70)
    print("STEP 5: K-MEANS CLUSTERING (k=2)")
    print("="*70)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=50, max_iter=1000)
    y_pred = kmeans.fit_predict(X_scaled)
    
    # Align
    accuracy_direct = np.mean(y_pred == y_true)
    accuracy_flipped = np.mean(y_pred == (1 - y_true))
    
    if accuracy_flipped > accuracy_direct:
        y_pred = 1 - y_pred
        accuracy = accuracy_flipped
    else:
        accuracy = accuracy_direct
    
    if len(np.unique(y_pred)) > 1:
        silhouette = silhouette_score(X_scaled, y_pred)
    else:
        silhouette = 0.0
    
    print(f"\n✓ Patient-Level Accuracy: {100*accuracy:.1f}%")
    print(f"✓ Silhouette Score: {silhouette:.3f}")
    
    # Results table
    print(f"\n{'Subject':<10} {'Severity':<12} {'True':<6} {'Pred':<6} {'Result':<8}")
    print("-"*60)
    
    df['y_pred'] = y_pred
    
    for _, row in df.iterrows():
        result_mark = '✓' if row['binary_label'] == row['y_pred'] else '✗'
        print(f"{row['subject_id']:<10} {row['severity']:<12} "
              f"{row['binary_label']:<6} {row['y_pred']:<6} {result_mark:<8}")
    
    # Confusion matrix
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        cm = confusion_matrix(y_true, y_pred)
        
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Healthy  Pathological")
        print(f"Actual Healthy      {cm[0,0]:<8} {cm[0,1]:<8}")
        print(f"       Pathological {cm[1,0]:<8} {cm[1,1]:<8}")
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        
        print(classification_report(y_true, y_pred,
                                    target_names=['Healthy', 'Pathological'],
                                    digits=3,
                                    zero_division=0))
    
    # Leave-One-Out CV
    print("\n" + "="*70)
    print("STEP 6: LEAVE-ONE-OUT CROSS-VALIDATION")
    print("="*70)
    
    if len(df) >= 5:
        loo = LeaveOneOut()
        svm = SVC(kernel='rbf', random_state=42)
        
        try:
            loo_scores = cross_val_score(svm, X_scaled, y_true, cv=loo, scoring='accuracy')
            print(f"\nLOO-CV Accuracy: {100*loo_scores.mean():.1f}% ± {100*loo_scores.std():.1f}%")
            print(f"Correct: {int(loo_scores.sum())}/{len(loo_scores)}")
        except Exception as e:
            print(f"⚠️  LOO-CV failed: {e}")
    
    # Feature importance (PCA)
    print("\n" + "="*70)
    print("STEP 7: FEATURE IMPORTANCE (PCA)")
    print("="*70)
    
    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nTop {pca.n_components_} components explain {100*pca.explained_variance_ratio_.sum():.1f}% variance")
    
    for i in range(pca.n_components_):
        print(f"  PC{i+1}: {100*pca.explained_variance_ratio_[i]:.1f}%")
    
    # Top features for PC1
    pc1_loadings = pca.components_[0]
    top_features_idx = np.argsort(np.abs(pc1_loadings))[-5:][::-1]
    
    print(f"\nTop 5 features (PC1):")
    for idx in top_features_idx:
        print(f"  {feature_cols[idx]}: {pc1_loadings[idx]:.3f}")
    
    # Visualization
    print("\n" + "="*70)
    print("STEP 8: VISUALIZATION")
    print("="*70)
    
    visualize_patient_level(df, X_scaled, X_pca, y_true, y_pred, accuracy, 
                           silhouette, feature_cols, pca, 
                           cm if len(np.unique(y_pred)) > 1 else None)
    
    print("\n" + "█"*70)
    print("█" + "  ANALYSIS COMPLETE!".center(68) + "█")
    print("█"*70)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - PATIENT-LEVEL AGGREGATION")
    print("="*70)
    print(f"\n✓ Patients analyzed: {len(df)}")
    print(f"✓ Features per patient: {len(feature_cols)} aggregate statistics")
    print(f"✓ Method: Segments vs reference → aggregate → cluster")
    print(f"\nRESULTS:")
    print(f"  Accuracy:        {100*accuracy:.1f}%")
    print(f"  Silhouette:      {silhouette:.3f}")
    
    if accuracy >= 75:
        print("\n🎉🎉🎉 SUCCESS! Accuracy ≥75% - APPROACH VALIDATED! 🎉🎉🎉")
        print("   Patient-level aggregation WORKS!")
    elif accuracy >= 65:
        print("\n✓✓ GOOD! Accuracy ≥65% - Approach shows strong promise")
    elif accuracy >= 55:
        print("\n✓ MODERATE: Accuracy ≥55% - Better than segment-level")
    else:
        print("\n⚠️  Still challenging - but explored thoroughly")
    
    print("\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_patient_level(df, X_scaled, X_pca, y_true, y_pred, accuracy, 
                            silhouette, feature_cols, pca, cm):
    """Create patient-level visualization"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Panel A: PCA
    ax1 = plt.subplot(2, 3, 1)
    
    colors_true = ['#27AE60' if y == 0 else '#E74C3C' for y in y_true]
    markers_pred = ['o' if y == 0 else 's' for y in y_pred]
    
    for i, (color, marker) in enumerate(zip(colors_true, markers_pred)):
        ax1.scatter(X_pca[i, 0], X_pca[i, 1], c=color, marker=marker, 
                   s=300, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.text(X_pca[i, 0], X_pca[i, 1], df.iloc[i]['subject_id'][-4:], 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)', 
                   fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)', 
                   fontsize=12, fontweight='bold')
    ax1.set_title(f'(a) PCA ({len(df)} Patients)\nCircle=Pred Healthy, Square=Pred Patho',
                  fontsize=13, fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Feature comparison
    ax2 = plt.subplot(2, 3, 2)
    
    healthy_mask = df['binary_label'] == 0
    patho_mask = df['binary_label'] == 1
    
    # Plot L_mean vs dtw_mean
    ax2.scatter(df[healthy_mask]['L_mean'], df[healthy_mask]['dtw_mean'],
               c='#27AE60', s=200, alpha=0.8, edgecolor='black', linewidth=2,
               label='Healthy', marker='o')
    ax2.scatter(df[patho_mask]['L_mean'], df[patho_mask]['dtw_mean'],
               c='#E74C3C', s=200, alpha=0.8, edgecolor='black', linewidth=2,
               label='Pathological', marker='s')
    
    for _, row in df.iterrows():
        ax2.text(row['L_mean'], row['dtw_mean'], row['subject_id'][-4:],
                ha='center', va='center', fontsize=7, fontweight='bold')
    
    ax2.set_xlabel('Mean Segment Length (L)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean DTW Distance', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Feature Space', fontsize=13, fontweight='bold', loc='left')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    
    if cm is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax3,
                    xticklabels=['Healthy', 'Pathological'],
                    yticklabels=['Healthy', 'Pathological'],
                    annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                    cbar_kws={'label': 'Count'})
        
        ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax3.set_ylabel('True', fontsize=12, fontweight='bold')
        ax3.set_title(f'(c) Confusion Matrix\nAcc: {100*accuracy:.1f}%, Sil: {silhouette:.3f}',
                      fontsize=13, fontweight='bold', loc='left')
    else:
        ax3.text(0.5, 0.5, f'Accuracy: {100*accuracy:.1f}%\nSilhouette: {silhouette:.3f}',
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax3.set_title('(c) Results', fontsize=13, fontweight='bold', loc='left')
        ax3.axis('off')
    
    # Panel D: Feature distributions
    ax4 = plt.subplot(2, 3, 4)
    
    feature_to_plot = 'L_mean'
    
    ax4.hist(df[healthy_mask][feature_to_plot], bins=5, alpha=0.7,
            label='Healthy', color='#27AE60', edgecolor='black', linewidth=1.5)
    ax4.hist(df[patho_mask][feature_to_plot], bins=5, alpha=0.7,
            label='Pathological', color='#E74C3C', edgecolor='black', linewidth=1.5)
    
    ax4.set_xlabel(feature_to_plot.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax4.set_ylabel('Patient Count', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Feature Distribution', fontsize=13, fontweight='bold', loc='left')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel E: Top features
    ax5 = plt.subplot(2, 3, 5)
    
    pc1_loadings = pca.components_[0]
    top_features_idx = np.argsort(np.abs(pc1_loadings))[-10:]
    
    y_pos = np.arange(len(top_features_idx))
    colors_bar = ['#E74C3C' if pc1_loadings[i] < 0 else '#27AE60' 
                  for i in top_features_idx]
    
    ax5.barh(y_pos, pc1_loadings[top_features_idx], color=colors_bar,
            alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([feature_cols[i].replace('_', ' ') for i in top_features_idx],
                        fontsize=9)
    ax5.set_xlabel('PC1 Loading', fontsize=12, fontweight='bold')
    ax5.set_title('(e) Top 10 Features (PC1)', fontsize=13, fontweight='bold', loc='left')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Panel F: Variance explained
    ax6 = plt.subplot(2, 3, 6)
    
    n_show = min(10, len(pca.explained_variance_ratio_))
    variance = pca.explained_variance_ratio_[:n_show]
    
    bars = ax6.bar(range(n_show), variance, color='#3498DB', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    ax6.set_xticks(range(n_show))
    ax6.set_xticklabels([f'PC{i+1}' for i in range(n_show)])
    ax6.set_ylabel('Explained Variance', fontsize=12, fontweight='bold')
    ax6.set_title('(f) PCA Components', fontsize=13, fontweight='bold', loc='left')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, variance)):
        ax6.text(i, val, f'{100*val:.1f}%', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_dir = Path("results_patient_aggregated")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "patient_aggregated_clustering.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    main()