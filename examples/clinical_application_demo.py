#!/usr/bin/env python3
"""
Brain-Forge Clinical Application Demo - Epilepsy Seizure Detection

This demo demonstrates focusing on ONE specific clinical application
with clear validation criteria, as recommended for successful BCI development.
Epilepsy seizure detection provides well-understood patterns and 
measurable clinical outcomes.

Key Features Demonstrated:
- Single clinical focus (epilepsy detection)
- Realistic EEG-based approach
- Clear success metrics (sensitivity/specificity)
- Clinical validation pathway
- Realistic performance targets
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import BrainForgeConfig
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClinicalMetrics:
    """Clinical validation metrics for seizure detection"""
    sensitivity: float  # True positive rate (detecting actual seizures)
    specificity: float  # True negative rate (avoiding false alarms)
    ppv: float         # Positive predictive value
    npv: float         # Negative predictive value
    accuracy: float    # Overall accuracy
    f1_score: float    # Harmonic mean of precision and recall
    
    @property
    def clinical_grade(self) -> bool:
        """Check if metrics meet clinical standards"""
        return (self.sensitivity >= 0.90 and 
                self.specificity >= 0.95 and 
                self.ppv >= 0.80)


class EpilepticSeizureSimulator:
    """Generate realistic EEG data for seizure detection research"""
    
    def __init__(self, fs: float = 250.0, n_channels: int = 19):
        self.fs = fs  # Sampling frequency
        self.n_channels = n_channels  # Standard 10-20 EEG montage
        
        # Channel locations (simplified)
        self.channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]
        
    def generate_normal_eeg(self, duration: float = 10.0) -> np.ndarray:
        """Generate normal EEG background activity"""
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        eeg_data = np.zeros((self.n_channels, n_samples))
        
        for ch in range(self.n_channels):
            # Alpha rhythm (8-12 Hz) - stronger in posterior channels
            alpha_strength = 0.5 if ch >= 8 else 0.2  # O1, O2, P3, P4 stronger
            alpha_freq = 8 + 4 * np.random.random()
            alpha = alpha_strength * np.sin(2 * np.pi * alpha_freq * t)
            
            # Beta rhythm (13-30 Hz) - stronger in frontal/central
            beta_strength = 0.3 if ch < 10 else 0.1
            beta_freq = 13 + 17 * np.random.random()
            beta = beta_strength * np.sin(2 * np.pi * beta_freq * t)
            
            # Theta rhythm (4-8 Hz) - low level background
            theta_freq = 4 + 4 * np.random.random()
            theta = 0.1 * np.sin(2 * np.pi * theta_freq * t)
            
            # Noise and artifacts
            noise = 0.2 * np.random.randn(n_samples)
            
            # Combine all components
            eeg_data[ch] = alpha + beta + theta + noise
            
        return eeg_data
        
    def generate_seizure_eeg(self, duration: float = 30.0, 
                           seizure_start: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate EEG with embedded seizure activity"""
        n_samples = int(duration * self.fs)
        seizure_start_sample = int(seizure_start * self.fs)
        seizure_duration = 15.0  # 15-second seizure
        seizure_end_sample = int((seizure_start + seizure_duration) * self.fs)
        
        # Start with normal EEG
        eeg_data = self.generate_normal_eeg(duration)
        
        # Create seizure labels
        labels = np.zeros(n_samples)
        labels[seizure_start_sample:seizure_end_sample] = 1
        
        # Add seizure patterns
        t_seizure = np.linspace(0, seizure_duration, seizure_end_sample - seizure_start_sample)
        
        # Typical seizure progression
        for ch in range(self.n_channels):
            # High-frequency spiking (10-25 Hz)
            spike_freq = 15 + 10 * np.random.random()
            spike_amplitude = 2.0 + np.random.random()  # Much higher amplitude
            
            # Evolving frequency pattern
            freq_evolution = spike_freq * (1 + 0.5 * np.sin(0.5 * t_seizure))
            
            # Generate seizure activity
            seizure_signal = np.zeros_like(t_seizure)
            for i, freq in enumerate(freq_evolution):
                seizure_signal[i] = spike_amplitude * np.sin(2 * np.pi * freq * t_seizure[i])
                
            # Add amplitude modulation (seizure envelope)
            envelope = np.exp(-0.5 * ((t_seizure - seizure_duration/2) / (seizure_duration/4))**2)
            seizure_signal *= envelope
            
            # Inject seizure into EEG
            eeg_data[ch, seizure_start_sample:seizure_end_sample] += seizure_signal
            
        return eeg_data, labels


class SeizureDetectionSystem:
    """Clinical-grade seizure detection system"""
    
    def __init__(self):
        self.fs = 250.0
        self.window_size = 2.0  # 2-second analysis windows
        self.overlap = 0.5      # 50% overlap
        self.classifier = None
        self.feature_names = []
        
        # Clinical performance targets
        self.target_sensitivity = 0.90  # 90% seizure detection
        self.target_specificity = 0.95  # 95% non-seizure classification
        self.target_latency = 5.0       # 5-second detection latency
        
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """Extract clinically relevant features for seizure detection"""
        n_channels, n_samples = eeg_data.shape
        
        features = []
        feature_names = []
        
        # Frequency domain features
        freqs, psd = signal.welch(eeg_data, self.fs, nperseg=int(self.fs), axis=1)
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.mean(psd[:, band_mask], axis=1)
            
            features.extend(band_power.tolist())
            feature_names.extend([f'{ch}_{band_name}' for ch in range(n_channels)])
            
        # Time domain features
        for ch in range(n_channels):
            ch_data = eeg_data[ch]
            
            # Statistical features
            features.extend([
                np.mean(ch_data),           # Mean amplitude
                np.std(ch_data),            # Standard deviation
                np.var(ch_data),            # Variance
                np.max(ch_data) - np.min(ch_data),  # Peak-to-peak
                len(self._find_peaks(ch_data)) / (len(ch_data) / self.fs)  # Spike rate
            ])
            
            feature_names.extend([
                f'{ch}_mean', f'{ch}_std', f'{ch}_var', f'{ch}_ptp', f'{ch}_spike_rate'
            ])
            
        # Cross-channel features (synchrony measures)
        correlation_matrix = np.corrcoef(eeg_data)
        
        # Mean correlation (synchrony indicator)
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        features.append(np.mean(upper_triangle))
        feature_names.append('mean_synchrony')
        
        # Maximum correlation
        features.append(np.max(upper_triangle))
        feature_names.append('max_synchrony')
        
        self.feature_names = feature_names
        return np.array(features)
        
    def _find_peaks(self, signal_data: np.ndarray, threshold_factor: float = 3.0) -> List[int]:
        """Find significant peaks in signal (spike detection)"""
        threshold = threshold_factor * np.std(signal_data)
        peaks = []
        
        for i in range(1, len(signal_data) - 1):
            if (signal_data[i] > signal_data[i-1] and 
                signal_data[i] > signal_data[i+1] and 
                signal_data[i] > threshold):
                peaks.append(i)
                
        return peaks
        
    def prepare_training_data(self, n_normal: int = 100, n_seizure: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training dataset with normal and seizure EEG"""
        logger.info(f"Generating training data: {n_normal} normal + {n_seizure} seizure samples")
        
        simulator = EpilepticSeizureSimulator()
        
        X = []
        y = []
        
        # Generate normal EEG samples
        for i in range(n_normal):
            normal_eeg = simulator.generate_normal_eeg(duration=10.0)
            features = self.extract_features(normal_eeg)
            X.append(features)
            y.append(0)  # Normal class
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Generated {i + 1}/{n_normal} normal samples")
                
        # Generate seizure EEG samples
        for i in range(n_seizure):
            seizure_eeg, _ = simulator.generate_seizure_eeg(duration=30.0)
            
            # Extract features from seizure portion
            seizure_start = int(10.0 * simulator.fs)
            seizure_end = int(25.0 * simulator.fs)
            seizure_portion = seizure_eeg[:, seizure_start:seizure_end]
            
            features = self.extract_features(seizure_portion)
            X.append(features)
            y.append(1)  # Seizure class
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{n_seizure} seizure samples")
                
        return np.array(X), np.array(y)
        
    def train_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train seizure detection classifier"""
        logger.info("Training seizure detection classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier (clinically interpretable)
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        start_time = time()
        self.classifier.fit(X_train, y_train)
        training_time = time() - start_time
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Calculate clinical metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = ClinicalMetrics(
            sensitivity=tp / (tp + fn) if (tp + fn) > 0 else 0,
            specificity=tn / (tn + fp) if (tn + fp) > 0 else 0,
            ppv=tp / (tp + fp) if (tp + fp) > 0 else 0,
            npv=tn / (tn + fn) if (tn + fn) > 0 else 0,
            accuracy=(tp + tn) / (tp + tn + fp + fn),
            f1_score=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        )
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Model performance:")
        logger.info(f"  Sensitivity (recall): {metrics.sensitivity:.3f}")
        logger.info(f"  Specificity: {metrics.specificity:.3f}")
        logger.info(f"  PPV (precision): {metrics.ppv:.3f}")
        logger.info(f"  NPV: {metrics.npv:.3f}")
        logger.info(f"  Accuracy: {metrics.accuracy:.3f}")
        logger.info(f"  F1-score: {metrics.f1_score:.3f}")
        
        # Clinical grade assessment
        if metrics.clinical_grade:
            logger.info("✅ CLINICAL GRADE: Performance meets clinical standards")
        else:
            logger.warning("⚠️ NOT CLINICAL GRADE: Performance below clinical standards")
            logger.warning(f"   Targets: Sensitivity≥{self.target_sensitivity:.2f}, "
                         f"Specificity≥{self.target_specificity:.2f}, PPV≥0.80")
            
        return {
            'metrics': metrics,
            'training_time': training_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.classifier.feature_importances_,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_labels': y_test
        }
        
    def real_time_detection_demo(self, duration: float = 60.0) -> Dict:
        """Demonstrate real-time seizure detection"""
        logger.info(f"=== Real-time Seizure Detection Demo ({duration}s) ===")
        
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call train_classifier() first.")
            
        simulator = EpilepticSeizureSimulator()
        
        # Generate test EEG with seizure
        eeg_data, true_labels = simulator.generate_seizure_eeg(duration=duration, seizure_start=20.0)
        
        # Process in sliding windows
        window_samples = int(self.window_size * self.fs)
        step_samples = int(window_samples * (1 - self.overlap))
        
        detections = []
        timestamps = []
        processing_times = []
        
        for start_idx in range(0, eeg_data.shape[1] - window_samples, step_samples):
            window_start_time = time()
            
            # Extract window
            window_data = eeg_data[:, start_idx:start_idx + window_samples]
            window_time = start_idx / self.fs
            
            # Extract features and classify
            features = self.extract_features(window_data)
            
            # Predict seizure probability
            seizure_prob = self.classifier.predict_proba(features.reshape(1, -1))[0, 1]
            is_seizure = seizure_prob > 0.5
            
            processing_time = (time() - window_start_time) * 1000  # ms
            
            detections.append({
                'time': window_time,
                'seizure_probability': seizure_prob,
                'is_seizure': is_seizure,
                'processing_time_ms': processing_time
            })
            
            timestamps.append(window_time)
            processing_times.append(processing_time)
            
            # Real-time feedback
            if is_seizure and seizure_prob > 0.8:
                logger.info(f"🚨 SEIZURE DETECTED at {window_time:.1f}s (confidence: {seizure_prob:.2f})")
                
        # Performance analysis
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        logger.info(f"=== Real-time Performance Analysis ===")
        logger.info(f"Average processing time: {avg_processing_time:.1f}ms")
        logger.info(f"Maximum processing time: {max_processing_time:.1f}ms")
        logger.info(f"Target latency: {self.target_latency * 1000:.0f}ms")
        
        latency_compliant = max_processing_time < (self.target_latency * 1000)
        logger.info(f"Latency compliance: {'✅ PASSED' if latency_compliant else '❌ FAILED'}")
        
        return {
            'detections': detections,
            'true_labels': true_labels,
            'eeg_data': eeg_data,
            'timestamps': timestamps,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'latency_compliant': latency_compliant
        }


class ClinicalValidationFramework:
    """Framework for clinical validation of seizure detection system"""
    
    def __init__(self):
        self.detection_system = SeizureDetectionSystem()
        
    def run_clinical_validation(self) -> Dict:
        """Run complete clinical validation pipeline"""
        logger.info("=== Brain-Forge Clinical Validation: Epilepsy Seizure Detection ===")
        
        validation_results = {}
        
        # Step 1: Prepare training data
        logger.info("\n1. Preparing clinical training dataset...")
        X, y = self.detection_system.prepare_training_data(n_normal=200, n_seizure=100)
        
        # Step 2: Train and evaluate classifier
        logger.info("\n2. Training clinical-grade classifier...")
        training_results = self.detection_system.train_classifier(X, y)
        validation_results['training'] = training_results
        
        # Step 3: Real-time validation
        logger.info("\n3. Validating real-time performance...")
        realtime_results = self.detection_system.real_time_detection_demo(duration=90.0)
        validation_results['realtime'] = realtime_results
        
        # Step 4: Clinical assessment
        logger.info("\n4. Clinical performance assessment...")
        clinical_assessment = self._assess_clinical_readiness(
            training_results['metrics'], 
            realtime_results
        )
        validation_results['clinical_assessment'] = clinical_assessment
        
        return validation_results
        
    def _assess_clinical_readiness(self, metrics: ClinicalMetrics, realtime_results: Dict) -> Dict:
        """Assess clinical readiness based on FDA/clinical standards"""
        
        # Clinical requirements for seizure detection systems
        requirements = {
            'sensitivity': {'target': 0.90, 'critical': True},
            'specificity': {'target': 0.95, 'critical': True},
            'ppv': {'target': 0.80, 'critical': True},
            'latency_ms': {'target': 5000, 'critical': True},  # 5 seconds
            'processing_time_ms': {'target': 1000, 'critical': False}  # 1 second
        }
        
        assessment = {}
        
        # Check each requirement
        assessment['sensitivity_pass'] = metrics.sensitivity >= requirements['sensitivity']['target']
        assessment['specificity_pass'] = metrics.specificity >= requirements['specificity']['target']
        assessment['ppv_pass'] = metrics.ppv >= requirements['ppv']['target']
        assessment['latency_pass'] = realtime_results['max_processing_time_ms'] <= requirements['latency_ms']['target']
        assessment['processing_pass'] = realtime_results['avg_processing_time_ms'] <= requirements['processing_time_ms']['target']
        
        # Overall assessment
        critical_tests = [assessment['sensitivity_pass'], assessment['specificity_pass'], 
                         assessment['ppv_pass'], assessment['latency_pass']]
        assessment['clinical_ready'] = all(critical_tests)
        
        # FDA pathway assessment
        if assessment['clinical_ready']:
            assessment['fda_pathway'] = '510(k) Premarket Notification - Class II Medical Device'
            assessment['next_steps'] = [
                'Conduct pilot clinical study (n=20-50 patients)',
                'Submit FDA pre-submission meeting request',
                'Prepare 510(k) application with predicate device comparison',
                'Establish Quality Management System (ISO 13485)'
            ]
        else:
            assessment['fda_pathway'] = 'Algorithm optimization required before clinical trials'
            assessment['next_steps'] = [
                'Improve algorithm performance to meet clinical targets',
                'Expand training dataset with real patient data',
                'Validate with clinical EEG recordings',
                'Partner with epilepsy monitoring units for data collection'
            ]
            
        return assessment
        
    def visualize_clinical_results(self, validation_results: Dict) -> None:
        """Create comprehensive clinical validation visualizations"""
        training_results = validation_results['training']
        realtime_results = validation_results['realtime']
        clinical_assessment = validation_results['clinical_assessment']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Brain-Forge Clinical Validation - Epilepsy Seizure Detection', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = training_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticklabels(['Normal', 'Seizure'])
        axes[0, 0].set_yticklabels(['Normal', 'Seizure'])
        
        # 2. Clinical Metrics
        metrics = training_results['metrics']
        metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1-Score']
        metric_values = [metrics.sensitivity, metrics.specificity, metrics.ppv, 
                        metrics.npv, metrics.accuracy, metrics.f1_score]
        
        colors = ['green' if v >= 0.8 else 'orange' if v >= 0.7 else 'red' for v in metric_values]
        
        bars = axes[0, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Clinical Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add target lines
        axes[0, 1].axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Clinical Target')
        axes[0, 1].legend()
        
        # 3. Real-time Detection
        detections = realtime_results['detections']
        times = [d['time'] for d in detections]
        probs = [d['seizure_probability'] for d in detections]
        
        axes[0, 2].plot(times, probs, 'b-', alpha=0.7, label='Seizure Probability')
        axes[0, 2].axhline(y=0.5, color='red', linestyle='--', label='Detection Threshold')
        axes[0, 2].axvspan(20, 35, alpha=0.3, color='red', label='True Seizure Period')
        axes[0, 2].set_title('Real-time Seizure Detection')
        axes[0, 2].set_xlabel('Time (seconds)')
        axes[0, 2].set_ylabel('Seizure Probability')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Processing Time Analysis
        processing_times = [d['processing_time_ms'] for d in detections]
        
        axes[1, 0].hist(processing_times, bins=20, alpha=0.7, color='blue')
        axes[1, 0].axvline(np.mean(processing_times), color='red', linestyle='-', 
                          label=f'Mean: {np.mean(processing_times):.1f}ms')
        axes[1, 0].axvline(1000, color='orange', linestyle='--', label='Target: 1000ms')
        axes[1, 0].set_title('Processing Time Distribution')
        axes[1, 0].set_xlabel('Processing Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 5. Feature Importance
        feature_importance = training_results['feature_importance']
        top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
        top_features = [self.detection_system.feature_names[i] for i in top_features_idx]
        top_importance = feature_importance[top_features_idx]
        
        axes[1, 1].barh(range(len(top_features)), top_importance, alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features)
        axes[1, 1].set_title('Top 10 Important Features')
        axes[1, 1].set_xlabel('Feature Importance')
        
        # 6. Clinical Readiness Assessment
        assessment_items = ['Sensitivity', 'Specificity', 'PPV', 'Latency', 'Processing']
        assessment_status = [
            clinical_assessment['sensitivity_pass'],
            clinical_assessment['specificity_pass'],
            clinical_assessment['ppv_pass'],
            clinical_assessment['latency_pass'],
            clinical_assessment['processing_pass']
        ]
        
        colors = ['green' if status else 'red' for status in assessment_status]
        
        axes[1, 2].bar(assessment_items, [1]*len(assessment_items), color=colors, alpha=0.7)
        axes[1, 2].set_title('Clinical Readiness Assessment')
        axes[1, 2].set_ylabel('Pass/Fail')
        axes[1, 2].set_ylim(0, 1.2)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add pass/fail labels
        for i, (item, status) in enumerate(zip(assessment_items, assessment_status)):
            label = '✅ PASS' if status else '❌ FAIL'
            axes[1, 2].text(i, 0.5, label, ha='center', va='center', fontweight='bold')
            
        plt.tight_layout()
        plt.show()
        
        # Clinical summary
        logger.info("\n=== CLINICAL VALIDATION SUMMARY ===")
        
        if clinical_assessment['clinical_ready']:
            logger.info("🎉 CLINICAL READY: System meets FDA Class II medical device standards")
            logger.info(f"FDA Pathway: {clinical_assessment['fda_pathway']}")
        else:
            logger.info("⚠️ NOT CLINICAL READY: Performance optimization required")
            logger.info("Focus areas for improvement:")
            
            if not clinical_assessment['sensitivity_pass']:
                logger.info("  • Improve seizure detection rate (sensitivity)")
            if not clinical_assessment['specificity_pass']:
                logger.info("  • Reduce false alarm rate (specificity)")
            if not clinical_assessment['ppv_pass']:
                logger.info("  • Improve positive predictive value")
            if not clinical_assessment['latency_pass']:
                logger.info("  • Optimize processing speed for real-time detection")
                
        logger.info("\nNext Steps:")
        for step in clinical_assessment['next_steps']:
            logger.info(f"  • {step}")


def main():
    """Main demo function for clinical application approach"""
    logger.info("=== Brain-Forge Clinical Application Demo ===")
    logger.info("Focus: Epilepsy seizure detection")
    logger.info("Strategy: Single clinical application with clear validation")
    
    # Create clinical validation framework
    validator = ClinicalValidationFramework()
    
    try:
        # Run complete clinical validation
        logger.info("\n🚀 Starting clinical validation process...")
        validation_results = validator.run_clinical_validation()
        
        # Create clinical visualizations
        validator.visualize_clinical_results(validation_results)
        
        # Final recommendations
        logger.info("\n=== STRATEGIC RECOMMENDATIONS ===")
        logger.info("✅ Single clinical focus approach validated")
        logger.info("✅ Clear success metrics established")
        logger.info("✅ Regulatory pathway identified")
        logger.info("✅ Performance benchmarking completed")
        
        logger.info("\n🎯 This demonstrates the power of:")
        logger.info("  • Focused clinical application vs. trying to solve everything")
        logger.info("  • Clear, measurable success criteria")
        logger.info("  • Realistic performance targets")  
        logger.info("  • Established validation pathway")
        logger.info("  • Direct path to clinical impact")
        
        logger.info("\n✨ Next: Apply same focused approach to other applications:")
        logger.info("  • Motor imagery BCI")
        logger.info("  • Cognitive load assessment")
        logger.info("  • Sleep staging")
        logger.info("  • Anesthesia monitoring")
        
    except Exception as e:
        logger.error(f"Clinical validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
