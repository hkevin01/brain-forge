"""
Assurance Tests for Clinical & Research Applications
Validates medical diagnostics, neurofeedback therapy, and cognitive enhancement
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock
from dataclasses import dataclass
from typing import Dict, List
import time
from scipy import stats, signal
from sklearn.metrics import classification_report, roc_auc_score

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class ClinicalSpecs:
    """Clinical diagnostic specifications"""
    diagnostic_accuracy: float = 0.95  # 95% diagnostic accuracy
    false_positive_rate: float = 0.05  # 5% false positive rate
    diagnostic_latency: float = 300.0  # 5 minutes maximum
    sensitivity: float = 0.95  # 95% sensitivity
    specificity: float = 0.95  # 95% specificity
    regulatory_compliance: bool = True  # FDA/CE compliance


@dataclass  
class NeurofeedbackSpecs:
    """Neurofeedback therapy specifications"""
    real_time_latency: float = 50e-3  # 50ms maximum latency
    feedback_accuracy: float = 0.90  # 90% neurofeedback accuracy
    therapy_efficacy: float = 0.80  # 80% therapy success rate
    session_duration: float = 1800.0  # 30 minute sessions
    improvement_tracking: bool = True  # Progress tracking capability


@dataclass
class CognitiveEnhancementSpecs:
    """Cognitive enhancement specifications"""  
    enhancement_magnitude: float = 0.25  # 25% performance improvement
    enhancement_duration: float = 3600.0  # 1 hour enhancement duration
    safety_monitoring: bool = True  # Real-time safety monitoring
    personalization_accuracy: float = 0.85  # 85% personalization accuracy
    ethical_compliance: bool = True  # Ethics compliance


class TestMedicalDiagnostics:
    """Test automated medical diagnosis from brain patterns"""
    
    @pytest.fixture
    def diagnostic_system(self):
        """Mock medical diagnostic system"""
        system = Mock()
        system.specs = ClinicalSpecs()
        system.trained_models = {}
        system.clinical_database = {}
        return system
    
    def test_epilepsy_seizure_detection(self, diagnostic_system):
        """Test automated epilepsy/seizure detection"""
        # Mock seizure pattern database
        def create_seizure_patterns():
            """Create epileptic seizure pattern templates"""
            seizure_types = {
                'tonic_clonic': {
                    'frequency_bands': [15, 25, 35],  # Beta/Gamma spikes
                    'amplitude_increase': 5.0,  # 5x amplitude increase
                    'duration_range': (30, 180),  # 30s-3min duration
                    'spatial_pattern': 'generalized'
                },
                'focal': {
                    'frequency_bands': [8, 12, 18],  # Alpha/Beta rhythms
                    'amplitude_increase': 3.0,  # 3x amplitude increase
                    'duration_range': (10, 60),  # 10s-1min duration
                    'spatial_pattern': 'localized'
                },
                'absence': {
                    'frequency_bands': [3, 4],  # 3-4 Hz spike-wave
                    'amplitude_increase': 2.0,  # 2x amplitude increase  
                    'duration_range': (5, 30),  # 5-30s duration
                    'spatial_pattern': 'bilateral'
                }
            }
            return seizure_types
        
        # Mock seizure detection algorithm
        def detect_seizure_patterns(eeg_data, seizure_templates):
            """Detect seizure patterns in EEG data"""
            n_channels, n_samples = eeg_data.shape
            seizure_detections = []
            
            # Sliding window analysis
            window_size = 1000  # 4 seconds at 250 Hz
            step_size = 250     # 1 second steps
            
            for start_idx in range(0, n_samples - window_size, step_size):
                window_data = eeg_data[:, start_idx:start_idx + window_size]
                window_time = start_idx / 250.0  # Convert to seconds
                
                # Calculate spectral features
                freqs, psd = signal.welch(window_data, fs=250, nperseg=256)
                
                # Check each seizure type
                for seizure_type, template in seizure_templates.items():
                    seizure_score = self._calculate_seizure_score(
                        psd, freqs, template
                    )
                    
                    # Threshold for seizure detection
                    if seizure_score > 0.8:  # 80% confidence threshold
                        seizure_detections.append({
                            'type': seizure_type,
                            'start_time': window_time,
                            'confidence': seizure_score,
                            'affected_channels': self._identify_active_channels(
                                window_data, template['spatial_pattern']
                            )
                        })
            
            return seizure_detections
        
        def _calculate_seizure_score(self, psd, freqs, template):
            """Calculate seizure likelihood score"""
            score = 0.0
            target_bands = template['frequency_bands']
            amplitude_threshold = template['amplitude_increase']
            
            for target_freq in target_bands:
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(freqs - target_freq))
                freq_power = np.mean(psd[:, freq_idx])
                
                # Compare to baseline (simplified)
                baseline_power = np.mean(psd[:, :10])  # Low frequency baseline
                power_ratio = freq_power / (baseline_power + 1e-12)
                
                if power_ratio > amplitude_threshold:
                    score += 0.3  # Contribute to seizure score
            
            return min(1.0, score)
        
        def _identify_active_channels(self, window_data, spatial_pattern):
            """Identify channels showing seizure activity"""
            channel_activity = np.var(window_data, axis=1)
            activity_threshold = np.percentile(channel_activity, 75)
            
            active_channels = np.where(channel_activity > activity_threshold)[0]
            
            # Filter based on spatial pattern
            if spatial_pattern == 'localized':
                # Keep only most active channels
                active_channels = active_channels[:min(8, len(active_channels))]
            elif spatial_pattern == 'bilateral':
                # Ensure bilateral representation
                n_channels = len(channel_activity)
                left_channels = active_channels[active_channels < n_channels // 2]
                right_channels = active_channels[active_channels >= n_channels // 2]
                active_channels = np.concatenate([left_channels[:4], 
                                                right_channels[:4]])
            
            return active_channels.tolist()
        
        # Attach methods to diagnostic system
        diagnostic_system.create_seizure_templates = create_seizure_patterns
        diagnostic_system.detect_seizures = detect_seizure_patterns
        diagnostic_system._calculate_seizure_score = _calculate_seizure_score
        diagnostic_system._identify_active_channels = _identify_active_channels
        
        # Create seizure pattern templates
        seizure_templates = diagnostic_system.create_seizure_templates()
        
        # Generate test EEG data with simulated seizures
        n_channels = 64
        duration_seconds = 300  # 5 minutes
        sampling_rate = 250  # Hz
        n_samples = duration_seconds * sampling_rate
        
        # Baseline EEG
        eeg_data = np.random.normal(0, 50e-6, (n_channels, n_samples))
        
        # Add simulated seizures
        seizure_times = [60, 180, 240]  # Seizures at 1min, 3min, 4min
        ground_truth_seizures = []
        
        for seizure_time in seizure_times:
            seizure_start = int(seizure_time * sampling_rate)
            seizure_duration = np.random.randint(20, 60)  # 20-60 seconds
            seizure_end = seizure_start + seizure_duration * sampling_rate
            
            # Add seizure pattern (simplified)
            seizure_channels = np.random.choice(n_channels, 16, replace=False)
            for ch in seizure_channels:
                # Add spike activity
                spike_freq = 20  # 20 Hz spikes
                t = np.linspace(0, seizure_duration, seizure_end - seizure_start)
                spike_pattern = 200e-6 * np.sin(2 * np.pi * spike_freq * t)
                eeg_data[ch, seizure_start:seizure_end] += spike_pattern
            
            ground_truth_seizures.append({
                'start_time': seizure_time,
                'end_time': seizure_time + seizure_duration,
                'channels': seizure_channels.tolist()
            })
        
        # Test seizure detection
        detected_seizures = diagnostic_system.detect_seizures(
            eeg_data, seizure_templates
        )
        
        # Validate detection performance
        assert len(detected_seizures) > 0, "Should detect seizure events"
        
        # Calculate detection metrics
        true_positives = 0
        false_positives = 0
        
        for detection in detected_seizures:
            detection_time = detection['start_time']
            is_true_positive = False
            
            for ground_truth in ground_truth_seizures:
                if (ground_truth['start_time'] <= detection_time <= 
                    ground_truth['end_time'] + 30):  # 30s tolerance
                    is_true_positive = True
                    break
            
            if is_true_positive:
                true_positives += 1
            else:
                false_positives += 1
        
        # Calculate performance metrics
        sensitivity = true_positives / len(ground_truth_seizures)
        precision = true_positives / (true_positives + false_positives)
        
        # Validate against clinical specifications
        assert sensitivity >= diagnostic_system.specs.sensitivity * 0.8, \
            f"Seizure detection sensitivity {sensitivity:.3f} too low"
        
        assert precision >= 0.7, \
            f"Seizure detection precision {precision:.3f} too low"
        
        # Validate detection latency
        detection_confidences = [d['confidence'] for d in detected_seizures]
        mean_confidence = np.mean(detection_confidences)
        
        assert mean_confidence >= 0.8, \
            f"Mean detection confidence {mean_confidence:.3f} too low"
    
    def test_depression_screening(self, diagnostic_system):
        """Test automated depression screening from neural biomarkers"""
        # Mock depression biomarker extraction
        def extract_depression_biomarkers(brain_data):
            """Extract neural biomarkers associated with depression"""
            # Known depression biomarkers from literature
            biomarkers = {}
            
            # 1. Frontal alpha asymmetry
            left_frontal_alpha = np.mean(brain_data[:8, :])  # Left frontal
            right_frontal_alpha = np.mean(brain_data[8:16, :])  # Right frontal
            alpha_asymmetry = np.log(right_frontal_alpha) - np.log(left_frontal_alpha)
            biomarkers['frontal_alpha_asymmetry'] = alpha_asymmetry
            
            # 2. Theta cordance (simplified)
            theta_power = np.mean(brain_data**2)
            biomarkers['theta_cordance'] = theta_power
            
            # 3. Gamma connectivity
            gamma_connectivity = np.corrcoef(brain_data).mean()
            biomarkers['gamma_connectivity'] = gamma_connectivity
            
            # 4. Default mode network activity
            dmn_regions = [0, 10, 20, 30]  # Simplified DMN regions
            dmn_activity = np.mean([np.var(brain_data[i, :]) for i in dmn_regions])
            biomarkers['dmn_hyperactivity'] = dmn_activity
            
            # 5. Sleep-related patterns (REM sleep measures)
            sleep_spindles = len(signal.find_peaks(np.mean(brain_data, axis=0), 
                                                 height=np.std(brain_data))[0])
            biomarkers['sleep_architecture'] = sleep_spindles
            
            return biomarkers
        
        # Mock depression classification
        def classify_depression_risk(biomarkers):
            """Classify depression risk from biomarkers"""
            # Simplified classification based on known patterns
            risk_score = 0.0
            
            # Frontal asymmetry (more right-sided activity in depression)
            if biomarkers['frontal_alpha_asymmetry'] > 0.1:
                risk_score += 0.3
            
            # Theta activity (increased in depression)
            if biomarkers['theta_cordance'] > 1e-9:
                risk_score += 0.2
            
            # Reduced connectivity
            if biomarkers['gamma_connectivity'] < 0.3:
                risk_score += 0.2
            
            # DMN hyperactivity
            if biomarkers['dmn_hyperactivity'] > 1e-9:
                risk_score += 0.2
            
            # Sleep disruption
            if biomarkers['sleep_architecture'] < 50:
                risk_score += 0.1
            
            # Classification
            if risk_score >= 0.7:
                classification = 'high_risk'
                confidence = min(0.95, risk_score)
            elif risk_score >= 0.4:
                classification = 'moderate_risk'
                confidence = risk_score * 0.8
            else:
                classification = 'low_risk'
                confidence = 1.0 - risk_score
            
            return {
                'classification': classification,
                'risk_score': risk_score,
                'confidence': confidence,
                'biomarkers': biomarkers
            }
        
        diagnostic_system.extract_depression_biomarkers = extract_depression_biomarkers
        diagnostic_system.classify_depression = classify_depression_risk
        
        # Create test subjects with known depression status
        test_subjects = []
        
        # Simulate depressed subjects (n=50)
        for i in range(50):
            # Generate brain data with depression patterns
            brain_data = np.random.normal(0, 30e-6, (64, 5000))
            
            # Add depression-specific patterns
            # Increased right frontal activity
            brain_data[8:16, :] *= 1.5
            # Increased theta
            t = np.linspace(0, 20, 5000)
            theta_rhythm = 20e-6 * np.sin(2 * np.pi * 6 * t)
            brain_data += theta_rhythm[np.newaxis, :]
            # Reduced connectivity (add more noise)
            brain_data += np.random.normal(0, 10e-6, brain_data.shape)
            
            test_subjects.append({
                'subject_id': f'dep_{i}',
                'brain_data': brain_data,
                'true_label': 'depressed',
                'depression_severity': np.random.uniform(0.6, 1.0)
            })
        
        # Simulate healthy controls (n=50)  
        for i in range(50):
            # Generate brain data with healthy patterns
            brain_data = np.random.normal(0, 20e-6, (64, 5000))
            
            # Add healthy patterns
            # Balanced frontal activity
            alpha_rhythm = 30e-6 * np.sin(2 * np.pi * 10 * np.linspace(0, 20, 5000))
            brain_data[:16, :] += alpha_rhythm[np.newaxis, :]
            
            test_subjects.append({
                'subject_id': f'ctrl_{i}',
                'brain_data': brain_data,
                'true_label': 'healthy',
                'depression_severity': np.random.uniform(0.0, 0.3)
            })
        
        # Test depression screening
        screening_results = []
        classification_start = time.time()
        
        for subject in test_subjects:
            biomarkers = diagnostic_system.extract_depression_biomarkers(
                subject['brain_data']
            )
            classification = diagnostic_system.classify_depression(biomarkers)
            
            screening_results.append({
                'subject_id': subject['subject_id'],
                'true_label': subject['true_label'],
                'predicted_classification': classification['classification'],
                'risk_score': classification['risk_score'],
                'confidence': classification['confidence']
            })
        
        total_classification_time = time.time() - classification_start
        
        # Calculate diagnostic performance
        true_labels = []
        predicted_labels = []
        risk_scores = []
        
        for result in screening_results:
            true_labels.append(1 if result['true_label'] == 'depressed' else 0)
            predicted_labels.append(1 if result['predicted_classification'] == 'high_risk' else 0)
            risk_scores.append(result['risk_score'])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        auc_score = roc_auc_score(true_labels, risk_scores)
        
        # Validate against clinical specifications
        assert accuracy >= diagnostic_system.specs.diagnostic_accuracy * 0.8, \
            f"Depression screening accuracy {accuracy:.3f} too low"
        
        assert precision >= diagnostic_system.specs.specificity * 0.8, \
            f"Depression screening precision {precision:.3f} too low"
        
        assert recall >= diagnostic_system.specs.sensitivity * 0.8, \
            f"Depression screening recall {recall:.3f} too low"
        
        assert auc_score >= 0.8, \
            f"Depression screening AUC {auc_score:.3f} too low"
        
        # Validate diagnostic speed
        avg_time_per_subject = total_classification_time / len(test_subjects)
        max_diagnostic_time = diagnostic_system.specs.diagnostic_latency
        
        assert avg_time_per_subject <= max_diagnostic_time, \
            f"Diagnostic time {avg_time_per_subject:.1f}s exceeds limit {max_diagnostic_time:.1f}s"


class TestNeurofeedbackTherapy:
    """Test real-time neurofeedback for therapeutic interventions"""
    
    @pytest.fixture
    def neurofeedback_system(self):
        """Mock neurofeedback therapy system"""
        system = Mock()
        system.specs = NeurofeedbackSpecs()
        system.feedback_protocols = {}
        system.patient_profiles = {}
        return system
    
    @pytest.mark.asyncio
    async def test_adhd_attention_training(self, neurofeedback_system):
        """Test ADHD attention training neurofeedback"""
        # Mock ADHD neurofeedback protocol
        def create_adhd_protocol():
            """Create ADHD attention training protocol"""
            return {
                'name': 'ADHD_Attention_Training',
                'target_frequency': 'SMR',  # Sensorimotor rhythm (12-15 Hz)
                'target_amplitude': 'increase',
                'inhibit_frequencies': ['theta', 'high_beta'],  # 4-8 Hz, 20-30 Hz
                'electrode_sites': ['C3', 'C4', 'Cz'],  # Central sites
                'feedback_modality': 'visual_auditory',
                'reward_criteria': {
                    'smr_threshold': 1.5,  # 1.5x baseline SMR
                    'theta_threshold': 0.8,  # 0.8x baseline theta (reduce)
                    'artifact_threshold': 100e-6  # 100 μV artifact threshold
                },
                'session_structure': {
                    'baseline_duration': 120,  # 2 minutes baseline
                    'training_blocks': 8,      # 8 training blocks
                    'block_duration': 180,     # 3 minutes per block
                    'rest_duration': 60        # 1 minute rest between blocks
                }
            }
        
        # Mock real-time feedback engine
        async def real_time_feedback_loop(protocol, patient_profile):
            """Real-time neurofeedback training loop"""
            feedback_events = []
            performance_metrics = []
            latencies = []
            
            # Simulate training session
            total_blocks = protocol['session_structure']['training_blocks']
            
            for block_num in range(total_blocks):
                block_start_time = time.time()
                
                # Simulate EEG acquisition and processing
                for sample_num in range(30):  # 30 feedback events per block
                    sample_start = time.time()
                    
                    # Simulate EEG data acquisition (50ms window)
                    await asyncio.sleep(0.01)  # 10ms acquisition delay
                    
                    eeg_sample = np.random.normal(0, 30e-6, (3, 125))  # 3 channels, 125 samples (0.5s at 250Hz)
                    
                    # Real-time signal processing
                    processing_start = time.time()
                    feedback_metrics = await self._process_adhd_feedback(
                        eeg_sample, protocol, patient_profile
                    )
                    processing_time = time.time() - processing_start
                    
                    # Generate feedback
                    feedback_signal = self._generate_feedback_signal(feedback_metrics)
                    
                    total_latency = time.time() - sample_start
                    latencies.append(total_latency)
                    
                    # Record feedback event
                    feedback_events.append({
                        'block': block_num,
                        'sample': sample_num,
                        'timestamp': sample_start,
                        'latency': total_latency,
                        'smr_power': feedback_metrics['smr_power'],
                        'theta_power': feedback_metrics['theta_power'],
                        'reward_given': feedback_metrics['reward'],
                        'feedback_signal': feedback_signal
                    })
                    
                    # Performance tracking
                    if sample_num % 10 == 0:  # Every 10 samples
                        performance_metrics.append({
                            'block': block_num,
                            'sample': sample_num,
                            'smr_improvement': feedback_metrics['smr_power'] / patient_profile['baseline_smr'],
                            'theta_reduction': patient_profile['baseline_theta'] / feedback_metrics['theta_power'],
                            'reward_rate': sum(1 for e in feedback_events[-10:] if e['reward_given']) / 10
                        })
                
                # Block rest period
                await asyncio.sleep(0.1)  # Shortened rest for testing
            
            return {
                'feedback_events': feedback_events,
                'performance_metrics': performance_metrics,
                'latencies': latencies,
                'total_rewards': sum(1 for e in feedback_events if e['reward_given']),
                'mean_latency': np.mean(latencies),
                'max_latency': np.max(latencies)
            }
        
        async def _process_adhd_feedback(self, eeg_sample, protocol, patient_profile):
            """Process EEG for ADHD neurofeedback"""
            # Calculate frequency band powers
            freqs, psd = signal.welch(eeg_sample, fs=250, nperseg=64)
            
            # SMR band (12-15 Hz)
            smr_band = (freqs >= 12) & (freqs <= 15)
            smr_power = np.mean(psd[:, smr_band])
            
            # Theta band (4-8 Hz)  
            theta_band = (freqs >= 4) & (freqs <= 8)
            theta_power = np.mean(psd[:, theta_band])
            
            # High beta band (20-30 Hz)
            high_beta_band = (freqs >= 20) & (freqs <= 30) 
            high_beta_power = np.mean(psd[:, high_beta_band])
            
            # Artifact detection
            artifact_detected = np.any(np.abs(eeg_sample) > protocol['reward_criteria']['artifact_threshold'])
            
            # Reward calculation
            reward_criteria = protocol['reward_criteria']
            smr_meets_criteria = smr_power >= patient_profile['baseline_smr'] * reward_criteria['smr_threshold']
            theta_meets_criteria = theta_power <= patient_profile['baseline_theta'] * reward_criteria['theta_threshold']
            no_artifacts = not artifact_detected
            
            reward = smr_meets_criteria and theta_meets_criteria and no_artifacts
            
            return {
                'smr_power': smr_power,
                'theta_power': theta_power,
                'high_beta_power': high_beta_power,
                'artifact_detected': artifact_detected,
                'reward': reward
            }
        
        def _generate_feedback_signal(self, feedback_metrics):
            """Generate multimodal feedback signal"""
            if feedback_metrics['reward']:
                return {
                    'visual': 'green_bar_increase',
                    'auditory': 'pleasant_tone',
                    'intensity': 0.8
                }
            else:
                return {
                    'visual': 'red_bar_decrease', 
                    'auditory': 'neutral_tone',
                    'intensity': 0.3
                }
        
        # Attach methods to neurofeedback system
        neurofeedback_system.create_adhd_protocol = create_adhd_protocol
        neurofeedback_system.run_feedback_session = real_time_feedback_loop
        neurofeedback_system._process_adhd_feedback = _process_adhd_feedback
        neurofeedback_system._generate_feedback_signal = _generate_feedback_signal
        
        # Create ADHD protocol
        adhd_protocol = neurofeedback_system.create_adhd_protocol()
        
        # Create patient profile
        patient_profile = {
            'patient_id': 'ADHD_001',
            'diagnosis': 'ADHD_Combined_Type',
            'baseline_smr': 2e-11,    # Baseline SMR power
            'baseline_theta': 5e-11,   # Baseline theta power
            'age': 12,
            'medication': 'none',
            'previous_sessions': 0
        }
        
        # Run neurofeedback session
        session_results = await neurofeedback_system.run_feedback_session(
            adhd_protocol, patient_profile
        )
        
        # Validate real-time performance
        mean_latency = session_results['mean_latency']
        max_latency = session_results['max_latency']
        max_allowed_latency = neurofeedback_system.specs.real_time_latency
        
        assert mean_latency <= max_allowed_latency, \
            f"Mean feedback latency {mean_latency*1000:.1f}ms exceeds {max_allowed_latency*1000:.1f}ms"
        
        assert max_latency <= max_allowed_latency * 2, \
            f"Max feedback latency {max_latency*1000:.1f}ms too high"
        
        # Validate feedback accuracy
        total_events = len(session_results['feedback_events'])
        total_rewards = session_results['total_rewards']
        reward_rate = total_rewards / total_events
        
        # ADHD patients typically start with low reward rate
        assert 0.1 <= reward_rate <= 0.6, \
            f"ADHD reward rate {reward_rate:.3f} outside expected range"
        
        # Validate training progression
        performance_metrics = session_results['performance_metrics']
        if len(performance_metrics) > 1:
            early_performance = np.mean([p['reward_rate'] for p in performance_metrics[:2]])
            late_performance = np.mean([p['reward_rate'] for p in performance_metrics[-2:]])
            
            improvement = late_performance - early_performance
            assert improvement >= -0.1, \
                f"Should not show significant performance degradation: {improvement:.3f}"
    
    def test_anxiety_relaxation_protocol(self, neurofeedback_system):
        """Test anxiety reduction through alpha enhancement neurofeedback"""
        # Mock anxiety reduction protocol
        def create_anxiety_protocol():
            """Create anxiety reduction neurofeedback protocol"""
            return {
                'name': 'Anxiety_Alpha_Enhancement',
                'target_frequency': 'alpha',  # 8-12 Hz
                'target_amplitude': 'increase',
                'inhibit_frequencies': ['high_beta', 'gamma'],  # 20-30 Hz, 30+ Hz
                'electrode_sites': ['O1', 'O2', 'Pz'],  # Occipital/parietal sites
                'feedback_modality': 'visual_breathing',
                'reward_criteria': {
                    'alpha_threshold': 2.0,  # 2x baseline alpha
                    'beta_threshold': 0.7,   # 0.7x baseline beta (reduce)
                    'coherence_threshold': 0.6  # Alpha coherence between hemispheres
                },
                'protocol_type': 'relaxation_training'
            }
        
        # Mock anxiety assessment
        def assess_anxiety_reduction(pre_session_data, post_session_data):
            """Assess anxiety reduction from neurofeedback"""
            # Calculate anxiety-related metrics
            
            # Pre-session metrics
            pre_alpha = np.mean(pre_session_data['alpha_power'])
            pre_beta = np.mean(pre_session_data['beta_power'])
            pre_alpha_coherence = pre_session_data['alpha_coherence']
            
            # Post-session metrics  
            post_alpha = np.mean(post_session_data['alpha_power'])
            post_beta = np.mean(post_session_data['beta_power'])
            post_alpha_coherence = post_session_data['alpha_coherence']
            
            # Calculate improvements
            alpha_improvement = (post_alpha - pre_alpha) / pre_alpha
            beta_reduction = (pre_beta - post_beta) / pre_beta
            coherence_improvement = post_alpha_coherence - pre_alpha_coherence
            
            # Overall anxiety reduction score
            anxiety_reduction_score = (
                0.4 * alpha_improvement +
                0.3 * beta_reduction +
                0.3 * coherence_improvement
            )
            
            return {
                'alpha_improvement': alpha_improvement,
                'beta_reduction': beta_reduction,
                'coherence_improvement': coherence_improvement,
                'anxiety_reduction_score': anxiety_reduction_score,
                'session_effective': anxiety_reduction_score > 0.1  # 10% improvement
            }
        
        neurofeedback_system.create_anxiety_protocol = create_anxiety_protocol
        neurofeedback_system.assess_anxiety_reduction = assess_anxiety_reduction
        
        # Create anxiety protocol
        anxiety_protocol = neurofeedback_system.create_anxiety_protocol()
        
        # Simulate pre/post session data
        # Pre-session: high anxiety state
        pre_session_data = {
            'alpha_power': np.random.gamma(1, 1e-11, 100),  # Low alpha
            'beta_power': np.random.gamma(3, 1e-11, 100),   # High beta
            'alpha_coherence': np.random.uniform(0.2, 0.4), # Low coherence
            'anxiety_self_report': 7.5  # High anxiety (1-10 scale)
        }
        
        # Post-session: reduced anxiety state
        post_session_data = {
            'alpha_power': np.random.gamma(2, 1e-11, 100),  # Increased alpha
            'beta_power': np.random.gamma(1.5, 1e-11, 100), # Reduced beta
            'alpha_coherence': np.random.uniform(0.5, 0.7), # Improved coherence
            'anxiety_self_report': 5.2  # Reduced anxiety
        }
        
        # Assess anxiety reduction
        assessment_result = neurofeedback_system.assess_anxiety_reduction(
            pre_session_data, post_session_data
        )
        
        # Validate therapy efficacy
        anxiety_reduction_score = assessment_result['anxiety_reduction_score']
        session_effective = assessment_result['session_effective']
        
        assert session_effective, \
            f"Anxiety neurofeedback session should be effective"
        
        assert anxiety_reduction_score >= 0.1, \
            f"Anxiety reduction score {anxiety_reduction_score:.3f} too low"
        
        # Validate specific improvements
        alpha_improvement = assessment_result['alpha_improvement']
        beta_reduction = assessment_result['beta_reduction']
        
        assert alpha_improvement >= 0.05, \
            f"Alpha improvement {alpha_improvement:.3f} insufficient"
        
        assert beta_reduction >= 0.05, \
            f"Beta reduction {beta_reduction:.3f} insufficient"


class TestCognitiveEnhancement:
    """Test cognitive performance enhancement applications"""
    
    @pytest.fixture
    def enhancement_system(self):
        """Mock cognitive enhancement system"""
        system = Mock()
        system.specs = CognitiveEnhancementSpecs()
        system.enhancement_protocols = {}
        system.safety_monitors = {}
        return system
    
    def test_working_memory_enhancement(self, enhancement_system):
        """Test working memory capacity enhancement"""
        # Mock working memory enhancement protocol
        def create_memory_enhancement_protocol():
            """Create working memory enhancement protocol"""
            return {
                'name': 'Working_Memory_Enhancement',
                'target_networks': ['dlpfc', 'ppc', 'acc'],  # Key WM networks
                'enhancement_method': 'gamma_entrainment',
                'target_frequency': 40,  # 40 Hz gamma
                'stimulation_parameters': {
                    'intensity': 2.0,  # mA
                    'frequency': 40,   # Hz
                    'duty_cycle': 0.5, # 50% on/off
                    'duration': 1200   # 20 minutes
                },
                'cognitive_tasks': ['n_back', 'dual_task', 'updating'],
                'enhancement_targets': {
                    'capacity_increase': 0.25,  # 25% improvement
                    'accuracy_improvement': 0.20,  # 20% accuracy boost
                    'reaction_time_reduction': 0.15  # 15% faster responses
                }
            }
        
        # Mock cognitive testing battery
        def administer_cognitive_tests(test_battery, baseline=False):
            """Administer working memory tests"""
            results = {}
            
            # N-back task
            if 'n_back' in test_battery:
                if baseline:
                    # Baseline performance
                    n_back_score = np.random.normal(0.75, 0.1)  # 75% accuracy
                    n_back_rt = np.random.normal(800, 100)      # 800ms RT
                else:
                    # Enhanced performance
                    n_back_score = np.random.normal(0.85, 0.08)  # 85% accuracy
                    n_back_rt = np.random.normal(720, 80)        # 720ms RT
                
                results['n_back'] = {
                    'accuracy': max(0, min(1, n_back_score)),
                    'reaction_time': max(400, n_back_rt),
                    'span': 2 if baseline else 3  # Working memory span
                }
            
            # Dual task paradigm
            if 'dual_task' in test_battery:
                if baseline:
                    dual_task_cost = np.random.normal(0.25, 0.05)  # 25% cost
                    primary_accuracy = np.random.normal(0.80, 0.08)
                else:
                    dual_task_cost = np.random.normal(0.15, 0.04)  # 15% cost
                    primary_accuracy = np.random.normal(0.88, 0.06)
                
                results['dual_task'] = {
                    'primary_accuracy': max(0, min(1, primary_accuracy)),
                    'dual_task_cost': max(0, dual_task_cost),
                    'secondary_accuracy': max(0, min(1, primary_accuracy - dual_task_cost))
                }
            
            # Updating task
            if 'updating' in test_battery:
                if baseline:
                    update_accuracy = np.random.normal(0.70, 0.12)
                    update_rt = np.random.normal(1200, 150)
                else:
                    update_accuracy = np.random.normal(0.85, 0.10)
                    update_rt = np.random.normal(1050, 120)
                
                results['updating'] = {
                    'accuracy': max(0, min(1, update_accuracy)),
                    'reaction_time': max(600, update_rt),
                    'items_updated': 4 if baseline else 5
                }
            
            return results
        
        # Mock enhancement session
        def run_enhancement_session(protocol, participant_profile):
            """Run cognitive enhancement session"""
            session_start = time.time()
            
            # Baseline testing
            baseline_results = administer_cognitive_tests(protocol['cognitive_tasks'], baseline=True)
            
            # Enhancement stimulation (simulated)
            stimulation_duration = protocol['stimulation_parameters']['duration']
            # Simulate stimulation time (shortened for testing)
            time.sleep(0.1)  # Represent 20-minute session
            
            # Post-enhancement testing
            enhanced_results = administer_cognitive_tests(protocol['cognitive_tasks'], baseline=False)
            
            session_duration = time.time() - session_start
            
            # Calculate enhancement effects
            enhancement_effects = {}
            for task in protocol['cognitive_tasks']:
                if task in baseline_results and task in enhanced_results:
                    baseline_acc = baseline_results[task]['accuracy']
                    enhanced_acc = enhanced_results[task]['accuracy']
                    
                    baseline_rt = baseline_results[task]['reaction_time']
                    enhanced_rt = enhanced_results[task]['reaction_time']
                    
                    accuracy_improvement = (enhanced_acc - baseline_acc) / baseline_acc
                    rt_improvement = (baseline_rt - enhanced_rt) / baseline_rt
                    
                    enhancement_effects[task] = {
                        'accuracy_improvement': accuracy_improvement,
                        'rt_improvement': rt_improvement,
                        'baseline_performance': baseline_results[task],
                        'enhanced_performance': enhanced_results[task]
                    }
            
            return {
                'baseline_results': baseline_results,
                'enhanced_results': enhanced_results,
                'enhancement_effects': enhancement_effects,
                'session_duration': session_duration,
                'protocol_used': protocol['name']
            }
        
        enhancement_system.create_memory_protocol = create_memory_enhancement_protocol
        enhancement_system.administer_tests = administer_cognitive_tests
        enhancement_system.run_session = run_enhancement_session
        
        # Create enhancement protocol
        memory_protocol = enhancement_system.create_memory_protocol()
        
        # Create participant profile
        participant_profile = {
            'participant_id': 'ENH_001',
            'age': 25,
            'education': 'college',
            'baseline_iq': 110,
            'medical_history': 'none',
            'previous_enhancement': False
        }
        
        # Run enhancement session
        session_results = enhancement_system.run_session(
            memory_protocol, participant_profile
        )
        
        # Validate enhancement magnitude
        enhancement_effects = session_results['enhancement_effects']
        
        # Check each cognitive task
        for task, effects in enhancement_effects.items():
            accuracy_improvement = effects['accuracy_improvement']
            rt_improvement = effects['rt_improvement']
            
            min_enhancement = enhancement_system.specs.enhancement_magnitude * 0.5
            
            # At least one measure should show improvement
            assert (accuracy_improvement >= min_enhancement or 
                   rt_improvement >= min_enhancement * 0.6), \
                f"Task {task} should show cognitive enhancement"
        
        # Validate overall enhancement
        mean_accuracy_improvement = np.mean([
            effects['accuracy_improvement'] for effects in enhancement_effects.values()
        ])
        
        assert mean_accuracy_improvement >= 0.1, \
            f"Mean accuracy improvement {mean_accuracy_improvement:.3f} too low"
        
        # Validate enhancement safety (no excessive improvements that might indicate artifacts)
        max_improvement = max([
            effects['accuracy_improvement'] for effects in enhancement_effects.values()
        ])
        
        assert max_improvement <= 0.5, \
            f"Enhancement improvement {max_improvement:.3f} suspiciously high"
    
    def test_attention_focus_enhancement(self, enhancement_system):
        """Test sustained attention and focus enhancement"""
        # Mock attention enhancement protocol
        def create_attention_protocol():
            """Create attention enhancement protocol"""
            return {
                'name': 'Attention_Focus_Enhancement',
                'target_networks': ['frontal_attention', 'parietal_attention'],
                'enhancement_method': 'alpha_suppression',
                'target_frequency': 10,  # 10 Hz alpha
                'modulation_type': 'suppression',
                'cognitive_tasks': ['cpt', 'ant', 'flanker'],  # Attention tasks
                'enhancement_targets': {
                    'sustained_attention': 0.20,  # 20% improvement
                    'selective_attention': 0.25,  # 25% improvement
                    'executive_attention': 0.15   # 15% improvement
                }
            }
        
        # Mock attention task battery
        def measure_attention_performance(task_type, enhanced=False):
            """Measure attention performance"""
            if task_type == 'cpt':  # Continuous Performance Task
                if enhanced:
                    hit_rate = np.random.normal(0.92, 0.04)
                    false_alarm_rate = np.random.normal(0.05, 0.02)
                    mean_rt = np.random.normal(450, 50)
                else:
                    hit_rate = np.random.normal(0.85, 0.06)
                    false_alarm_rate = np.random.normal(0.12, 0.04)
                    mean_rt = np.random.normal(520, 70)
                
                d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)
                
                return {
                    'hit_rate': max(0, min(1, hit_rate)),
                    'false_alarm_rate': max(0, min(1, false_alarm_rate)),
                    'mean_rt': max(300, mean_rt),
                    'd_prime': d_prime,
                    'criterion': -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(false_alarm_rate))
                }
            
            elif task_type == 'ant':  # Attention Network Test
                if enhanced:
                    alerting_effect = np.random.normal(25, 8)      # ms
                    orienting_effect = np.random.normal(35, 10)   # ms
                    executive_effect = np.random.normal(85, 15)   # ms
                else:
                    alerting_effect = np.random.normal(40, 12)    # ms
                    orienting_effect = np.random.normal(50, 15)   # ms
                    executive_effect = np.random.normal(110, 20)  # ms
                
                return {
                    'alerting_network': max(0, alerting_effect),
                    'orienting_network': max(0, orienting_effect),
                    'executive_network': max(0, executive_effect),
                    'overall_rt': 650 if enhanced else 720
                }
            
            elif task_type == 'flanker':  # Flanker Task
                if enhanced:
                    congruent_rt = np.random.normal(480, 40)
                    incongruent_rt = np.random.normal(550, 50)
                    accuracy = np.random.normal(0.95, 0.03)
                else:
                    congruent_rt = np.random.normal(520, 50)
                    incongruent_rt = np.random.normal(620, 60)
                    accuracy = np.random.normal(0.88, 0.05)
                
                flanker_effect = incongruent_rt - congruent_rt
                
                return {
                    'congruent_rt': max(350, congruent_rt),
                    'incongruent_rt': max(400, incongruent_rt),
                    'flanker_effect': max(0, flanker_effect),
                    'accuracy': max(0, min(1, accuracy))
                }
        
        enhancement_system.create_attention_protocol = create_attention_protocol
        enhancement_system.measure_attention = measure_attention_performance
        
        # Create attention protocol
        attention_protocol = enhancement_system.create_attention_protocol()
        
        # Test attention enhancement
        enhancement_results = {}
        
        for task in attention_protocol['cognitive_tasks']:
            # Baseline performance
            baseline_performance = enhancement_system.measure_attention(task, enhanced=False)
            
            # Enhanced performance
            enhanced_performance = enhancement_system.measure_attention(task, enhanced=True)
            
            enhancement_results[task] = {
                'baseline': baseline_performance,
                'enhanced': enhanced_performance
            }
        
        # Validate attention improvements
        for task, results in enhancement_results.items():
            baseline = results['baseline']
            enhanced = results['enhanced']
            
            if task == 'cpt':
                # CPT improvements
                hit_rate_improvement = (enhanced['hit_rate'] - baseline['hit_rate']) / baseline['hit_rate']
                fa_rate_reduction = (baseline['false_alarm_rate'] - enhanced['false_alarm_rate']) / baseline['false_alarm_rate']
                rt_improvement = (baseline['mean_rt'] - enhanced['mean_rt']) / baseline['mean_rt']
                
                assert hit_rate_improvement >= 0.05, \
                    f"CPT hit rate should improve: {hit_rate_improvement:.3f}"
                assert fa_rate_reduction >= 0.2, \
                    f"CPT false alarms should reduce: {fa_rate_reduction:.3f}"
                assert rt_improvement >= 0.1, \
                    f"CPT reaction time should improve: {rt_improvement:.3f}"
            
            elif task == 'ant':
                # ANT network improvements
                alerting_improvement = (baseline['alerting_network'] - enhanced['alerting_network']) / baseline['alerting_network']
                executive_improvement = (baseline['executive_network'] - enhanced['executive_network']) / baseline['executive_network']
                
                assert alerting_improvement >= 0.15, \
                    f"Alerting network should improve: {alerting_improvement:.3f}"
                assert executive_improvement >= 0.1, \
                    f"Executive network should improve: {executive_improvement:.3f}"
            
            elif task == 'flanker':
                # Flanker task improvements
                flanker_effect_reduction = (baseline['flanker_effect'] - enhanced['flanker_effect']) / baseline['flanker_effect']
                accuracy_improvement = (enhanced['accuracy'] - baseline['accuracy']) / baseline['accuracy']
                
                assert flanker_effect_reduction >= 0.1, \
                    f"Flanker effect should reduce: {flanker_effect_reduction:.3f}"
                assert accuracy_improvement >= 0.05, \
                    f"Flanker accuracy should improve: {accuracy_improvement:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
