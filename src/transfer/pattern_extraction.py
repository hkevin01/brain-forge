"""
Pattern Transfer Learning Module

Implementation of brain pattern extraction and transfer algorithms for
mapping learned patterns between subjects and transferring to neural
simulations.

This module provides the foundation for the transfer learning capabilities
referenced throughout the Brain-Forge system.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from core.config import Config
from core.exceptions import ValidationError, ProcessingError
from core.logger import get_logger, LogContext


@dataclass
class BrainPattern:
    """Represents an extracted brain pattern"""
    pattern_id: str
    subject_id: str
    pattern_type: str  # 'motor', 'visual', 'cognitive', etc.
    features: np.ndarray
    metadata: Dict[str, Any]
    extraction_timestamp: float
    quality_score: float


@dataclass
class TransferResult:
    """Results from pattern transfer operation"""
    source_pattern_id: str
    target_subject_id: str
    transfer_accuracy: float
    adapted_features: np.ndarray
    confidence_score: float
    transfer_timestamp: float


class PatternExtractor:
    """Extracts brain patterns from multi-modal neural data"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize pattern extractor
        
        Args:
            config: System configuration
        """
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.PatternExtractor")
        
        # Pattern extraction parameters
        self.extraction_params = \
            self.config.transfer_learning.pattern_extraction
        self.frequency_bands = self.extraction_params.frequency_bands
        self.spatial_filters = self.extraction_params.spatial_filters

    def extract_motor_patterns(self, neural_data: np.ndarray,
                               labels: np.ndarray) -> List[BrainPattern]:
        """
        Extract motor-related brain patterns
        
        Args:
            neural_data: Multi-channel neural data (channels x time x trials)
            labels: Movement labels for each trial
            
        Returns:
            List of extracted motor patterns
        """
        context = LogContext(processing_stage="motor_pattern_extraction")
        
        try:
            patterns = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                # Extract trials for this motor pattern
                trial_indices = np.where(labels == label)[0]
                pattern_data = neural_data[:, :, trial_indices]
                
                # Extract features using CSP-like spatial filtering
                features = self._extract_spatial_spectral_features(
                    pattern_data, pattern_type='motor'
                )
                
                # Calculate pattern quality
                quality = self._calculate_pattern_quality(
                    features, pattern_data
                )
                
                pattern = BrainPattern(
                    pattern_id=f"motor_{label}_{len(patterns)}",
                    subject_id=self.extraction_params.current_subject_id,
                    pattern_type=f"motor_{label}",
                    features=features,
                    metadata={
                        'label': label,
                        'num_trials': len(trial_indices),
                        'data_shape': pattern_data.shape,
                        'extraction_method': 'spatial_spectral_csp'
                    },
                    extraction_timestamp=self._get_timestamp(),
                    quality_score=quality
                )
                
                patterns.append(pattern)
                
            self.logger.info(f"Extracted {len(patterns)} motor patterns", context)
            return patterns
            
        except Exception as e:
            error_msg = f"Motor pattern extraction failed: {str(e)}"
            self.logger.error(error_msg, context, exception=e)
            raise ProcessingError(error_msg, processing_stage="pattern_extraction")
    
    def extract_cognitive_patterns(self, neural_data: np.ndarray,
                                 task_conditions: List[str]) -> List[BrainPattern]:
        """Extract cognitive task-related patterns"""
        context = LogContext(processing_stage="cognitive_pattern_extraction")
        
        try:
            patterns = []
            
            for condition in task_conditions:
                # Extract condition-specific features
                features = self._extract_cognitive_features(neural_data, condition)
                
                # Calculate pattern discriminability
                quality = self._calculate_cognitive_pattern_quality(features)
                
                pattern = BrainPattern(
                    pattern_id=f"cognitive_{condition}_{len(patterns)}",
                    subject_id=self.extraction_params.current_subject_id,
                    pattern_type=f"cognitive_{condition}",
                    features=features,
                    metadata={
                        'condition': condition,
                        'extraction_method': 'multiband_connectivity',
                        'frequency_bands': self.frequency_bands
                    },
                    extraction_timestamp=self._get_timestamp(),
                    quality_score=quality
                )
                
                patterns.append(pattern)
            
            self.logger.info(f"Extracted {len(patterns)} cognitive patterns", context)
            return patterns
            
        except Exception as e:
            error_msg = f"Cognitive pattern extraction failed: {str(e)}"
            self.logger.error(error_msg, context, exception=e)
            raise ProcessingError(error_msg, processing_stage="pattern_extraction")
    
    def _extract_spatial_spectral_features(self, data: np.ndarray, 
                                         pattern_type: str) -> np.ndarray:
        """Extract spatial-spectral features using CSP-like methods"""
        channels, timepoints, trials = data.shape
        
        # Calculate covariance matrices for each frequency band
        features = []
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Bandpass filter data (simplified - would use proper filtering)
            filtered_data = self._apply_bandpass_filter(data, low_freq, high_freq)
            
            # Calculate spatial covariance
            cov_matrices = []
            for trial in range(trials):
                trial_data = filtered_data[:, :, trial]
                cov_matrix = np.cov(trial_data)
                cov_matrices.append(cov_matrix)
            
            # Average covariance matrix
            avg_cov = np.mean(cov_matrices, axis=0)
            
            # Extract eigenvalues as features
            eigenvals, _ = np.linalg.eigh(avg_cov)
            features.append(eigenvals[-10:])  # Top 10 eigenvalues
        
        return np.concatenate(features)
    
    def _extract_cognitive_features(self, data: np.ndarray, 
                                  condition: str) -> np.ndarray:
        """Extract cognitive task features"""
        # Simplified feature extraction
        # In practice, would use more sophisticated methods
        
        # Calculate spectral power in different frequency bands
        spectral_features = []
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Calculate power in this frequency band
            power = np.mean(np.abs(data) ** 2, axis=1)  # Power per channel
            spectral_features.append(power)
        
        # Calculate connectivity features
        connectivity_matrix = np.corrcoef(data)
        connectivity_features = connectivity_matrix[np.triu_indices(
            connectivity_matrix.shape[0], k=1
        )]
        
        # Combine features
        features = np.concatenate([
            np.concatenate(spectral_features),
            connectivity_features
        ])
        
        return features
    
    def _apply_bandpass_filter(self, data: np.ndarray, 
                             low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter (simplified implementation)"""
        # In practice, would use proper filtering from processing module
        # For now, return data as-is
        return data
    
    def _calculate_pattern_quality(self, features: np.ndarray, 
                                 data: np.ndarray) -> float:
        """Calculate pattern quality score"""
        # Simplified quality metric based on feature variance and consistency
        feature_variance = np.var(features)
        data_snr = np.mean(data) / np.std(data) if np.std(data) > 0 else 0
        
        # Combine metrics (0-1 scale)
        quality = min(1.0, max(0.0, (feature_variance + abs(data_snr)) / 10.0))
        return quality
    
    def _calculate_cognitive_pattern_quality(self, features: np.ndarray) -> float:
        """Calculate cognitive pattern quality"""
        # Based on feature discriminability
        feature_range = np.max(features) - np.min(features)
        feature_std = np.std(features)
        
        quality = min(1.0, max(0.0, feature_range * feature_std / 100.0))
        return quality
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()


class FeatureMapper:
    """Maps features between different subjects and contexts"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize feature mapper"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.FeatureMapper")
        
    def map_between_subjects(self, source_pattern: BrainPattern,
                           target_subject_data: np.ndarray) -> np.ndarray:
        """
        Map pattern features from source to target subject
        
        Args:
            source_pattern: Pattern from source subject
            target_subject_data: Neural data from target subject
            
        Returns:
            Adapted features for target subject
        """
        context = LogContext(processing_stage="feature_mapping")
        
        try:
            # Extract baseline features from target subject
            target_features = self._extract_baseline_features(target_subject_data)
            
            # Apply domain adaptation transformation
            adapted_features = self._apply_domain_adaptation(
                source_pattern.features, target_features
            )
            
            self.logger.info("Feature mapping completed", context)
            return adapted_features
            
        except Exception as e:
            error_msg = f"Feature mapping failed: {str(e)}"
            self.logger.error(error_msg, context, exception=e)
            raise ProcessingError(error_msg, processing_stage="feature_mapping")
    
    def _extract_baseline_features(self, data: np.ndarray) -> np.ndarray:
        """Extract baseline features from target subject"""
        # Simplified baseline feature extraction
        mean_features = np.mean(data, axis=1)
        std_features = np.std(data, axis=1)
        return np.concatenate([mean_features, std_features])
    
    def _apply_domain_adaptation(self, source_features: np.ndarray,
                               target_features: np.ndarray) -> np.ndarray:
        """Apply domain adaptation transformation"""
        # Simplified domain adaptation using linear transformation
        # In practice, would use more sophisticated methods
        
        if len(source_features) != len(target_features):
            # Resize to match dimensions
            min_len = min(len(source_features), len(target_features))
            source_features = source_features[:min_len]
            target_features = target_features[:min_len]
        
        # Apply simple linear adaptation
        adaptation_factor = np.mean(target_features) / (np.mean(source_features) + 1e-8)
        adapted_features = source_features * adaptation_factor
        
        return adapted_features


class TransferLearningEngine:
    """Main transfer learning engine coordinating pattern extraction and mapping"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize transfer learning engine"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.TransferLearningEngine")
        
        self.pattern_extractor = PatternExtractor(config)
        self.feature_mapper = FeatureMapper(config)
        
        # Storage for patterns
        self.pattern_database: Dict[str, BrainPattern] = {}
    
    def learn_patterns(self, subject_id: str, neural_data: np.ndarray,
                      labels: np.ndarray, pattern_types: List[str]) -> List[BrainPattern]:
        """
        Learn patterns from subject data
        
        Args:
            subject_id: Subject identifier
            neural_data: Multi-channel neural data
            labels: Labels for pattern classification
            pattern_types: Types of patterns to extract
            
        Returns:
            List of learned patterns
        """
        context = LogContext(processing_stage="pattern_learning")
        
        try:
            all_patterns = []
            
            # Set current subject in config
            self.config.transfer_learning.pattern_extraction.current_subject_id = subject_id
            
            for pattern_type in pattern_types:
                if pattern_type == 'motor':
                    patterns = self.pattern_extractor.extract_motor_patterns(
                        neural_data, labels
                    )
                elif pattern_type == 'cognitive':
                    # Convert labels to task conditions
                    task_conditions = [f"task_{label}" for label in np.unique(labels)]
                    patterns = self.pattern_extractor.extract_cognitive_patterns(
                        neural_data, task_conditions
                    )
                else:
                    self.logger.warning(f"Unknown pattern type: {pattern_type}", context)
                    continue
                
                all_patterns.extend(patterns)
            
            # Store patterns in database
            for pattern in all_patterns:
                self.pattern_database[pattern.pattern_id] = pattern
            
            self.logger.info(f"Learned {len(all_patterns)} patterns for subject {subject_id}", context)
            return all_patterns
            
        except Exception as e:
            error_msg = f"Pattern learning failed: {str(e)}"
            self.logger.error(error_msg, context, exception=e)
            raise ProcessingError(error_msg, processing_stage="pattern_learning")
    
    def transfer_pattern(self, pattern_id: str, target_subject_id: str,
                        target_data: np.ndarray) -> TransferResult:
        """
        Transfer learned pattern to target subject
        
        Args:
            pattern_id: ID of pattern to transfer
            target_subject_id: Target subject ID
            target_data: Target subject neural data
            
        Returns:
            Transfer result with adapted pattern
        """
        context = LogContext(processing_stage="pattern_transfer")
        
        try:
            # Get source pattern
            if pattern_id not in self.pattern_database:
                raise ValidationError(f"Pattern {pattern_id} not found in database")
            
            source_pattern = self.pattern_database[pattern_id]
            
            # Map pattern to target subject
            adapted_features = self.feature_mapper.map_between_subjects(
                source_pattern, target_data
            )
            
            # Calculate transfer metrics
            transfer_accuracy = self._calculate_transfer_accuracy(
                source_pattern.features, adapted_features
            )
            
            confidence_score = min(1.0, transfer_accuracy * source_pattern.quality_score)
            
            result = TransferResult(
                source_pattern_id=pattern_id,
                target_subject_id=target_subject_id,
                transfer_accuracy=transfer_accuracy,
                adapted_features=adapted_features,
                confidence_score=confidence_score,
                transfer_timestamp=self._get_timestamp()
            )
            
            self.logger.info(f"Pattern transfer completed with {transfer_accuracy:.2f} accuracy", context)
            return result
            
        except Exception as e:
            error_msg = f"Pattern transfer failed: {str(e)}"
            self.logger.error(error_msg, context, exception=e)
            raise ProcessingError(error_msg, processing_stage="pattern_transfer")
    
    def _calculate_transfer_accuracy(self, source_features: np.ndarray,
                                   adapted_features: np.ndarray) -> float:
        """Calculate transfer accuracy metric"""
        # Simplified accuracy based on feature similarity
        correlation = np.corrcoef(source_features, adapted_features)[0, 1]
        accuracy = max(0.0, min(1.0, (correlation + 1.0) / 2.0))  # Scale to 0-1
        return accuracy
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of stored patterns"""
        summary = {
            'total_patterns': len(self.pattern_database),
            'pattern_types': {},
            'subjects': set(),
            'average_quality': 0.0
        }
        
        if self.pattern_database:
            qualities = []
            for pattern in self.pattern_database.values():
                summary['subjects'].add(pattern.subject_id)
                
                if pattern.pattern_type not in summary['pattern_types']:
                    summary['pattern_types'][pattern.pattern_type] = 0
                summary['pattern_types'][pattern.pattern_type] += 1
                
                qualities.append(pattern.quality_score)
            
            summary['average_quality'] = np.mean(qualities)
            summary['subjects'] = list(summary['subjects'])
        
        return summary


# Convenience functions for common operations
def extract_motor_patterns(neural_data: np.ndarray, labels: np.ndarray,
                         config: Optional[Config] = None) -> List[BrainPattern]:
    """Convenience function for motor pattern extraction"""
    extractor = PatternExtractor(config)
    return extractor.extract_motor_patterns(neural_data, labels)


def transfer_pattern_between_subjects(source_pattern: BrainPattern,
                                    target_data: np.ndarray,
                                    target_subject_id: str,
                                    config: Optional[Config] = None) -> TransferResult:
    """Convenience function for pattern transfer"""
    engine = TransferLearningEngine(config)
    engine.pattern_database[source_pattern.pattern_id] = source_pattern
    
    return engine.transfer_pattern(
        source_pattern.pattern_id, target_subject_id, target_data
    )
