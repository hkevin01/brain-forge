# Interactive Data Acquisition Tutorial - README

## Overview

The **Interactive Data Acquisition Tutorial** (`01_Interactive_Data_Acquisition.ipynb`) provides a hands-on, step-by-step introduction to Brain-Forge's data acquisition capabilities through an interactive Jupyter notebook. This tutorial is designed for researchers, clinicians, and developers who want to understand and experiment with Brain-Forge's multi-modal brain data collection system.

## Purpose

- **Interactive Learning**: Hands-on exploration of Brain-Forge data acquisition
- **Step-by-Step Guidance**: Progressive tutorial from basic to advanced concepts
- **Practical Examples**: Real code examples with immediate feedback
- **Visual Learning**: Interactive plots and visualizations throughout
- **Experimental Playground**: Safe environment to test and modify parameters

## Tutorial Structure

### Section 1: Introduction to Brain-Forge Data Acquisition
- Overview of multi-modal brain imaging (OPM, EEG, fMRI)
- Brain-Forge architecture and capabilities
- Hardware requirements and setup considerations
- Clinical applications and use cases

### Section 2: Basic Data Acquisition Setup
- Importing Brain-Forge libraries and dependencies
- Setting up mock data sources for tutorial purposes
- Configuring acquisition parameters (sampling rates, channels)
- Understanding data formats and structures

### Section 3: Single-Modal Data Acquisition
- OPM magnetometer data collection simulation
- EEG electrode data acquisition examples
- fMRI BOLD signal integration basics
- Data quality assessment and validation

### Section 4: Multi-Modal Synchronization
- Time synchronization across imaging modalities
- Cross-modal data alignment techniques
- Handling timing discrepancies and drift
- Validation of synchronization accuracy

### Section 5: Real-Time Processing Pipeline
- Setting up real-time data processing
- Implementing sliding window analysis
- Real-time quality monitoring and alerts
- Performance optimization techniques

### Section 6: Data Streaming and Visualization
- WebSocket streaming implementation
- Real-time data visualization dashboards
- Multi-client data distribution
- Interactive plotting and monitoring

### Section 7: Clinical Integration Examples
- EHR system integration patterns
- Clinical workflow integration
- Patient data management and privacy
- Regulatory compliance considerations

### Section 8: Advanced Features and Customization
- Custom acquisition protocols
- Advanced signal processing integration
- API development for external systems
- Performance tuning and optimization

## Running the Tutorial

### Prerequisites
```bash
# Install Brain-Forge with all dependencies
pip install -e .

# Install Jupyter and visualization libraries
pip install jupyter matplotlib plotly ipywidgets
pip install seaborn pandas numpy scipy

# Start Jupyter notebook server
jupyter notebook
```

### Opening the Tutorial
1. Navigate to `examples/jupyter_notebooks/`
2. Open `01_Interactive_Data_Acquisition.ipynb`
3. Run cells sequentially for best learning experience
4. Experiment with parameters and code modifications

### Expected Runtime
**~45 minutes** - Complete tutorial with hands-on exercises

## Learning Objectives

### Technical Learning Outcomes
1. **Data Acquisition Fundamentals**: Understand multi-modal brain data collection principles
2. **System Architecture**: Learn Brain-Forge architecture and component interactions
3. **Synchronization Techniques**: Master time synchronization across imaging modalities
4. **Real-Time Systems**: Implement low-latency data processing pipelines
5. **Quality Assurance**: Develop skills in data quality monitoring and validation

### Practical Skills Development
1. **Code Implementation**: Write Brain-Forge data acquisition code from scratch
2. **Parameter Optimization**: Tune acquisition parameters for different applications
3. **Troubleshooting**: Identify and resolve common acquisition issues
4. **Integration Patterns**: Implement clinical and research workflow integration
5. **Performance Analysis**: Assess and optimize system performance

### Clinical Application Understanding
1. **Hospital Workflows**: Understand clinical data acquisition requirements
2. **Patient Safety**: Learn patient monitoring and safety protocols
3. **Regulatory Compliance**: Understand medical device data requirements
4. **Quality Standards**: Implement clinical-grade quality assurance
5. **Workflow Integration**: Design systems that fit clinical practice

## Tutorial Highlights

### Interactive Code Examples

#### Setting Up Multi-Modal Acquisition
```python
# Interactive example from the tutorial
from brain_forge.acquisition import MultiModalAcquisition
from brain_forge.visualization import RealTimePlotter

# Create acquisition system with user-configurable parameters
acquisition = MultiModalAcquisition(
    omp_channels=306,      # Adjustable in interactive widget
    eeg_channels=64,       # Adjustable in interactive widget
    sampling_rate=1000,    # Adjustable in interactive widget
    buffer_size=1024       # Adjustable in interactive widget
)

# Interactive parameter adjustment widgets
import ipywidgets as widgets
from IPython.display import display

# Create interactive controls
omp_slider = widgets.IntSlider(value=306, min=64, max=306, description='OMP Channels:')
eeg_slider = widgets.IntSlider(value=64, min=32, max=128, description='EEG Channels:')
rate_slider = widgets.IntSlider(value=1000, min=250, max=2000, description='Sample Rate:')

# Real-time parameter updates
def update_acquisition(change):
    acquisition.update_parameters(
        omp_channels=omp_slider.value,
        eeg_channels=eeg_slider.value,
        sampling_rate=rate_slider.value
    )
    print(f"Updated: OMP={omp_slider.value}, EEG={eeg_slider.value}, Rate={rate_slider.value}Hz")

omp_slider.observe(update_acquisition, names='value')
eeg_slider.observe(update_acquisition, names='value')
rate_slider.observe(update_acquisition, names='value')

display(omp_slider, eeg_slider, rate_slider)
```

#### Real-Time Data Visualization
```python
# Interactive real-time plotting example
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Create interactive real-time plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Real-time data buffers
omp_buffer = np.zeros(1000)
eeg_buffer = np.zeros(1000)
spectrum_buffer = np.zeros(500)
connectivity_buffer = np.zeros((68, 68))

# Animation function for real-time updates
def animate(frame):
    # Simulate new data arrival
    new_omp_data = acquisition.get_omp_data(duration=0.1)
    new_eeg_data = acquisition.get_eeg_data(duration=0.1)
    
    # Update buffers
    omp_buffer[:-100] = omp_buffer[100:]
    omp_buffer[-100:] = new_omp_data[0, -100:]  # First OMP channel
    
    eeg_buffer[:-100] = eeg_buffer[100:]
    eeg_buffer[-100:] = new_eeg_data[0, -100:]  # First EEG channel
    
    # Update plots
    axes[0].clear()
    axes[0].plot(omp_buffer)
    axes[0].set_title('Real-Time OMP Signal')
    axes[0].set_ylabel('Magnetic Field (fT)')
    
    axes[1].clear()
    axes[1].plot(eeg_buffer)
    axes[1].set_title('Real-Time EEG Signal')
    axes[1].set_ylabel('Voltage (μV)')
    
    # Spectral analysis
    freqs, psd = welch(omp_buffer, fs=1000)
    axes[2].clear()
    axes[2].semilogy(freqs, psd)
    axes[2].set_title('Power Spectral Density')
    axes[2].set_xlabel('Frequency (Hz)')
    
    # Connectivity matrix
    connectivity = np.corrcoef(new_omp_data)[:68, :68]  # First 68 channels
    axes[3].clear()
    axes[3].imshow(connectivity, cmap='coolwarm', vmin=-1, vmax=1)
    axes[3].set_title('Functional Connectivity')
    
    plt.tight_layout()

# Start real-time animation
anim = FuncAnimation(fig, animate, interval=100, blit=False)
plt.show()
```

### Interactive Exercises

#### Exercise 1: Parameter Optimization
Students adjust acquisition parameters and observe effects on:
- Data quality metrics
- Processing latency
- Memory usage
- Signal-to-noise ratio

#### Exercise 2: Synchronization Challenge
Students implement multi-modal synchronization and validate:
- Timing accuracy across modalities
- Drift correction algorithms
- Cross-modal correlation analysis
- Synchronization error quantification

#### Exercise 3: Quality Monitoring System
Students build a real-time quality monitoring system:
- Automatic artifact detection
- Signal quality alerting
- Performance metric tracking
- Dashboard visualization

## Interactive Features

### Jupyter Widgets Integration
- **Parameter Sliders**: Adjust acquisition parameters in real-time
- **Toggle Switches**: Enable/disable different processing stages
- **Dropdown Menus**: Select different acquisition modes and protocols
- **Text Inputs**: Configure custom acquisition parameters
- **Progress Bars**: Monitor data acquisition and processing progress

### Real-Time Visualizations
- **Live Signal Plots**: Streaming time-series data visualization
- **Spectral Analysis**: Real-time frequency domain analysis
- **Connectivity Maps**: Dynamic functional connectivity visualization
- **Quality Metrics**: Live signal quality and system performance indicators
- **3D Brain Visualization**: Interactive 3D brain activity rendering

### Educational Assessments
- **Knowledge Checks**: Interactive quizzes throughout the tutorial
- **Coding Challenges**: Hands-on programming exercises
- **Troubleshooting Scenarios**: Simulated problem-solving exercises
- **Performance Benchmarks**: Students compare their implementations
- **Case Studies**: Real-world application scenarios

## Tutorial Sections Detail

### Section 1: Foundation Concepts (10 minutes)
**Learning Goals**: Understand Brain-Forge ecosystem and multi-modal brain imaging
**Interactive Elements**: 
- Clickable brain anatomy diagrams
- Comparison tables of imaging modalities
- Interactive timeline of brain imaging history
- Video demonstrations of data acquisition setups

**Key Concepts Covered**:
- Principles of OPM magnetometry
- EEG electrode placement and recording
- fMRI BOLD signal physics
- Multi-modal integration advantages
- Clinical applications overview

### Section 2: System Setup and Configuration (8 minutes)
**Learning Goals**: Configure Brain-Forge for data acquisition
**Interactive Elements**:
- Configuration wizards with immediate feedback
- Parameter validation with error messages
- Hardware compatibility checking tools
- Installation verification scripts

**Hands-On Activities**:
- Import Brain-Forge libraries
- Configure mock hardware interfaces
- Set up data directories and logging
- Test system connectivity and performance

### Section 3: Single-Modal Data Collection (12 minutes)
**Learning Goals**: Master individual modality data acquisition
**Interactive Elements**:
- Live signal generation and visualization
- Parameter adjustment with immediate visual feedback
- Quality metrics dashboard
- Artifact injection and detection exercises

**Progressive Exercises**:
1. Basic OPM data collection with quality monitoring
2. EEG electrode impedance checking and data acquisition
3. fMRI sequence parameter optimization
4. Cross-modal data format comparison

### Section 4: Multi-Modal Integration (10 minutes)
**Learning Goals**: Synchronize multiple brain imaging modalities
**Interactive Elements**:
- Synchronization accuracy visualization
- Timing drift simulation and correction
- Cross-modal correlation analysis tools
- Interactive synchronization troubleshooting

**Advanced Concepts**:
- Hardware clock synchronization
- Software-based time alignment
- Cross-modal validation techniques
- Handling timing discrepancies

### Section 5: Real-Time Processing Implementation (15 minutes)
**Learning Goals**: Build real-time data processing pipelines
**Interactive Elements**:
- Pipeline performance visualization
- Latency measurement tools
- Processing stage timing analysis
- Real-time parameter optimization

**Complex Exercises**:
- Implement sliding window analysis
- Build real-time artifact detection
- Create performance monitoring dashboards
- Optimize processing for clinical latency requirements

## Testing and Validation

### Tutorial Completion Assessment
```python
# Interactive assessment embedded in notebook
class TutorialAssessment:
    def __init__(self):
        self.score = 0
        self.max_score = 100
        
    def check_acquisition_setup(self, student_code):
        """Validate student's acquisition setup implementation"""
        try:
            # Execute student code
            exec(student_code)
            
            # Check for required components
            checks = [
                'acquisition' in locals(),
                hasattr(acquisition, 'omp_channels'),
                hasattr(acquisition, 'eeg_channels'),
                acquisition.omp_channels == 306,
                acquisition.eeg_channels == 64
            ]
            
            score = sum(checks) * 4  # 20 points total
            self.score += score
            
            return {
                'score': score,
                'max_score': 20,
                'feedback': self.generate_feedback(checks),
                'passed': all(checks)
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 20,
                'feedback': f"Code execution error: {str(e)}",
                'passed': False
            }
    
    def check_synchronization_accuracy(self, sync_result):
        """Assess synchronization implementation accuracy"""
        accuracy_thresholds = [
            (sync_result['omp_eeg_sync'] < 1.0, 15),  # <1ms sync
            (sync_result['omp_fmri_sync'] < 2.0, 10), # <2ms sync
            (sync_result['drift_correction'], 10),     # Drift correction
            (sync_result['validation_passed'], 15)     # Validation tests
        ]
        
        score = sum(points for check, points in accuracy_thresholds if check)
        self.score += score
        
        return {
            'score': score,
            'max_score': 50,
            'details': accuracy_thresholds,
            'passed': score >= 40
        }
    
    def final_assessment(self):
        """Provide final tutorial completion assessment"""
        percentage = (self.score / self.max_score) * 100
        
        if percentage >= 90:
            grade = "Excellent"
            message = "Outstanding mastery of Brain-Forge data acquisition!"
        elif percentage >= 80:
            grade = "Good"
            message = "Good understanding with room for improvement."
        elif percentage >= 70:
            grade = "Fair"
            message = "Basic understanding achieved, consider reviewing key concepts."
        else:
            grade = "Needs Improvement"
            message = "Please review tutorial sections and try again."
            
        return {
            'score': self.score,
            'max_score': self.max_score,
            'percentage': percentage,
            'grade': grade,
            'message': message
        }

# Interactive assessment widget
assessment = TutorialAssessment()
assessment_widget = widgets.VBox([
    widgets.HTML("<h3>Tutorial Assessment</h3>"),
    widgets.HTML("<p>Complete the exercises below to test your understanding:</p>"),
    # Assessment questions and code input areas would be added here
])
display(assessment_widget)
```

### Knowledge Validation Exercises
1. **Implementation Challenges**: Students implement key acquisition functions
2. **Parameter Optimization**: Find optimal settings for different scenarios  
3. **Troubleshooting**: Diagnose and fix simulated acquisition problems
4. **Performance Analysis**: Measure and optimize system performance
5. **Integration Testing**: Validate multi-modal integration accuracy

## Troubleshooting Guide

### Common Tutorial Issues

1. **Jupyter Notebook Won't Start**
   ```bash
   # Solution: Check Jupyter installation and port conflicts
   pip install --upgrade jupyter
   jupyter notebook --port=8889  # Try different port
   ```

2. **Interactive Widgets Not Displaying**
   ```bash
   # Solution: Enable Jupyter widgets extension
   pip install ipywidgets
   jupyter nbextension enable --py widgetsnbextension
   ```

3. **Real-Time Plots Not Updating**
   ```python
   # Solution: Enable matplotlib interactive backend
   %matplotlib notebook  # Use in Jupyter cell
   # Or for newer versions:
   %matplotlib widget
   ```

4. **Memory Issues with Large Datasets**
   ```python
   # Solution: Implement data streaming and buffering
   # Use smaller buffer sizes and data chunking
   buffer_size = 1024  # Reduce from default
   ```

### Tutorial Support Resources
- **Discussion Forum**: Community support for tutorial questions
- **Video Walkthroughs**: Recorded demonstrations of key concepts
- **Troubleshooting FAQ**: Common issues and solutions
- **Live Office Hours**: Weekly live Q&A sessions
- **Peer Learning Groups**: Student collaboration opportunities

## Success Criteria

### ✅ Tutorial Success If:
- Student completes all 8 sections with >80% assessment scores
- All interactive exercises execute without errors
- Real-time visualizations display correctly
- Multi-modal synchronization achieves <2ms accuracy
- Student demonstrates understanding through final project

### ⚠️ Review Needed If:
- Assessment scores 70-80%
- Some interactive elements not functioning
- Synchronization accuracy 2-5ms
- Minor understanding gaps in complex topics

### ❌ Tutorial Incomplete If:
- Assessment scores <70%
- Major technical issues preventing completion
- Cannot achieve basic synchronization
- Fundamental misunderstanding of key concepts

## Extensions and Advanced Topics

### Optional Advanced Exercises
1. **Custom Acquisition Protocols**: Design application-specific acquisition protocols
2. **Hardware Interface Development**: Implement interfaces for new hardware devices
3. **Advanced Signal Processing**: Integrate custom processing algorithms
4. **Cloud Integration**: Deploy acquisition system to cloud infrastructure
5. **Mobile Applications**: Develop mobile interfaces for data acquisition

### Research Project Ideas
1. **Multi-Site Deployment Study**: Compare acquisition performance across sites
2. **Novel Synchronization Methods**: Develop improved synchronization techniques
3. **Real-Time Analysis Optimization**: Optimize processing for ultra-low latency
4. **Clinical Workflow Integration**: Design hospital-specific acquisition workflows
5. **Quality Assurance Automation**: Develop automated quality monitoring systems

## Continuing Education

### Next Steps After Tutorial
1. **Advanced Processing Tutorial**: `02_Incremental_Development_Strategy.ipynb`
2. **Clinical Application Demos**: Explore clinical-specific examples
3. **API Development Workshop**: Build custom Brain-Forge applications
4. **Research Collaboration**: Join Brain-Forge research community
5. **Certification Program**: Pursue Brain-Forge operator certification

### Community Resources
- **Brain-Forge User Forum**: Community discussion and support
- **Monthly Webinars**: Advanced topics and new feature demonstrations
- **Research Publications**: Latest Brain-Forge research and applications
- **Conference Presentations**: Present tutorial-based research projects
- **Mentorship Program**: Connect with experienced Brain-Forge developers

---

## Summary

The **Interactive Data Acquisition Tutorial** provides a comprehensive, hands-on introduction to Brain-Forge's data acquisition capabilities through:

- **✅ Progressive Learning**: 8 structured sections building from basic to advanced concepts
- **✅ Interactive Elements**: Jupyter widgets, real-time visualizations, and hands-on exercises
- **✅ Practical Skills**: Real code implementation with immediate feedback and validation
- **✅ Assessment Integration**: Built-in knowledge checks and performance validation
- **✅ Clinical Context**: Hospital workflow integration and regulatory compliance understanding

**Educational Impact**: The tutorial transforms complex multi-modal brain data acquisition into an accessible, interactive learning experience that prepares students for real-world Brain-Forge deployment.

**Next Recommended Tutorial**: Progress to `02_Incremental_Development_Strategy.ipynb` to learn strategic Brain-Forge development and deployment approaches.
