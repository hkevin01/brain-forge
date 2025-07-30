# Brain-Forge Issue Templates & Customization Guide

## 📋 **Issue Templates Overview**

This document consolidates the comprehensive issue template system for Brain-Forge, designed specifically for neuroscience research and brain-computer interface development.

**Last Updated**: July 30, 2025  
**Template Version**: 1.0

---

## 🎯 **Available Templates**

### **1. 🐛 Bug Report** (`bug_report.yml`)
**Purpose:** Report software bugs, crashes, or unexpected behavior

**Key Features:**
- Comprehensive environment reporting (OS, Python, hardware)
- Neuroscience-specific component categorization
- Severity assessment with research impact context
- Hardware troubleshooting steps
- Minimal reproducible example requirements
- Data context information (MEG/EEG/optical data)

**Best For:** Software crashes, incorrect results, performance degradation, hardware driver issues

### **2. 🚀 Feature Request** (`feature_request.yml`)
**Purpose:** Suggest new features or enhancements

**Key Features:**
- Scientific background and literature references
- Implementation complexity assessment
- User impact estimation
- Technical requirements specification
- Success criteria definition
- Contribution willingness tracking

**Best For:** New analysis methods, hardware support, API enhancements, visualization improvements

### **3. ⚡ Hardware Issue** (`hardware_issue.yml`)
**Purpose:** Report hardware-related problems with neuroscience equipment

**Key Features:**
- Safety verification checklist
- Detailed hardware specifications
- Environmental condition reporting
- Data quality impact assessment
- Vendor communication tracking
- Calibration and maintenance history

**Best For:** OMP helmet issues, sensor problems, data acquisition failures, driver problems

### **4. 📚 Documentation Issue** (`documentation_issue.yml`)
**Purpose:** Report problems or suggest improvements for documentation

**Key Features:**
- Documentation type categorization
- User level targeting (beginner to developer)
- Content suggestions and examples
- Multiple format preferences
- User research context

**Best For:** Missing docs, unclear explanations, outdated information, formatting issues

### **5. 🔒 Security Issue** (`security_issue.yml`)
**Purpose:** Report security vulnerabilities or concerns (use cautiously)

**Key Features:**
- Confidential reporting mechanism
- Severity assessment framework
- Impact evaluation guidelines
- Responsible disclosure process
- Contact information for security team

**Best For:** Security vulnerabilities, data privacy concerns, access control issues

### **6. ⚡ Performance Issue** (`performance_issue.yml`)
**Purpose:** Report performance problems and optimization requests

**Key Features:**
- Performance benchmark comparisons
- Resource usage monitoring
- Scalability assessment
- Hardware configuration details
- Profiling data integration

**Best For:** Slow processing, memory issues, scaling problems, optimization needs

### **7. ❓ Question** (`question.yml`)
**Purpose:** Ask questions and request support

**Key Features:**
- Context gathering for effective support
- Research background information
- Previous troubleshooting attempts
- Community resource references
- Learning objective identification

**Best For:** Usage questions, conceptual clarification, getting started help

---

## 🎨 **Template Customization Guide**

### **File Structure**

```
.github/ISSUE_TEMPLATE/
├── bug_report.yml           # Bug reports
├── feature_request.yml      # Feature requests
├── hardware_issue.yml       # Hardware problems
├── documentation_issue.yml  # Documentation issues
├── security_issue.yml       # Security concerns
├── performance_issue.yml    # Performance problems
├── question.yml            # Questions and support
├── config.yml              # Template configuration
└── custom/                 # Organization-specific templates
    ├── clinical_issue.yml  # Clinical research specific
    └── multi_site.yml      # Multi-site studies
```

### **Template Components**

Each template contains:

```yaml
name: Template Name           # Display name in GitHub UI
description: Brief description # Shown in template chooser
title: "[PREFIX] "           # Auto-generated title prefix
labels: ["label1", "label2"] # Automatic labels
assignees: ["username"]      # Auto-assigned reviewers

body:                        # Form definition
  - type: markdown          # Static content
  - type: checkboxes       # Multiple checkboxes
  - type: input            # Single line input
  - type: textarea         # Multi-line input
  - type: dropdown         # Selection dropdown
```

---

## 🔧 **Customization Options**

### **1. Basic Customization**

#### **Modify Labels**
```yaml
# Change automatic labels
labels: ["bug", "neuroscience", "needs-triage", "your-org"]

# Add project-specific labels
labels: ["meg-analysis", "clinical-study", "multi-site"]
```

#### **Update Assignees**
```yaml
# Assign to different team members
assignees:
  - lead-scientist
  - data-analyst
  - hardware-specialist
```

#### **Custom Title Prefixes**
```yaml
# Modify automatic title generation
title: "[BUG] "              # For bug reports
title: "[FEATURE] "          # For feature requests
title: "[HARDWARE] "         # For hardware issues
```

### **2. Organization-Specific Templates**

#### **Clinical Research Template**
```yaml
name: Clinical Research Issue
description: Issues specific to clinical neuroscience research
labels: ["clinical", "research", "needs-irb-review"]

body:
  - type: dropdown
    id: study_phase
    attributes:
      label: Study Phase
      options:
        - Protocol Development
        - IRB Review
        - Data Collection
        - Analysis
        - Publication
```

#### **Multi-Site Study Template**
```yaml
name: Multi-Site Study Issue
description: Issues affecting multiple research sites
labels: ["multi-site", "coordination", "high-priority"]

body:
  - type: checkboxes
    id: affected_sites
    attributes:
      label: Affected Sites
      options:
        - label: Site A - University Hospital
        - label: Site B - Research Institute
        - label: Site C - Clinical Center
```

### **3. Field Customization**

#### **Hardware-Specific Dropdowns**
```yaml
- type: dropdown
  id: hardware_type
  attributes:
    label: Hardware Component
    options:
      - OMP Helmet (306 channels)
      - Kernel Flow2 Optical
      - Kernel Flux Optical
      - Brown Accelo-hat
      - LSL Synchronization
      - Custom Hardware
```

#### **Neuroscience-Specific Checkboxes**
```yaml
- type: checkboxes
  id: data_modalities
  attributes:
    label: Data Modalities Affected
    options:
      - label: MEG (Magnetoencephalography)
      - label: fNIRS (Functional Near-Infrared Spectroscopy)
      - label: EEG (Electroencephalography)
      - label: Motion/Accelerometer
      - label: Synchronized Multi-modal
```

---

## 🌐 **Multi-Language Support**

### **Template Localization**

#### **Directory Structure**
```
.github/ISSUE_TEMPLATE/
├── en/                     # English templates (default)
├── es/                     # Spanish templates
├── fr/                     # French templates
└── localization.yml        # Language configuration
```

#### **Language-Specific Templates**
```yaml
# Spanish Bug Report Template
name: Reporte de Error
description: Reportar errores de software o comportamiento inesperado
title: "[ERROR] "
labels: ["bug", "spanish", "necesita-triaje"]
```

### **Cultural Customization**

#### **Academic Institution Templates**
```yaml
# University-specific template
name: Academic Research Issue
labels: ["academic", "university", "research-grant"]

body:
  - type: input
    id: grant_number
    attributes:
      label: Grant/Funding Number
      placeholder: "NSF-1234567, NIH-R01-..."
```

---

## 🧪 **Testing and Validation**

### **Template Testing Process**

#### **1. Syntax Validation**
```bash
# Validate YAML syntax
yamllint .github/ISSUE_TEMPLATE/

# Check GitHub template format
gh api repos/:owner/:repo/issues/templates
```

#### **2. User Experience Testing**
- Create test issues using each template
- Verify form functionality and validation
- Test auto-labeling and assignment
- Validate required field enforcement

#### **3. Integration Testing**
- Test with different user permission levels
- Verify template chooser appearance
- Test mobile/tablet responsiveness
- Validate accessibility features

### **Quality Assurance Checklist**

#### **Template Requirements**
- [ ] YAML syntax is valid
- [ ] All required fields are specified
- [ ] Labels are consistently formatted
- [ ] Assignees exist in the repository
- [ ] Description text is clear and helpful
- [ ] Examples are provided where appropriate

#### **User Experience Requirements**
- [ ] Template chooser shows correct information
- [ ] Form fields are logically ordered
- [ ] Required fields are clearly marked
- [ ] Help text is informative
- [ ] Submit button works correctly

---

## 📊 **Maintenance Guidelines**

### **Regular Maintenance Tasks**

#### **Monthly Reviews**
- Review template usage analytics
- Check for user feedback on templates
- Update examples and help text
- Verify assignee availability

#### **Quarterly Updates**
- Review and update label taxonomy
- Assess template effectiveness
- Update field options based on usage
- Sync with project evolution

#### **Annual Assessments**
- Comprehensive template review
- User survey on template effectiveness
- Technology stack updates
- Accessibility compliance review

### **Version Control**

#### **Change Management**
- Document all template changes
- Test changes in development environment
- Gradual rollout of major changes
- Maintain backward compatibility

#### **Release Notes**
- Communicate template updates to users
- Provide migration guides for breaking changes
- Document new features and capabilities
- Share best practices and tips

---

## 📈 **Analytics and Optimization**

### **Usage Metrics**

#### **Template Effectiveness**
- Track template usage frequency
- Monitor completion rates
- Analyze user drop-off points
- Measure issue resolution time

#### **Quality Indicators**
- Issue clarity and completeness
- Triage efficiency improvements
- Developer response time
- User satisfaction scores

### **Continuous Improvement**

#### **Data-Driven Optimization**
- A/B test template variations
- Optimize field ordering and grouping
- Refine help text based on user behavior
- Simplify complex templates

#### **User Feedback Integration**
- Regular user surveys
- Issue template feedback collection
- Community input on improvements
- Developer experience feedback

The Brain-Forge issue template system provides a comprehensive, customizable framework for managing neuroscience research and development issues effectively, supporting the platform's mission as a world-class brain-computer interface system.
