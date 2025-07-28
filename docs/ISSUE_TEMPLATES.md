# Brain-Forge Issue Templates

This document describes the comprehensive issue template system for Brain-Forge, designed specifically for neuroscience research and brain-computer interface development.

## Available Templates

### 1. 🐛 Bug Report (`bug_report.yml`)
**Purpose:** Report software bugs, crashes, or unexpected behavior

**Key Features:**
- Comprehensive environment reporting (OS, Python, hardware)
- Neuroscience-specific component categorization
- Severity assessment with research impact context
- Hardware troubleshooting steps
- Minimal reproducible example requirements
- Data context information (MEG/EEG/optical data)

**Best For:** Software crashes, incorrect results, performance degradation, hardware driver issues

### 2. 🚀 Feature Request (`feature_request.yml`)
**Purpose:** Suggest new features or enhancements

**Key Features:**
- Scientific background and literature references
- Implementation complexity assessment
- User impact estimation
- Technical requirements specification
- Success criteria definition
- Contribution willingness tracking

**Best For:** New analysis methods, hardware support, API enhancements, visualization improvements

### 3. ⚡ Hardware Issue (`hardware_issue.yml`)
**Purpose:** Report hardware-related problems with neuroscience equipment

**Key Features:**
- Safety verification checklist
- Detailed hardware specifications
- Environmental condition reporting
- Data quality impact assessment
- Vendor communication tracking
- Calibration and maintenance history

**Best For:** OMP helmet issues, sensor problems, data acquisition failures, driver problems

### 4. 📚 Documentation Issue (`documentation_issue.yml`)
**Purpose:** Report problems or suggest improvements for documentation

**Key Features:**
- Documentation type categorization
- User level targeting (beginner to developer)
- Content suggestions and examples
- Multiple format preferences
- User research context

**Best For:** Missing docs, unclear explanations, outdated information, formatting issues

### 5. 🔒 Security Issue (`security_issue.yml`)
**Purpose:** Report security vulnerabilities or concerns (use cautiously)

**Key Features:**
- Confidentiality assessment
- Medical data privacy considerations
- HIPAA/GDPR compliance implications
- Responsible disclosure commitment
- Research data protection focus

**Best For:** Authentication issues, data protection concerns, access control problems
**Note:** Critical vulnerabilities should be reported via email to security@brain-forge.org

### 6. ⚡ Performance Issue (`performance_issue.yml`)
**Purpose:** Report performance problems, bottlenecks, or optimization opportunities

**Key Features:**
- Real-time processing context (latency requirements)
- Comprehensive performance metrics
- Hardware utilization assessment
- Scalability analysis
- Profiling data integration

**Best For:** Slow processing, memory leaks, real-time deadline misses, scalability problems

### 7. ❓ Question / Support (`question.yml`)
**Purpose:** Ask questions, get help, or start discussions

**Key Features:**
- User experience level assessment
- Research context categorization
- Specific question breakdown
- Community collaboration options
- Follow-up preference specification

**Best For:** Usage questions, best practice guidance, method discussions, integration help

## Template Selection Guide

### Quick Decision Tree

```
Are you experiencing unexpected behavior? → 🐛 Bug Report
├─ Software crash/error → 🐛 Bug Report  
├─ Hardware malfunction → ⚡ Hardware Issue
└─ Performance problem → ⚡ Performance Issue

Do you want something new or different? → 🚀 Feature Request
├─ New functionality → 🚀 Feature Request
├─ Better documentation → 📚 Documentation Issue  
└─ Security improvement → 🔒 Security Issue

Do you need help or have questions? → ❓ Question / Support
├─ How to use Brain-Forge → ❓ Question / Support
├─ Scientific methods → ❓ Question / Support
└─ Best practices → ❓ Question / Support
```

### By Component

| Component | Bug Report | Feature Request | Hardware Issue | Performance |
|-----------|------------|----------------|----------------|-------------|
| Data Acquisition | ✅ | ✅ | ✅ | ✅ |
| Signal Processing | ✅ | ✅ | ❌ | ✅ |
| Brain Mapping | ✅ | ✅ | ❌ | ✅ |
| Hardware Interface | ✅ | ✅ | ✅ | ✅ |
| Visualization | ✅ | ✅ | ❌ | ✅ |
| API/WebSocket | ✅ | ✅ | ❌ | ✅ |

### By User Type

| User Type | Primary Templates | Secondary Templates |
|-----------|------------------|-------------------|
| **Researcher (Beginner)** | ❓ Question, 📚 Documentation | 🐛 Bug Report |
| **Researcher (Experienced)** | 🐛 Bug Report, 🚀 Feature Request | ⚡ Performance, ❓ Question |
| **Hardware Technician** | ⚡ Hardware Issue, 🐛 Bug Report | ⚡ Performance |
| **Developer/Contributor** | 🐛 Bug Report, 🚀 Feature Request | 🔒 Security, ⚡ Performance |
| **System Administrator** | 🔒 Security Issue, ⚡ Performance | 📚 Documentation |

## Template Configuration

### config.yml Settings

The `config.yml` file configures the issue template chooser:

- **Blank issues disabled:** Users must choose a template to ensure structured reporting
- **External links:** Direct users to community resources before creating issues
- **Template ordering:** Templates appear in priority order for typical workflows

### Customization Options

Each template includes:
- **Smart defaults:** Pre-filled values for common scenarios
- **Conditional fields:** Fields that appear based on previous selections
- **Validation rules:** Required fields and format checking
- **Auto-labeling:** Automatic application of relevant labels
- **Auto-assignment:** Automatic assignment to appropriate maintainers

## Best Practices for Users

### Before Creating an Issue

1. **Search existing issues** for similar problems or requests
2. **Check documentation** and FAQ for common solutions
3. **Test with latest version** to ensure issue still exists
4. **Gather information** before starting the form

### Writing Effective Issues

1. **Be specific and detailed** - more information is better
2. **Include context** - explain your research scenario
3. **Provide examples** - code samples, data characteristics
4. **Be patient** - complex issues may take time to resolve
5. **Stay engaged** - respond to questions from maintainers

### Neuroscience-Specific Tips

1. **Include data characteristics** - sampling rate, channels, duration
2. **Specify hardware details** - exact models, firmware versions
3. **Describe research context** - experimental design, requirements
4. **Consider real-time constraints** - latency requirements, deadlines
5. **Mention compliance needs** - HIPAA, IRB, institutional requirements

## For Maintainers

### Template Maintenance

- **Regular review:** Update templates based on common issues
- **Field optimization:** Add/remove fields based on usage patterns
- **Label management:** Ensure labels match actual project workflow
- **Assignment rules:** Update auto-assignment based on team changes

### Processing Guidelines

- **Triage quickly:** Use template structure for fast initial assessment
- **Request information:** Use template fields to identify missing details
- **Cross-reference:** Link related issues identified through template tags
- **Track metrics:** Monitor template usage to improve issue handling

## Integration with Workflows

### Automated Processing

Templates enable automated:
- **Label application** based on component and issue type
- **Project board assignment** for issue tracking
- **Notification routing** to appropriate team members
- **Metrics collection** for issue analysis

### Research Impact Tracking

Templates collect information for:
- **Research impact assessment** - how issues affect ongoing studies
- **Priority determination** - research deadline-driven prioritization
- **Resource allocation** - hardware vs. software issue distribution
- **Community needs** - feature request trend analysis

## Support Resources

- **Community Discussions:** https://github.com/hkevin01/brain-forge/discussions
- **Documentation:** https://brain-forge.readthedocs.io
- **Tutorials:** https://brain-forge.readthedocs.io/en/latest/tutorials
- **FAQ:** https://brain-forge.readthedocs.io/en/latest/faq.html

---

*This template system is designed to support the Brain-Forge community in advancing neuroscience research through better issue tracking and resolution.*
