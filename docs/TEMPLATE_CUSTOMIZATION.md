# Customizing Brain-Forge Issue Templates

This guide explains how to customize and maintain the Brain-Forge issue template system for different research environments and organizational needs.

## Table of Contents

- [Template Architecture](#template-architecture)
- [Customization Options](#customization-options)
- [Adding New Templates](#adding-new-templates)
- [Modifying Existing Templates](#modifying-existing-templates)
- [Organization-Specific Customization](#organization-specific-customization)
- [Multi-Language Support](#multi-language-support)
- [Testing and Validation](#testing-and-validation)
- [Maintenance Guidelines](#maintenance-guidelines)

## Template Architecture

### File Structure

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
    ├── regulatory.yml      # Regulatory compliance
    └── multi_site.yml      # Multi-site studies
```

### Template Components

Each template contains:

```yaml
name: Template Name           # Display name in GitHub UI
description: Brief description # Shown in template chooser
title: "[PREFIX] "           # Auto-generated title prefix
labels: ["label1", "label2"] # Automatic labels
assignees: ["username"]      # Auto-assigned reviewers

body:                        # Form definition
  - type: markdown          # Static content
  - type: input            # Single-line text
  - type: textarea         # Multi-line text
  - type: dropdown         # Selection menu
  - type: checkboxes       # Multiple checkboxes
```

## Customization Options

### 1. Basic Customization

#### Modify Labels
```yaml
# Change automatic labels
labels: ["bug", "neuroscience", "needs-triage", "your-org"]

# Add project-specific labels
labels: ["meg-analysis", "clinical-study", "multi-site"]
```

#### Update Assignees
```yaml
# Assign to different team members
assignees:
  - lead-scientist
  - hardware-specialist
  - data-analyst
```

#### Customize Title Prefixes
```yaml
# Use organization-specific prefixes
title: "[YOUR-ORG BUG] "     # Instead of "[BUG] "
title: "[CLINICAL-ISSUE] "   # For clinical studies
title: "[SITE-A] "          # For multi-site studies
```

### 2. Field Customization

#### Add Custom Dropdown Options
```yaml
- type: dropdown
  id: research-area
  attributes:
    label: Research Area
    options:
      - Cognitive Neuroscience
      - Clinical Neurology  
      - Brain-Computer Interface
      - Neuroplasticity
      - Your Custom Area
```

#### Modify Hardware Categories
```yaml
- type: dropdown
  id: hardware-type
  attributes:
    label: Hardware Type
    options:
      - Your Custom MEG System
      - Institutional EEG Setup
      - Custom Accelerometer Array
      - Third-party Integration
```

#### Add Organization Fields
```yaml
- type: input
  id: institution
  attributes:
    label: Institution
    placeholder: University of Example
  validations:
    required: true

- type: dropdown
  id: funding-source
  attributes:
    label: Funding Source
    options:
      - NIH Grant
      - NSF Grant
      - Industry Partnership
      - Internal Funding
```

## Adding New Templates

### 1. Clinical Research Template

Create `.github/ISSUE_TEMPLATE/clinical_issue.yml`:

```yaml
name: Clinical Research Issue
description: Report issues specific to clinical neuroscience research
title: "[CLINICAL] "
labels: ["clinical", "research", "patient-data", "triage"]
assignees:
  - clinical-lead
  - compliance-officer

body:
  - type: markdown
    attributes:
      value: |
        # 🏥 Clinical Research Issue
        
        **HIPAA/Privacy Notice:** Do not include any patient identifiers or PHI in this report.
        
        This template is for issues related to clinical neuroscience research using Brain-Forge.

  - type: checkboxes
    id: privacy-compliance
    attributes:
      label: Privacy Compliance
      options:
        - label: I confirm this report contains no patient identifiers (PHI/PII)
          required: true
        - label: I have IRB approval for the research mentioned in this report
          required: true

  - type: dropdown
    id: clinical-context
    attributes:
      label: Clinical Context
      options:
        - Stroke Recovery Study
        - Epilepsy Monitoring
        - Cognitive Assessment
        - Pre-surgical Mapping
        - Clinical Trial
        - Diagnostic Procedure
        - Other Clinical Research
    validations:
      required: true

  - type: dropdown
    id: patient-population
    attributes:
      label: Patient Population (General)
      options:
        - Adult (18-65)
        - Pediatric (<18)
        - Elderly (>65)
        - Mixed Ages
        - Healthy Controls
    validations:
      required: true

  # ... additional clinical-specific fields
```

### 2. Multi-Site Collaboration Template

Create `.github/ISSUE_TEMPLATE/multi_site.yml`:

```yaml
name: Multi-Site Study Issue
description: Issues related to multi-site collaborative studies
title: "[MULTI-SITE] "
labels: ["multi-site", "collaboration", "data-sharing"]

body:
  - type: input
    id: coordinating-site
    attributes:
      label: Coordinating Site
      placeholder: University of Example (Primary Site)
    validations:
      required: true

  - type: textarea
    id: participating-sites
    attributes:
      label: Participating Sites
      placeholder: |
        - Site A: University of Example (Lead: Dr. Smith)
        - Site B: Research Institute XYZ (Lead: Dr. Jones)
        - Site C: Medical Center ABC (Lead: Dr. Brown)

  - type: dropdown
    id: data-sharing-model
    attributes:
      label: Data Sharing Model
      options:
        - Centralized (all data to coordinating site)
        - Federated (distributed analysis)
        - Hybrid (partial centralization)
        - Site-specific processing
    validations:
      required: true

  # ... additional multi-site fields
```

## Modifying Existing Templates

### 1. Extending Bug Report Template

Add organization-specific sections to `bug_report.yml`:

```yaml
# Add after existing fields
- type: dropdown
  id: internal-system
  attributes:
    label: Internal System Configuration
    options:
      - Production Cluster A
      - Research Lab Setup B
      - Mobile Research Unit
      - Cloud Instance
      - Local Development

- type: input
  id: internal-ticket
  attributes:
    label: Internal Ticket Reference
    placeholder: "HELP-2024-001234"
    description: Reference to internal help desk or tracking system

- type: checkboxes
  id: organizational-requirements
  attributes:
    label: Organizational Requirements
    options:
      - label: Issue affects grant deliverables
      - label: Issue impacts scheduled publication
      - label: Issue affects clinical timeline
      - label: Issue requires vendor communication
```

### 2. Customizing Feature Requests

Modify `feature_request.yml` for specific research focus:

```yaml
# Replace generic options with research-specific ones
- type: dropdown
  id: research-impact
  attributes:
    label: Research Impact Area
    options:
      - Alzheimer's Disease Research
      - Stroke Recovery Studies
      - Brain-Computer Interface Development
      - Pediatric Neurology
      - Computational Neuroscience
      - Your Research Focus Area

- type: textarea
  id: grant-alignment
  attributes:
    label: Grant and Funding Alignment
    description: How does this feature align with funded research goals?
    placeholder: |
      This feature directly supports our NIH R01 grant (1R01NS123456) by enabling:
      - Real-time analysis required for Aim 2
      - Multi-site data harmonization for Aim 3
      - Clinical translation pathway for Aim 4
```

## Organization-Specific Customization

### 1. University/Hospital Environment

```yaml
# Add institutional context
- type: dropdown
  id: institutional-review
  attributes:
    label: Institutional Review Status
    options:
      - IRB Approved
      - IRB Pending
      - IRB Not Required
      - Animal Care Committee Approved
      - No Human/Animal Subjects

- type: input
  id: protocol-number
  attributes:
    label: Protocol Number
    placeholder: "IRB-2024-001234"

- type: dropdown
  id: data-classification
  attributes:
    label: Data Classification Level
    options:
      - Public
      - Internal Use
      - Confidential
      - Restricted (PHI/PII)
```

### 2. Industry/Commercial Environment

```yaml
# Add business context
- type: dropdown
  id: business-priority
  attributes:
    label: Business Priority
    options:
      - Critical (Product Release Blocking)
      - High (Customer Impact)
      - Medium (Internal Efficiency)
      - Low (Future Enhancement)

- type: input
  id: customer-reference
  attributes:
    label: Customer Reference
    placeholder: "CUST-2024-001234"

- type: dropdown
  id: regulatory-impact
  attributes:
    label: Regulatory Impact
    options:
      - FDA Submission Related
      - CE Marking Required
      - Clinical Trial Impact
      - Quality Management System
      - No Regulatory Impact
```

## Multi-Language Support

### 1. Creating Translated Templates

Create language-specific template directories:

```
.github/ISSUE_TEMPLATE/
├── en/                     # English (default)
│   ├── bug_report.yml
│   └── feature_request.yml
├── es/                     # Spanish
│   ├── reporte_error.yml
│   └── solicitud_caracteristica.yml
├── fr/                     # French
│   ├── rapport_bogue.yml
│   └── demande_fonctionnalite.yml
└── de/                     # German
    ├── fehlerbericht.yml
    └── funktionsanfrage.yml
```

### 2. Localized Content Example

Spanish Bug Report (`es/reporte_error.yml`):

```yaml
name: Reporte de Error
description: Reportar errores de software, fallos o comportamiento inesperado
title: "[ERROR] "
labels: ["error", "triaje", "necesita-investigación"]

body:
  - type: markdown
    attributes:
      value: |
        # 🐛 Reporte de Error
        
        ¡Gracias por tomarte el tiempo de completar este reporte de error! 
        Esto nos ayuda a mejorar Brain-Forge para todos.

  - type: textarea
    id: descripcion-error
    attributes:
      label: Descripción del Error
      description: Una descripción clara y concisa de qué es el error
      placeholder: |
        Ejemplo: Al procesar datos MEG con >200 canales, el sistema se bloquea 
        durante el análisis de conectividad...
    validations:
      required: true
```

## Testing and Validation

### 1. Template Validation

Create validation script (`scripts/validate_templates.py`):

```python
#!/usr/bin/env python3
"""Validate GitHub issue templates."""

import yaml
import os
from pathlib import Path

def validate_template(template_path):
    """Validate a single template file."""
    try:
        with open(template_path, 'r') as f:
            template = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['name', 'description', 'body']
        for field in required_fields:
            if field not in template:
                print(f"❌ {template_path}: Missing required field '{field}'")
                return False
        
        # Validate body structure
        for i, item in enumerate(template['body']):
            if 'type' not in item:
                print(f"❌ {template_path}: Body item {i} missing 'type'")
                return False
        
        print(f"✅ {template_path}: Valid")
        return True
        
    except Exception as e:
        print(f"❌ {template_path}: Error - {e}")
        return False

def main():
    """Validate all templates."""
    template_dir = Path('.github/ISSUE_TEMPLATE')
    templates = list(template_dir.glob('*.yml'))
    
    all_valid = True
    for template in templates:
        if template.name != 'config.yml':  # Skip config file
            if not validate_template(template):
                all_valid = False
    
    if all_valid:
        print("\n🎉 All templates are valid!")
    else:
        print("\n💥 Some templates have issues!")
        exit(1)

if __name__ == '__main__':
    main()
```

### 2. Template Testing

Create test scenarios (`tests/test_templates.py`):

```python
import pytest
import yaml
from pathlib import Path

class TestIssueTemplates:
    """Test issue template functionality."""
    
    @pytest.fixture
    def template_dir(self):
        return Path('.github/ISSUE_TEMPLATE')
    
    def test_all_templates_have_required_fields(self, template_dir):
        """Test that all templates have required fields."""
        templates = list(template_dir.glob('*.yml'))
        templates = [t for t in templates if t.name != 'config.yml']
        
        required_fields = ['name', 'description', 'body']
        
        for template_path in templates:
            with open(template_path) as f:
                template = yaml.safe_load(f)
            
            for field in required_fields:
                assert field in template, f"{template_path} missing {field}"
    
    def test_dropdown_options_not_empty(self, template_dir):
        """Test dropdown fields have options."""
        templates = list(template_dir.glob('*.yml'))
        templates = [t for t in templates if t.name != 'config.yml']
        
        for template_path in templates:
            with open(template_path) as f:
                template = yaml.safe_load(f)
            
            for item in template['body']:
                if item.get('type') == 'dropdown':
                    options = item.get('attributes', {}).get('options', [])
                    assert len(options) > 0, f"Dropdown in {template_path} has no options"
```

## Maintenance Guidelines

### 1. Regular Review Schedule

- **Monthly:** Review template usage analytics
- **Quarterly:** Update field options based on common inputs
- **Annually:** Major template restructuring if needed

### 2. Analytics and Optimization

Track template effectiveness:

```python
# Example analytics script
def analyze_template_usage():
    """Analyze which templates are used most frequently."""
    issues = get_github_issues()  # Your GitHub API call
    
    template_usage = {}
    for issue in issues:
        labels = issue.get('labels', [])
        for label in labels:
            if label['name'] in ['bug', 'enhancement', 'hardware']:
                template_usage[label['name']] = template_usage.get(label['name'], 0) + 1
    
    print("Template Usage:")
    for template, count in sorted(template_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {template}: {count} issues")
```

### 3. User Feedback Integration

Collect feedback on template effectiveness:

```yaml
# Add to templates periodically
- type: dropdown
  id: template-feedback
  attributes:
    label: Template Feedback (Optional)
    description: How helpful was this template?
    options:
      - Very helpful - easy to fill out
      - Somewhat helpful - mostly clear
      - Neutral - adequate for my needs
      - Somewhat confusing - some unclear parts
      - Very confusing - difficult to use
```

### 4. Version Control Best Practices

- **Branch protection:** Require reviews for template changes
- **Testing:** Validate templates before merging
- **Documentation:** Update guides when templates change
- **Rollback plan:** Keep previous versions for reference

### 5. Change Management Process

1. **Identify need** for template changes
2. **Design changes** with stakeholder input
3. **Test changes** in development environment
4. **Review changes** with maintainers
5. **Deploy changes** with monitoring
6. **Collect feedback** and iterate

## Best Practices Summary

### Do's ✅
- Keep templates focused and specific
- Use clear, descriptive field labels
- Provide helpful placeholder text
- Include validation where appropriate
- Test templates thoroughly
- Document customizations

### Don'ts ❌
- Make templates too long or complex
- Use ambiguous language
- Forget to update related documentation
- Deploy untested changes
- Ignore user feedback
- Leave outdated options in dropdowns

### Accessibility Considerations
- Use clear, simple language
- Provide adequate context for all fields
- Consider screen reader compatibility
- Test with diverse user groups
- Provide alternative contact methods

---

This customization guide should help you adapt the Brain-Forge issue template system to your specific research environment and organizational needs. Remember to test all changes thoroughly and maintain documentation as you customize the templates.
