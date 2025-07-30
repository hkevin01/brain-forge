#!/bin/bash

# Brain-Forge Production Deployment Script
# This script deploys Brain-Forge to a Kubernetes cluster

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
K8S_DIR="$PROJECT_ROOT/k8s"
NAMESPACE="brain-forge"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Brain-Forge Kubernetes Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy          Deploy Brain-Forge to Kubernetes
    update          Update existing deployment
    rollback        Rollback to previous version
    status          Check deployment status
    logs            Show application logs
    cleanup         Remove all resources
    validate        Validate configuration
    help            Show this help message

Options:
    --namespace     Kubernetes namespace (default: brain-forge)
    --image-tag     Docker image tag (default: latest)
    --dry-run       Show what would be deployed without applying
    --wait          Wait for deployment to complete
    --force         Force deployment even if validation fails

Examples:
    $0 deploy --image-tag v1.0.0 --wait
    $0 update --namespace brain-forge-staging
    $0 status
    $0 logs --follow
    $0 cleanup --force

EOF
}

# Parse command line arguments
COMMAND=""
IMAGE_TAG="latest"
DRY_RUN=false
WAIT=false
FORCE=false
FOLLOW_LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|update|rollback|status|logs|cleanup|validate|help)
            COMMAND="$1"
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --wait)
            WAIT=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --follow)
            FOLLOW_LOGS=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show help if no command provided
if [[ -z "$COMMAND" ]]; then
    show_help
    exit 0
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if k8s directory exists
    if [[ ! -d "$K8S_DIR" ]]; then
        log_error "Kubernetes manifests directory not found: $K8S_DIR"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate configuration
validate_config() {
    log_info "Validating configuration..."
    
    local errors=0
    
    # Check required environment variables
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "JWT_SECRET_KEY"
        "ENCRYPTION_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_warning "Environment variable $var is not set"
            ((errors++))
        fi
    done
    
    # Validate Kubernetes manifests
    for manifest in "$K8S_DIR"/*.yaml; do
        if [[ -f "$manifest" ]]; then
            if ! kubectl apply --dry-run=client -f "$manifest" &> /dev/null; then
                log_error "Invalid Kubernetes manifest: $manifest"
                ((errors++))
            fi
        fi
    done
    
    if [[ $errors -gt 0 && "$FORCE" != true ]]; then
        log_error "Configuration validation failed with $errors errors"
        log_info "Use --force to deploy anyway or fix the errors"
        exit 1
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_warning "Configuration validation found $errors errors, but continuing due to --force"
    else
        log_success "Configuration validation passed"
    fi
}

# Create namespace if it doesn't exist
ensure_namespace() {
    log_info "Ensuring namespace $NAMESPACE exists..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        log_info "Creating namespace $NAMESPACE..."
        kubectl apply -f "$K8S_DIR/namespace.yaml"
        log_success "Namespace $NAMESPACE created"
    fi
}

# Generate secrets if they don't exist
generate_secrets() {
    log_info "Generating secrets..."
    
    if kubectl get secret brain-forge-secrets -n "$NAMESPACE" &> /dev/null; then
        log_info "Secrets already exist"
        return
    fi
    
    log_info "Creating new secrets..."
    
    # Generate secure random values
    local db_password=$(openssl rand -base64 32)
    local redis_password=$(openssl rand -base64 32)
    local jwt_secret=$(openssl rand -base64 64)
    local encryption_key=$(openssl rand -base64 32)
    
    # Create secret
    kubectl create secret generic brain-forge-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=DATABASE_USER="brain_forge_user" \
        --from-literal=DATABASE_PASSWORD="$db_password" \
        --from-literal=DATABASE_URL="postgresql://brain_forge_user:$db_password@postgres-service:5432/brain_forge" \
        --from-literal=REDIS_PASSWORD="$redis_password" \
        --from-literal=REDIS_URL="redis://:$redis_password@redis-service:6379" \
        --from-literal=JWT_SECRET_KEY="$jwt_secret" \
        --from-literal=ENCRYPTION_KEY="$encryption_key" \
        --from-literal=JWT_ALGORITHM="HS256"
    
    log_success "Secrets created successfully"
}

# Deploy function
deploy() {
    log_info "Starting Brain-Forge deployment..."
    
    check_prerequisites
    validate_config
    ensure_namespace
    generate_secrets
    
    # Apply manifests in order
    local manifests=(
        "namespace.yaml"
        "configmap.yaml"
        "secrets.yaml"
        "postgres-deployment.yaml"
        "redis-deployment.yaml"
        "services.yaml"
        "brain-forge-deployment.yaml"
        "hpa.yaml"
        "networkpolicy.yaml"
        "ingress.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        local manifest_path="$K8S_DIR/$manifest"
        
        if [[ -f "$manifest_path" ]]; then
            log_info "Applying $manifest..."
            
            if [[ "$DRY_RUN" == true ]]; then
                kubectl apply --dry-run=client -f "$manifest_path"
            else
                kubectl apply -f "$manifest_path"
            fi
        else
            log_warning "Manifest not found: $manifest_path"
        fi
    done
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run completed - no resources were actually created"
        return
    fi
    
    log_success "Deployment manifests applied"
    
    if [[ "$WAIT" == true ]]; then
        log_info "Waiting for deployment to complete..."
        kubectl rollout status deployment/brain-forge-api -n "$NAMESPACE" --timeout=600s
        kubectl rollout status deployment/postgres -n "$NAMESPACE" --timeout=300s
        kubectl rollout status deployment/redis -n "$NAMESPACE" --timeout=300s
    fi
    
    log_success "Brain-Forge deployment completed"
    
    # Show status
    show_status
}

# Update function
update() {
    log_info "Updating Brain-Forge deployment..."
    
    # Update image tag in deployment
    kubectl set image deployment/brain-forge-api \
        brain-forge-api="ghcr.io/brain-forge/brain-forge:$IMAGE_TAG" \
        -n "$NAMESPACE"
    
    if [[ "$WAIT" == true ]]; then
        kubectl rollout status deployment/brain-forge-api -n "$NAMESPACE" --timeout=600s
    fi
    
    log_success "Update completed"
}

# Rollback function
rollback() {
    log_info "Rolling back Brain-Forge deployment..."
    
    kubectl rollout undo deployment/brain-forge-api -n "$NAMESPACE"
    
    if [[ "$WAIT" == true ]]; then
        kubectl rollout status deployment/brain-forge-api -n "$NAMESPACE" --timeout=600s
    fi
    
    log_success "Rollback completed"
}

# Status function
show_status() {
    log_info "Brain-Forge deployment status:"
    
    echo
    log_info "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo
    log_info "Services:"
    kubectl get services -n "$NAMESPACE"
    
    echo
    log_info "Ingress:"
    kubectl get ingress -n "$NAMESPACE"
    
    echo
    log_info "Persistent Volume Claims:"
    kubectl get pvc -n "$NAMESPACE"
    
    echo
    log_info "Horizontal Pod Autoscalers:"
    kubectl get hpa -n "$NAMESPACE"
}

# Logs function
show_logs() {
    if [[ "$FOLLOW_LOGS" == true ]]; then
        kubectl logs -f deployment/brain-forge-api -n "$NAMESPACE"
    else
        kubectl logs deployment/brain-forge-api -n "$NAMESPACE" --tail=100
    fi
}

# Cleanup function
cleanup() {
    if [[ "$FORCE" != true ]]; then
        log_warning "This will delete all Brain-Forge resources in namespace $NAMESPACE"
        read -p "Are you sure? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Cleanup cancelled"
            exit 0
        fi
    fi
    
    log_info "Cleaning up Brain-Forge resources..."
    
    # Delete all resources in the namespace
    kubectl delete all --all -n "$NAMESPACE"
    kubectl delete pvc --all -n "$NAMESPACE"
    kubectl delete secrets --all -n "$NAMESPACE"
    kubectl delete configmaps --all -n "$NAMESPACE"
    kubectl delete ingress --all -n "$NAMESPACE"
    kubectl delete networkpolicies --all -n "$NAMESPACE"
    
    # Delete namespace
    kubectl delete namespace "$NAMESPACE"
    
    log_success "Cleanup completed"
}

# Main command dispatcher
case "$COMMAND" in
    deploy)
        deploy
        ;;
    update)
        update
        ;;
    rollback)
        rollback
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup
        ;;
    validate)
        check_prerequisites
        validate_config
        log_success "Validation completed"
        ;;
    help)
        show_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
