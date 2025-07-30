# Brain-Forge Kubernetes Deployment Manifests

This directory contains Kubernetes manifests for deploying Brain-Forge in production environments.

## Files Overview

- `namespace.yaml` - Kubernetes namespace for Brain-Forge
- `configmap.yaml` - Configuration management
- `secrets.yaml` - Secrets management (template)
- `postgres-deployment.yaml` - PostgreSQL database deployment
- `redis-deployment.yaml` - Redis cache deployment
- `brain-forge-deployment.yaml` - Main Brain-Forge API deployment
- `services.yaml` - Kubernetes services
- `ingress.yaml` - Ingress configuration for external access
- `hpa.yaml` - Horizontal Pod Autoscaler
- `networkpolicy.yaml` - Network security policies

## Quick Deploy

```bash
# Create namespace and apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n brain-forge

# Access logs
kubectl logs -f deployment/brain-forge-api -n brain-forge
```

## Production Considerations

1. **Secrets Management**: Replace template secrets with actual values
2. **Persistent Storage**: Configure appropriate storage classes
3. **Resource Limits**: Adjust CPU/memory based on your cluster
4. **Network Policies**: Review and customize network security
5. **Monitoring**: Add Prometheus/Grafana monitoring
6. **SSL/TLS**: Configure proper SSL certificates for ingress

## Environment Variables

Set these in your environment or CI/CD pipeline:

```bash
export BRAIN_FORGE_IMAGE=ghcr.io/your-org/brain-forge:latest
export POSTGRES_PASSWORD=your-secure-password
export REDIS_PASSWORD=your-redis-password
export JWT_SECRET=your-jwt-secret
export ENCRYPTION_KEY=your-encryption-key
```
