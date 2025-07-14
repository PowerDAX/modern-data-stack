# Container Security Best Practices
## Modern Data Stack Showcase - Infrastructure Security Guide

### Table of Contents
1. [Container Image Security](#container-image-security)
2. [Runtime Security](#runtime-security)
3. [Network Security](#network-security)
4. [Data Protection](#data-protection)
5. [Access Control](#access-control)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Vulnerability Management](#vulnerability-management)
8. [Compliance and Governance](#compliance-and-governance)
9. [Security Testing](#security-testing)
10. [Incident Response](#incident-response)

---

## Container Image Security

### 1. Base Image Selection

**Use Official and Minimal Base Images**
```dockerfile
# ✅ GOOD: Use official, minimal base images
FROM python:3.11-slim-bullseye

# ❌ BAD: Avoid large, general-purpose images
FROM ubuntu:latest
```

**Use Specific Version Tags**
```dockerfile
# ✅ GOOD: Use specific version tags
FROM postgres:15.4-alpine3.18

# ❌ BAD: Avoid latest tags in production
FROM postgres:latest
```

### 2. Multi-Stage Builds

**Minimize Attack Surface**
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY app.py .
CMD ["python", "app.py"]
```

### 3. Package Management

**Keep Packages Updated**
```dockerfile
# Update package index and install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        package1 \
        package2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**Remove Unnecessary Packages**
```dockerfile
# Remove package manager after installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        required-package && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*
```

### 4. User Security

**Run as Non-Root User**
```dockerfile
# Create dedicated user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set proper ownership
COPY --chown=appuser:appuser app.py /app/
USER appuser

# ❌ BAD: Don't run as root
# USER root
```

**Use Specific User IDs**
```dockerfile
# Use specific UID/GID for consistency
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser
```

---

## Runtime Security

### 1. Container Runtime Configuration

**Security Options**
```yaml
# Docker Compose security configuration
services:
  app:
    image: myapp:latest
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=1g
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
```

**Resource Limits**
```yaml
services:
  app:
    image: myapp:latest
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### 2. File System Security

**Read-Only Root Filesystem**
```yaml
services:
  app:
    image: myapp:latest
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=100m
      - /var/run:rw,noexec,nosuid,size=50m
```

**Secure Volume Mounts**
```yaml
services:
  app:
    image: myapp:latest
    volumes:
      - app_data:/data:rw,nosuid,nodev,noexec
      - app_config:/config:ro,nosuid,nodev,noexec
```

---

## Network Security

### 1. Network Segmentation

**Separate Networks**
```yaml
networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  backend:
    driver: bridge
    internal: true  # No internet access
    ipam:
      config:
        - subnet: 172.21.0.0/24
```

**Service-Specific Networks**
```yaml
services:
  web:
    networks:
      - frontend
  database:
    networks:
      - backend
  app:
    networks:
      - frontend
      - backend
```

### 2. Port Security

**Minimize Exposed Ports**
```yaml
services:
  app:
    # Only expose necessary ports
    ports:
      - "127.0.0.1:8080:8080"  # Bind to localhost only
    # Don't expose internal ports
    expose:
      - "9000"  # Internal communication only
```

**Use Reverse Proxy**
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - app
  app:
    image: myapp:latest
    # No direct port exposure
    expose:
      - "8080"
```

---

## Data Protection

### 1. Secrets Management

**Use Docker Secrets**
```yaml
services:
  app:
    image: myapp:latest
    secrets:
      - db_password
      - api_key
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - API_KEY_FILE=/run/secrets/api_key

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    external: true
```

**Avoid Environment Variables for Secrets**
```yaml
# ❌ BAD: Don't put secrets in environment variables
environment:
  - DB_PASSWORD=supersecret123

# ✅ GOOD: Use secrets or external secret management
secrets:
  - db_password
environment:
  - DB_PASSWORD_FILE=/run/secrets/db_password
```

### 2. Data Encryption

**Encrypt Data at Rest**
```yaml
services:
  database:
    image: postgres:15
    environment:
      - POSTGRES_INITDB_ARGS=--auth-host=md5
    volumes:
      - encrypted_data:/var/lib/postgresql/data
```

**Encrypt Data in Transit**
```yaml
services:
  app:
    image: myapp:latest
    environment:
      - TLS_CERT_FILE=/run/secrets/tls_cert
      - TLS_KEY_FILE=/run/secrets/tls_key
    secrets:
      - tls_cert
      - tls_key
```

---

## Access Control

### 1. Authentication and Authorization

**Service-to-Service Authentication**
```yaml
services:
  app:
    image: myapp:latest
    environment:
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
      - SERVICE_ACCOUNT_KEY_FILE=/run/secrets/service_account_key
    secrets:
      - jwt_secret
      - service_account_key
```

**Role-Based Access Control**
```dockerfile
# Define application roles
RUN groupadd -r readonly && \
    groupadd -r readwrite && \
    groupadd -r admin && \
    useradd -r -g readonly app_readonly && \
    useradd -r -g readwrite app_readwrite && \
    useradd -r -g admin app_admin
```

### 2. Container Privileges

**Drop Unnecessary Capabilities**
```yaml
services:
  app:
    image: myapp:latest
    cap_drop:
      - ALL
    cap_add:
      - CHOWN      # Only if needed
      - SETGID     # Only if needed
      - SETUID     # Only if needed
```

**Use Security Profiles**
```yaml
services:
  app:
    image: myapp:latest
    security_opt:
      - apparmor:docker-default
      - seccomp:./seccomp-profile.json
```

---

## Monitoring and Logging

### 1. Container Monitoring

**Health Checks**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

**Resource Monitoring**
```yaml
services:
  app:
    image: myapp:latest
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

### 2. Security Logging

**Audit Logging**
```yaml
services:
  app:
    image: myapp:latest
    volumes:
      - audit_logs:/var/log/audit
    environment:
      - AUDIT_LOG_LEVEL=INFO
      - AUDIT_LOG_FILE=/var/log/audit/security.log
```

**Centralized Logging**
```yaml
services:
  app:
    image: myapp:latest
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "app.security"
```

---

## Vulnerability Management

### 1. Image Scanning

**Pre-Deployment Scanning**
```bash
# Scan images before deployment
trivy image myapp:latest

# Scan for specific vulnerabilities
trivy image --severity HIGH,CRITICAL myapp:latest

# Scan with custom policies
trivy image --policy-bundle ./policies myapp:latest
```

**Continuous Scanning**
```yaml
services:
  scanner:
    image: aquasec/trivy:latest
    command: server --listen 0.0.0.0:8080
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - TRIVY_DB_REPOSITORY=ghcr.io/aquasecurity/trivy-db
```

### 2. Runtime Security

**Runtime Monitoring**
```yaml
services:
  falco:
    image: falcosecurity/falco:latest
    privileged: true
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /dev:/host/dev
      - /proc:/host/proc:ro
      - /boot:/host/boot:ro
      - /lib/modules:/host/lib/modules:ro
      - /usr:/host/usr:ro
      - /etc:/host/etc:ro
```

---

## Compliance and Governance

### 1. CIS Docker Benchmark

**Automated Compliance Checking**
```bash
# Run Docker Bench Security
docker run --rm --net host --pid host --userns host --cap-add audit_control \
    -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
    -v /etc:/etc:ro \
    -v /var/lib:/var/lib:ro \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    -v /usr/lib/systemd:/usr/lib/systemd:ro \
    -v /etc/systemd:/etc/systemd:ro \
    --label docker_bench_security \
    docker/docker-bench-security
```

### 2. Security Policies

**Policy as Code**
```yaml
# OPA Gatekeeper policy
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: k8srequiredsecuritycontext
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredSecurityContext
      validation:
        properties:
          runAsNonRoot:
            type: boolean
          readOnlyRootFilesystem:
            type: boolean
```

---

## Security Testing

### 1. Static Analysis

**Dockerfile Linting**
```bash
# Hadolint for Dockerfile best practices
hadolint Dockerfile

# Check for security issues
docker run --rm -i hadolint/hadolint < Dockerfile
```

**Container Structure Testing**
```bash
# Test container structure
container-structure-test test --image myapp:latest --config tests/structure-test.yaml
```

### 2. Dynamic Analysis

**Penetration Testing**
```bash
# Run security tests against running containers
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/docker-bench-security
```

---

## Incident Response

### 1. Security Incident Detection

**Automated Alerting**
```yaml
services:
  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
```

### 2. Incident Response Plan

**Container Isolation**
```bash
# Isolate compromised container
docker network disconnect <network> <container>

# Stop container
docker stop <container>

# Create forensic image
docker commit <container> forensic-image:latest
```

**Log Collection**
```bash
# Collect container logs
docker logs <container> > incident-logs.txt

# Collect system logs
journalctl -u docker.service > docker-system-logs.txt
```

---

## Security Checklist

### Pre-Deployment
- [ ] Base image vulnerability scan completed
- [ ] Dockerfile security review completed
- [ ] Secrets properly managed (not in environment variables)
- [ ] Non-root user configured
- [ ] Unnecessary capabilities dropped
- [ ] Health checks implemented
- [ ] Resource limits configured
- [ ] Network segmentation implemented

### Runtime
- [ ] Container monitoring enabled
- [ ] Security logging configured
- [ ] Runtime security tools deployed
- [ ] Compliance policies enforced
- [ ] Incident response plan activated
- [ ] Regular security assessments scheduled

### Post-Deployment
- [ ] Continuous vulnerability scanning
- [ ] Security metrics monitored
- [ ] Compliance reports generated
- [ ] Security training completed
- [ ] Documentation updated

---

## Tools and Technologies

### Security Scanning
- **Trivy**: Comprehensive vulnerability scanner
- **Clair**: Container vulnerability analysis
- **Anchore**: Container security and compliance
- **Snyk**: Developer security platform

### Runtime Security
- **Falco**: Runtime security monitoring
- **Twistlock**: Container security platform
- **Aqua Security**: Cloud-native security
- **Sysdig**: Container security and monitoring

### Compliance
- **Docker Bench**: CIS Docker Benchmark
- **OPA Gatekeeper**: Policy enforcement
- **Checkov**: Infrastructure as code security
- **Terrascan**: Static code analysis

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **ELK Stack**: Logging and analysis
- **Jaeger**: Distributed tracing

---

## Additional Resources

- [NIST Container Security Guide](https://csrc.nist.gov/publications/detail/sp/800-190/final)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Docker Security](https://owasp.org/www-project-docker-security/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Maintainer**: Data Engineering Team 