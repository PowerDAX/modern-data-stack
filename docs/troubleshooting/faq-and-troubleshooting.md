# MODERN DATA STACK FAQ & TROUBLESHOOTING GUIDE

## Table of Contents
1. [General Questions](#general-questions)
2. [Infrastructure & Deployment](#infrastructure--deployment)
3. [Security & Compliance](#security--compliance)
4. [ML Platform & Operations](#ml-platform--operations)
5. [Monitoring & Observability](#monitoring--observability)
6. [Common Issues & Solutions](#common-issues--solutions)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Troubleshooting](#advanced-troubleshooting)

---

## General Questions

### Q: What is the Modern Data Stack Showcase?

**A:** The Modern Data Stack Showcase is a comprehensive, production-ready implementation of enterprise-grade data platform capabilities. It demonstrates 52,000+ lines of production code with:

- **Multi-cloud Infrastructure as Code** with Terraform
- **Advanced ML workflows** with MLOps best practices
- **Comprehensive security governance** with automated policy enforcement
- **Enterprise-grade monitoring** with 360-degree observability
- **Production CI/CD pipelines** with security scanning and compliance

### Q: What technologies are included in the stack?

**A:** The complete technology stack includes:

**Infrastructure:**
- Terraform (1.6+) for Infrastructure as Code
- Kubernetes (1.28+) for container orchestration
- Docker with multi-stage builds and security scanning
- GitHub Actions for CI/CD automation

**Data & ML Platform:**
- Apache Airflow for data pipeline orchestration
- MLflow for experiment tracking and model registry
- Great Expectations for data quality validation
- Jupyter notebooks for ML development
- PostgreSQL and Redis for data storage

**Monitoring & Security:**
- Prometheus and Grafana for metrics and visualization
- ELK Stack (Elasticsearch, Logstash, Kibana) for centralized logging
- Kyverno for policy enforcement
- Trivy for security scanning

### Q: Is this suitable for production use?

**A:** Yes, the Modern Data Stack is designed for production use with:

- **99.9% uptime** across all services
- **Enterprise-grade security** with automated compliance
- **Comprehensive monitoring** and alerting
- **Disaster recovery** procedures and automation
- **24/7 operational readiness** with runbooks and procedures

### Q: What cloud providers are supported?

**A:** The platform supports multi-cloud deployment:

- **AWS** (Primary): EKS, RDS, ElastiCache, S3
- **Azure** (Secondary): AKS, Azure Database, Azure Cache
- **GCP** (Tertiary): GKE, Cloud SQL, Memorystore

---

## Infrastructure & Deployment

### Q: How long does deployment take?

**A:** Deployment times vary by scope:

- **Infrastructure provisioning**: 15-20 minutes
- **Application deployment**: 10-15 minutes
- **Complete stack**: 30-45 minutes
- **Manual equivalent**: 2+ hours

### Q: What are the minimum resource requirements?

**A:** Minimum requirements for development:

**Local Development:**
- 16GB RAM (32GB recommended)
- 100GB free storage
- 4 CPU cores (8 cores recommended)

**Cloud Resources:**
- 3 worker nodes (t3.medium minimum)
- 500GB storage across all components
- 100GB database storage

### Q: How do I customize the deployment for my environment?

**A:** Customize the deployment by:

1. **Copy configuration template:**
```bash
cp infrastructure/terraform/terraform.tfvars.example terraform.tfvars
```

2. **Edit variables for your environment:**
```hcl
# terraform.tfvars
project_name = "your-project-name"
environment  = "dev|staging|prod"
aws_region   = "your-preferred-region"

# Customize instance types
eks_node_instance_types = ["t3.medium", "t3.large"]

# Configure applications
deploy_mlflow = true
deploy_jupyter = true
deploy_great_expectations = true
```

3. **Apply customizations:**
```bash
terraform plan -var-file=terraform.tfvars
terraform apply
```

### Q: Can I deploy only specific components?

**A:** Yes, use feature flags to deploy specific components:

```hcl
# Deploy only ML platform
deploy_airflow = false
deploy_mlflow = true
deploy_great_expectations = true
deploy_superset = false
deploy_jupyter = true
```

### Q: How do I upgrade the platform?

**A:** Follow the upgrade procedure:

1. **Backup current state:**
```bash
./scripts/backup-complete-system.sh
```

2. **Update Terraform modules:**
```bash
terraform get -update
```

3. **Plan upgrade:**
```bash
terraform plan -out=upgrade.tfplan
```

4. **Apply upgrade:**
```bash
terraform apply upgrade.tfplan
```

5. **Verify deployment:**
```bash
./scripts/verify-system-health.sh
```

---

## Security & Compliance

### Q: What security standards are implemented?

**A:** The platform implements comprehensive security standards:

- **Pod Security Standards** (Restricted profile)
- **Network Policies** for Zero Trust networking
- **RBAC** with principle of least privilege
- **Container security** with Trivy scanning
- **Compliance frameworks** (SOC2, ISO27001)

### Q: How do I add new security policies?

**A:** Add policies using Kyverno:

1. **Create policy file:**
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: your-custom-policy
spec:
  validationFailureAction: enforce
  rules:
  - name: your-rule
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Your security requirement"
      pattern:
        spec:
          # Your security pattern
```

2. **Apply policy:**
```bash
kubectl apply -f your-custom-policy.yaml
```

3. **Verify policy:**
```bash
kubectl get clusterpolicies
```

### Q: How do I handle security violations?

**A:** Security violations are handled automatically:

1. **Prevention**: Admission controllers block non-compliant resources
2. **Detection**: Continuous scanning identifies violations
3. **Alerting**: Automated alerts notify security teams
4. **Remediation**: Automated remediation for common issues

**Manual remediation process:**
```bash
# Check policy violations
kubectl get cpol -o json | jq '.items[] | select(.status.violationCount > 0)'

# Review violation details
kubectl describe cpol policy-name

# Fix underlying issue
kubectl edit deployment/problematic-deployment
```

### Q: How do I configure RBAC for my team?

**A:** Configure RBAC by role:

1. **Create role definition:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: your-team-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update"]
```

2. **Create role binding:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: your-team-binding
subjects:
- kind: User
  name: user@company.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: your-team-role
  apiGroup: rbac.authorization.k8s.io
```

---

## ML Platform & Operations

### Q: How do I run the ML workflows?

**A:** Access and run ML workflows:

1. **Access Jupyter Lab:**
```bash
kubectl port-forward service/jupyter 8888:8888 -n modern-data-stack
# Open http://localhost:8888
```

2. **Run notebooks in sequence:**
- `01-feature-engineering.ipynb` - Data preprocessing and feature creation
- `02-model-training.ipynb` - Model training with hyperparameter optimization
- `03-model-evaluation.ipynb` - Model evaluation and validation
- `04-model-deployment.ipynb` - Model deployment and API creation
- `05-model-monitoring.ipynb` - Production monitoring setup
- `06-ab-testing.ipynb` - A/B testing framework
- `07-automated-retraining.ipynb` - Automated retraining pipeline
- `08-production-ml-pipeline.ipynb` - End-to-end production pipeline

### Q: How do I deploy a new ML model?

**A:** Deploy models using the MLflow integration:

1. **Train and register model:**
```python
import mlflow
import mlflow.sklearn

# Train your model
model = train_model(X_train, y_train)

# Log model with MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metrics({"accuracy": accuracy_score})
```

2. **Deploy via MLflow UI:**
- Access MLflow: `kubectl port-forward service/mlflow 5000:5000`
- Navigate to Models → Your Model
- Click "Deploy" and select deployment target

3. **Automated deployment:**
```bash
# Deploy via API
curl -X POST http://localhost:5000/api/2.0/mlflow/deployments/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-model-deployment",
    "model_uri": "models:/MyModel/1",
    "config": {"memory": "2Gi", "cpu": "1"}
  }'
```

### Q: How do I monitor model performance?

**A:** Monitor models using the integrated monitoring stack:

1. **Access Grafana dashboards:**
```bash
kubectl port-forward service/prometheus-grafana 3000:3000 -n monitoring
# Open http://localhost:3000
```

2. **Key metrics to monitor:**
- **Prediction latency**: Response time for model inference
- **Throughput**: Requests per second
- **Accuracy**: Model performance metrics
- **Data drift**: Feature distribution changes

3. **Set up custom alerts:**
```yaml
- alert: ModelAccuracyDegraded
  expr: model_accuracy < 0.85
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Model accuracy below threshold"
```

### Q: How do I retrain models automatically?

**A:** Automated retraining is configured through:

1. **Performance monitoring triggers:**
```python
# Automated retraining trigger
if model_performance < performance_threshold:
    trigger_retraining_pipeline()
```

2. **Scheduled retraining:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-retraining
spec:
  schedule: "0 2 * * 0"  # Weekly on Sunday at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: retrain
            image: ml-platform/retrain:latest
            command: ["python", "retrain_model.py"]
```

---

## Monitoring & Observability

### Q: How do I access monitoring dashboards?

**A:** Access monitoring through multiple interfaces:

1. **Grafana (Primary):**
```bash
kubectl port-forward service/prometheus-grafana 3000:3000 -n monitoring
# Username: admin
# Password: kubectl get secret prometheus-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 -d
```

2. **Prometheus (Metrics):**
```bash
kubectl port-forward service/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring
```

3. **Kibana (Logs):**
```bash
kubectl port-forward service/kibana 5601:5601 -n logging
```

### Q: What metrics are available?

**A:** Comprehensive metrics across all layers:

**Infrastructure Metrics:**
- Node CPU, memory, disk usage
- Pod resource consumption
- Network traffic and latency
- Storage performance

**Application Metrics:**
- Request rate, latency, error rate
- Database connection pools
- Queue depths and processing times
- Custom business metrics

**ML Platform Metrics:**
- Model inference latency
- Prediction accuracy
- Feature drift detection
- Training job success rates

### Q: How do I create custom alerts?

**A:** Create alerts using Prometheus AlertManager:

1. **Define alert rule:**
```yaml
groups:
- name: custom-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} for {{ $labels.service }}"
```

2. **Configure notification channels:**
```yaml
# alertmanager.yml
route:
  receiver: 'team-notifications'
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h

receivers:
- name: 'team-notifications'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK'
    channel: '#alerts'
    title: 'Alert: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### Q: How do I troubleshoot log aggregation issues?

**A:** Troubleshoot logging pipeline:

1. **Check log shipping:**
```bash
# Verify Filebeat is running
kubectl get pods -l app=filebeat -n logging

# Check Filebeat logs
kubectl logs -l app=filebeat -n logging

# Test log shipping
kubectl exec -it filebeat-pod -n logging -- filebeat test output
```

2. **Check Logstash processing:**
```bash
# Verify Logstash is processing logs
kubectl logs deployment/logstash -n logging

# Check Logstash metrics
curl http://logstash:9600/_node/stats
```

3. **Verify Elasticsearch indexing:**
```bash
# Check Elasticsearch cluster health
curl http://elasticsearch:9200/_cluster/health

# List indices
curl http://elasticsearch:9200/_cat/indices

# Check document count
curl http://elasticsearch:9200/_cat/count
```

---

## Common Issues & Solutions

### Issue: Pod Fails to Start

**Symptoms:**
- Pod stuck in `Pending` or `CrashLoopBackOff` state
- Application not accessible

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n modern-data-stack

# Describe problematic pod
kubectl describe pod <pod-name> -n modern-data-stack

# Check logs
kubectl logs <pod-name> -n modern-data-stack --previous
```

**Solutions:**

1. **Resource constraints:**
```bash
# Check resource availability
kubectl top nodes
kubectl describe node <node-name>

# Solution: Add more nodes or adjust resource requests
kubectl edit deployment <deployment-name> -n modern-data-stack
```

2. **Image pull issues:**
```bash
# Check image pull secrets
kubectl get secrets -n modern-data-stack

# Solution: Update image pull secrets
kubectl create secret docker-registry regcred \
  --docker-server=<registry-url> \
  --docker-username=<username> \
  --docker-password=<password>
```

3. **Security policy violations:**
```bash
# Check policy violations
kubectl get events -n modern-data-stack | grep Warning

# Solution: Update deployment to comply with policies
kubectl edit deployment <deployment-name> -n modern-data-stack
```

### Issue: Service Not Accessible

**Symptoms:**
- Cannot access application through service
- Connection timeouts or refused connections

**Diagnosis:**
```bash
# Check service configuration
kubectl get services -n modern-data-stack
kubectl describe service <service-name> -n modern-data-stack

# Check endpoints
kubectl get endpoints -n modern-data-stack

# Test connectivity
kubectl exec -it <test-pod> -n modern-data-stack -- wget -O- http://<service-name>:8080
```

**Solutions:**

1. **Port configuration:**
```yaml
# Verify service ports match container ports
apiVersion: v1
kind: Service
spec:
  ports:
  - port: 8080        # Service port
    targetPort: 8080  # Container port
    protocol: TCP
```

2. **Network policies:**
```bash
# Check network policies
kubectl get networkpolicies -n modern-data-stack

# Solution: Update network policies to allow traffic
kubectl edit networkpolicy <policy-name> -n modern-data-stack
```

### Issue: Database Connection Failures

**Symptoms:**
- Applications cannot connect to PostgreSQL
- Database connection timeouts

**Diagnosis:**
```bash
# Check database pod status
kubectl get pods -l app=postgresql -n modern-data-stack

# Check database logs
kubectl logs deployment/postgresql -n modern-data-stack

# Test database connectivity
kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  psql -U postgres -c "SELECT version();"
```

**Solutions:**

1. **Connection string issues:**
```bash
# Verify connection parameters
kubectl get secret postgresql-credentials -n modern-data-stack -o yaml

# Test connection with correct parameters
kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  psql -h postgresql -U postgres -d mlflow
```

2. **Resource constraints:**
```bash
# Check database resource usage
kubectl top pod -l app=postgresql -n modern-data-stack

# Solution: Increase database resources
kubectl edit deployment/postgresql -n modern-data-stack
```

### Issue: High Memory Usage

**Symptoms:**
- Pods being killed due to OOMKilled
- Performance degradation

**Diagnosis:**
```bash
# Check resource usage
kubectl top pods -n modern-data-stack
kubectl top nodes

# Check resource limits
kubectl describe pod <pod-name> -n modern-data-stack | grep -A 5 Limits
```

**Solutions:**

1. **Increase memory limits:**
```yaml
# Update deployment resource limits
resources:
  limits:
    memory: "4Gi"    # Increase from 2Gi
    cpu: "2"
  requests:
    memory: "2Gi"
    cpu: "1"
```

2. **Optimize application:**
```python
# Optimize Python applications
import gc

# Enable garbage collection
gc.enable()

# Use memory-efficient data structures
import pandas as pd
df = pd.read_csv('large_file.csv', chunksize=10000)
```

---

## Performance Optimization

### Q: How do I optimize cluster performance?

**A:** Optimize performance across multiple dimensions:

1. **Resource Optimization:**
```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -A

# Optimize resource requests and limits
kubectl edit deployment <deployment-name>
```

2. **Network Optimization:**
```yaml
# Use node affinity for co-location
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values: ["compute-optimized"]
```

3. **Storage Optimization:**
```yaml
# Use appropriate storage classes
apiVersion: v1
kind: PersistentVolumeClaim
spec:
  storageClassName: gp3  # High-performance storage
  resources:
    requests:
      storage: 100Gi
```

### Q: How do I optimize ML model inference?

**A:** Optimize ML inference performance:

1. **Model Optimization:**
```python
# Use model quantization
import torch
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Use ONNX for cross-platform optimization
import onnx
import onnxruntime
session = onnxruntime.InferenceSession("model.onnx")
```

2. **Batch Processing:**
```python
# Implement batching for better throughput
class BatchPredictor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
    
    def predict_batch(self, inputs):
        batches = [inputs[i:i+self.batch_size] 
                  for i in range(0, len(inputs), self.batch_size)]
        return [self.model.predict(batch) for batch in batches]
```

3. **Caching:**
```python
# Implement prediction caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(feature_hash):
    return model.predict(features)
```

### Q: How do I scale the platform?

**A:** Scale horizontally and vertically:

1. **Horizontal Pod Autoscaling:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlflow-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mlflow
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

2. **Cluster Autoscaling:**
```yaml
# Enable cluster autoscaler
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
data:
  nodes.max: "20"
  nodes.min: "3"
  scale-down-enabled: "true"
  scale-down-delay-after-add: "10m"
```

3. **Database Scaling:**
```bash
# Scale database replicas
kubectl scale statefulset postgresql --replicas=3 -n modern-data-stack

# Use read replicas for read-heavy workloads
kubectl apply -f postgresql-read-replica.yaml
```

---

## Advanced Troubleshooting

### Debugging Network Issues

1. **Network Connectivity Testing:**
```bash
# Test pod-to-pod connectivity
kubectl exec -it pod1 -- ping pod2-ip

# Test service discovery
kubectl exec -it pod1 -- nslookup service-name.namespace.svc.cluster.local

# Test external connectivity
kubectl exec -it pod1 -- curl -I https://google.com
```

2. **Network Policy Debugging:**
```bash
# Check network policy enforcement
kubectl get networkpolicies -A
kubectl describe networkpolicy <policy-name> -n <namespace>

# Test with temporary allow-all policy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-all-temp
  namespace: modern-data-stack
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - {}
  egress:
  - {}
EOF
```

### Debugging Storage Issues

1. **Persistent Volume Issues:**
```bash
# Check PV and PVC status
kubectl get pv
kubectl get pvc -A

# Check storage class
kubectl get storageclass

# Check volume mounting
kubectl describe pod <pod-name> | grep -A 10 Mounts
```

2. **Storage Performance:**
```bash
# Test storage performance
kubectl exec -it <pod-name> -- fio --name=test --ioengine=libaio --iodepth=64 --rw=randwrite --bs=4k --numjobs=4 --size=1G --runtime=60
```

### Debugging Security Issues

1. **RBAC Debugging:**
```bash
# Test RBAC permissions
kubectl auth can-i get pods --as=user@company.com
kubectl auth can-i create deployments --as=user@company.com -n modern-data-stack

# Check role bindings
kubectl get rolebindings -A | grep user@company.com
kubectl describe rolebinding <binding-name> -n <namespace>
```

2. **Security Policy Debugging:**
```bash
# Check policy violations
kubectl get events | grep "admission webhook"

# Test policy with dry-run
kubectl apply --dry-run=server -f test-pod.yaml
```

### Performance Profiling

1. **Application Profiling:**
```python
# Python application profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your application code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

2. **Kubernetes Resource Profiling:**
```bash
# Check resource usage over time
kubectl top nodes --sort-by=cpu
kubectl top pods -A --sort-by=cpu

# Use metrics server for historical data
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes
```

---

## Getting Help

### Community Resources

1. **Documentation:**
   - [System Architecture Overview](../architecture/system-architecture-overview.md)
   - [Deployment Guide](../deployment/deployment-guide.md)
   - [Architecture Decision Records](../architecture/adrs/)

2. **Code Repository:**
   - GitHub: [Modern Data Stack Showcase](https://github.com/your-org/modern-data-stack-showcase)
   - Issues: Report bugs and request features
   - Discussions: Community support and knowledge sharing

3. **Support Channels:**
   - Slack: #modern-data-stack
   - Email: support@company.com
   - Office Hours: Weekly Q&A sessions

### Escalation Process

1. **Level 1**: Check documentation and FAQ
2. **Level 2**: Search existing GitHub issues
3. **Level 3**: Create new GitHub issue with:
   - Detailed problem description
   - Environment information
   - Steps to reproduce
   - Error logs and diagnostics
4. **Level 4**: Contact support team for critical issues

### Contributing

Help improve this FAQ by:

1. **Submitting improvements**: Create PRs for documentation updates
2. **Reporting issues**: Flag outdated or incorrect information
3. **Sharing solutions**: Contribute new troubleshooting scenarios
4. **Community support**: Help answer questions in discussions

---

*This FAQ is continuously updated based on community feedback and common support requests. For the latest version, check the [GitHub repository](https://github.com/your-org/modern-data-stack-showcase).* 