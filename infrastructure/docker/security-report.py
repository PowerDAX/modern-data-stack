#!/usr/bin/env python3
"""
Security Report Generator
Processes security scan results and generates comprehensive reports
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityReportGenerator:
    """Security report generator and analyzer"""
    
    def __init__(self, input_dir: str, output_file: str, scan_id: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.scan_id = scan_id
        self.timestamp = datetime.now().isoformat()
        
        self.severity_weights = {
            'CRITICAL': 10,
            'HIGH': 7,
            'MEDIUM': 4,
            'LOW': 1,
            'INFO': 0.5
        }
        
        self.compliance_scores = {
            'cis': 0,
            'nist': 0,
            'pci-dss': 0
        }
        
        self.report_data = {
            'scan_id': scan_id,
            'timestamp': self.timestamp,
            'scanner_version': '1.0.0',
            'scan_type': 'comprehensive',
            'results': {
                'vulnerability_scan': None,
                'misconfiguration_scan': None,
                'secrets_scan': None,
                'docker_bench': None,
                'compliance_checks': None
            },
            'summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'high_issues': 0,
                'medium_issues': 0,
                'low_issues': 0,
                'info_issues': 0,
                'overall_score': 0,
                'vulnerability_score': 0,
                'configuration_score': 0,
                'compliance_score': 0,
                'security_posture': 'UNKNOWN'
            },
            'recommendations': [],
            'artifacts': [],
            'detailed_findings': []
        }
    
    def process_vulnerability_scan(self, scan_file: Path) -> Dict[str, Any]:
        """Process Trivy vulnerability scan results"""
        try:
            with open(scan_file, 'r') as f:
                data = json.load(f)
            
            vulnerabilities = []
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for result in data.get('Results', []):
                for vuln in result.get('Vulnerabilities', []):
                    severity = vuln.get('Severity', 'UNKNOWN')
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                    
                    vulnerabilities.append({
                        'id': vuln.get('VulnerabilityID', 'N/A'),
                        'severity': severity,
                        'title': vuln.get('Title', 'N/A'),
                        'description': vuln.get('Description', 'N/A'),
                        'package': vuln.get('PkgName', 'N/A'),
                        'installed_version': vuln.get('InstalledVersion', 'N/A'),
                        'fixed_version': vuln.get('FixedVersion', 'N/A'),
                        'references': vuln.get('References', [])
                    })
            
            vulnerability_score = self._calculate_vulnerability_score(severity_counts)
            
            return {
                'total_vulnerabilities': len(vulnerabilities),
                'severity_breakdown': severity_counts,
                'vulnerability_score': vulnerability_score,
                'vulnerabilities': vulnerabilities[:50]  # Limit for report size
            }
            
        except Exception as e:
            logger.error(f"Error processing vulnerability scan: {e}")
            return {'error': str(e)}
    
    def process_misconfiguration_scan(self, scan_file: Path) -> Dict[str, Any]:
        """Process Trivy misconfiguration scan results"""
        try:
            with open(scan_file, 'r') as f:
                data = json.load(f)
            
            misconfigurations = []
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for result in data.get('Results', []):
                for misconfig in result.get('Misconfigurations', []):
                    severity = misconfig.get('Severity', 'UNKNOWN')
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                    
                    misconfigurations.append({
                        'id': misconfig.get('ID', 'N/A'),
                        'severity': severity,
                        'title': misconfig.get('Title', 'N/A'),
                        'description': misconfig.get('Description', 'N/A'),
                        'resolution': misconfig.get('Resolution', 'N/A'),
                        'references': misconfig.get('References', [])
                    })
            
            config_score = self._calculate_configuration_score(severity_counts)
            
            return {
                'total_misconfigurations': len(misconfigurations),
                'severity_breakdown': severity_counts,
                'configuration_score': config_score,
                'misconfigurations': misconfigurations[:50]  # Limit for report size
            }
            
        except Exception as e:
            logger.error(f"Error processing misconfiguration scan: {e}")
            return {'error': str(e)}
    
    def process_secrets_scan(self, scan_file: Path) -> Dict[str, Any]:
        """Process Trivy secrets scan results"""
        try:
            with open(scan_file, 'r') as f:
                data = json.load(f)
            
            secrets = []
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for result in data.get('Results', []):
                for secret in result.get('Secrets', []):
                    severity = secret.get('Severity', 'HIGH')  # Default to HIGH for secrets
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                    
                    secrets.append({
                        'rule_id': secret.get('RuleID', 'N/A'),
                        'severity': severity,
                        'title': secret.get('Title', 'N/A'),
                        'category': secret.get('Category', 'N/A'),
                        'file': secret.get('File', 'N/A'),
                        'line': secret.get('StartLine', 'N/A'),
                        'match': secret.get('Match', 'N/A')[:100]  # Truncate for security
                    })
            
            return {
                'total_secrets': len(secrets),
                'severity_breakdown': severity_counts,
                'secrets': secrets
            }
            
        except Exception as e:
            logger.error(f"Error processing secrets scan: {e}")
            return {'error': str(e)}
    
    def process_docker_bench(self, scan_file: Path) -> Dict[str, Any]:
        """Process Docker Bench Security results"""
        try:
            with open(scan_file, 'r') as f:
                data = json.load(f)
            
            checks = []
            status_counts = {'PASS': 0, 'WARN': 0, 'INFO': 0, 'NOTE': 0}
            
            for check in data.get('tests', []):
                status = check.get('result', 'UNKNOWN')
                if status in status_counts:
                    status_counts[status] += 1
                
                checks.append({
                    'id': check.get('id', 'N/A'),
                    'desc': check.get('desc', 'N/A'),
                    'result': status,
                    'details': check.get('details', 'N/A')
                })
            
            compliance_score = self._calculate_docker_bench_score(status_counts)
            
            return {
                'total_checks': len(checks),
                'status_breakdown': status_counts,
                'compliance_score': compliance_score,
                'checks': checks[:100]  # Limit for report size
            }
            
        except Exception as e:
            logger.error(f"Error processing Docker Bench results: {e}")
            return {'error': str(e)}
    
    def _calculate_vulnerability_score(self, severity_counts: Dict[str, int]) -> float:
        """Calculate vulnerability score based on severity distribution"""
        total_weight = sum(count * self.severity_weights[severity] 
                          for severity, count in severity_counts.items())
        
        if total_weight == 0:
            return 100.0
        
        # Score decreases with more severe vulnerabilities
        max_expected_weight = 50  # Expected maximum weight
        score = max(0, 100 - (total_weight / max_expected_weight) * 100)
        
        return round(score, 2)
    
    def _calculate_configuration_score(self, severity_counts: Dict[str, int]) -> float:
        """Calculate configuration score based on misconfigurations"""
        total_weight = sum(count * self.severity_weights[severity] 
                          for severity, count in severity_counts.items())
        
        if total_weight == 0:
            return 100.0
        
        # Score decreases with more severe misconfigurations
        max_expected_weight = 30  # Expected maximum weight
        score = max(0, 100 - (total_weight / max_expected_weight) * 100)
        
        return round(score, 2)
    
    def _calculate_docker_bench_score(self, status_counts: Dict[str, int]) -> float:
        """Calculate Docker Bench compliance score"""
        total_checks = sum(status_counts.values())
        if total_checks == 0:
            return 0.0
        
        # Weight different statuses
        weighted_score = (
            status_counts.get('PASS', 0) * 1.0 +
            status_counts.get('INFO', 0) * 0.8 +
            status_counts.get('NOTE', 0) * 0.6 +
            status_counts.get('WARN', 0) * 0.2
        )
        
        score = (weighted_score / total_checks) * 100
        return round(score, 2)
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall security score"""
        vuln_score = self.report_data['summary']['vulnerability_score']
        config_score = self.report_data['summary']['configuration_score']
        compliance_score = self.report_data['summary']['compliance_score']
        
        # Weighted average
        overall_score = (
            vuln_score * 0.4 +
            config_score * 0.3 +
            compliance_score * 0.3
        )
        
        return round(overall_score, 2)
    
    def _determine_security_posture(self, overall_score: float) -> str:
        """Determine security posture based on overall score"""
        if overall_score >= 90:
            return 'EXCELLENT'
        elif overall_score >= 80:
            return 'GOOD'
        elif overall_score >= 70:
            return 'ACCEPTABLE'
        elif overall_score >= 60:
            return 'POOR'
        else:
            return 'CRITICAL'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        # Vulnerability recommendations
        if self.report_data['summary']['critical_issues'] > 0:
            recommendations.append("URGENT: Address critical vulnerabilities immediately")
        
        if self.report_data['summary']['high_issues'] > 5:
            recommendations.append("Review and patch high-severity vulnerabilities")
        
        # Configuration recommendations
        if self.report_data['summary']['configuration_score'] < 70:
            recommendations.append("Review and fix container misconfigurations")
        
        # Compliance recommendations
        if self.report_data['summary']['compliance_score'] < 80:
            recommendations.append("Improve compliance with security benchmarks")
        
        # Secrets recommendations
        if self.report_data['results'].get('secrets_scan', {}).get('total_secrets', 0) > 0:
            recommendations.append("CRITICAL: Remove hardcoded secrets and implement secret management")
        
        # General recommendations
        if self.report_data['summary']['overall_score'] < 70:
            recommendations.extend([
                "Implement regular security scanning in CI/CD pipeline",
                "Establish security policies and procedures",
                "Conduct regular security training for development team"
            ])
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        logger.info(f"Generating security report for scan ID: {self.scan_id}")
        
        # Process individual scan results
        scan_files = {
            'vulnerability': list(self.input_dir.glob('vulnerability-*.json')),
            'misconfiguration': list(self.input_dir.glob('misconfig-*.json')),
            'secrets': list(self.input_dir.glob('secrets-*.json')),
            'docker_bench': list(self.input_dir.glob('docker-bench-*.json'))
        }
        
        # Process vulnerability scan
        if scan_files['vulnerability']:
            latest_vuln_file = max(scan_files['vulnerability'], key=os.path.getctime)
            self.report_data['results']['vulnerability_scan'] = self.process_vulnerability_scan(latest_vuln_file)
            
            vuln_data = self.report_data['results']['vulnerability_scan']
            if 'severity_breakdown' in vuln_data:
                self.report_data['summary']['critical_issues'] += vuln_data['severity_breakdown'].get('CRITICAL', 0)
                self.report_data['summary']['high_issues'] += vuln_data['severity_breakdown'].get('HIGH', 0)
                self.report_data['summary']['medium_issues'] += vuln_data['severity_breakdown'].get('MEDIUM', 0)
                self.report_data['summary']['low_issues'] += vuln_data['severity_breakdown'].get('LOW', 0)
                self.report_data['summary']['vulnerability_score'] = vuln_data.get('vulnerability_score', 0)
        
        # Process misconfiguration scan
        if scan_files['misconfiguration']:
            latest_misconfig_file = max(scan_files['misconfiguration'], key=os.path.getctime)
            self.report_data['results']['misconfiguration_scan'] = self.process_misconfiguration_scan(latest_misconfig_file)
            
            misconfig_data = self.report_data['results']['misconfiguration_scan']
            if 'severity_breakdown' in misconfig_data:
                self.report_data['summary']['critical_issues'] += misconfig_data['severity_breakdown'].get('CRITICAL', 0)
                self.report_data['summary']['high_issues'] += misconfig_data['severity_breakdown'].get('HIGH', 0)
                self.report_data['summary']['medium_issues'] += misconfig_data['severity_breakdown'].get('MEDIUM', 0)
                self.report_data['summary']['low_issues'] += misconfig_data['severity_breakdown'].get('LOW', 0)
                self.report_data['summary']['configuration_score'] = misconfig_data.get('configuration_score', 0)
        
        # Process secrets scan
        if scan_files['secrets']:
            latest_secrets_file = max(scan_files['secrets'], key=os.path.getctime)
            self.report_data['results']['secrets_scan'] = self.process_secrets_scan(latest_secrets_file)
            
            secrets_data = self.report_data['results']['secrets_scan']
            if 'severity_breakdown' in secrets_data:
                self.report_data['summary']['critical_issues'] += secrets_data['severity_breakdown'].get('CRITICAL', 0)
                self.report_data['summary']['high_issues'] += secrets_data['severity_breakdown'].get('HIGH', 0)
        
        # Process Docker Bench results
        if scan_files['docker_bench']:
            latest_bench_file = max(scan_files['docker_bench'], key=os.path.getctime)
            self.report_data['results']['docker_bench'] = self.process_docker_bench(latest_bench_file)
            
            bench_data = self.report_data['results']['docker_bench']
            if 'compliance_score' in bench_data:
                self.report_data['summary']['compliance_score'] = bench_data['compliance_score']
        
        # Calculate totals
        self.report_data['summary']['total_issues'] = (
            self.report_data['summary']['critical_issues'] +
            self.report_data['summary']['high_issues'] +
            self.report_data['summary']['medium_issues'] +
            self.report_data['summary']['low_issues']
        )
        
        # Calculate overall score
        self.report_data['summary']['overall_score'] = self._calculate_overall_score()
        self.report_data['summary']['security_posture'] = self._determine_security_posture(
            self.report_data['summary']['overall_score']
        )
        
        # Generate recommendations
        self.report_data['recommendations'] = self._generate_recommendations()
        
        # List artifacts
        self.report_data['artifacts'] = [str(f) for f in self.input_dir.glob('*.json')]
        
        return self.report_data
    
    def save_report(self, report_data: Dict[str, Any]):
        """Save the generated report to file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Security report saved to: {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate comprehensive security report')
    parser.add_argument('--input-dir', required=True, help='Directory containing scan results')
    parser.add_argument('--output', required=True, help='Output report file')
    parser.add_argument('--scan-id', required=True, help='Scan identifier')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Generate report
        generator = SecurityReportGenerator(args.input_dir, args.output, args.scan_id)
        report_data = generator.generate_report()
        generator.save_report(report_data)
        
        # Print summary
        summary = report_data['summary']
        print(f"\n{'='*50}")
        print(f"SECURITY SCAN SUMMARY")
        print(f"{'='*50}")
        print(f"Scan ID: {report_data['scan_id']}")
        print(f"Timestamp: {report_data['timestamp']}")
        print(f"Overall Score: {summary['overall_score']}")
        print(f"Security Posture: {summary['security_posture']}")
        print(f"Total Issues: {summary['total_issues']}")
        print(f"  Critical: {summary['critical_issues']}")
        print(f"  High: {summary['high_issues']}")
        print(f"  Medium: {summary['medium_issues']}")
        print(f"  Low: {summary['low_issues']}")
        print(f"{'='*50}")
        
        # Exit with appropriate code
        if summary['critical_issues'] > 0:
            sys.exit(2)
        elif summary['high_issues'] > 10:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error generating security report: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 