# Production Incident Response Runbook

## Severity Levels

Severity 1 incidents are defined as complete loss of a customer-facing service.
Severity 2 incidents are defined as a partial degradation affecting more than
10 percent of requests. All other incidents are Severity 3.

## Acknowledgement Targets

Severity 1 incidents must be acknowledged by the on-call engineer within 5
minutes. Severity 2 incidents must be acknowledged within 15 minutes. Severity
3 incidents may be acknowledged within 4 hours during business days.

## Escalation Path

If the on-call engineer does not acknowledge a Severity 1 incident within 5
minutes, the alert is escalated to the incident commander. The incident
commander must post a written status update every 30 minutes until the incident
is resolved.

## Retry and Circuit Breakers

Services must not retry a failed downstream call more than 3 times. After 3
failed attempts the circuit breaker opens and traffic is routed to the
degraded-mode endpoint. Circuit breakers reset automatically after 60 seconds
of successful traffic.

## Postmortems

A written postmortem is required for every Severity 1 and Severity 2 incident.
The postmortem must be completed within 14 days and reviewed by the Engineering
Leadership team.