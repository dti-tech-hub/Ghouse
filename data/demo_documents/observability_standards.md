# Observability and Monitoring Standards

## Metric Naming

All metrics must use the prefix `app_` followed by the service name and the unit,
for example `app_request_latency_ms`. Names without a unit suffix are rejected
by the metric gateway.

## Alert Thresholds

Latency alerts fire when the 95th percentile latency exceeds 300 milliseconds
for a 5 minute window. Error rate alerts fire when the error rate exceeds 1
percent over 10 minutes. Throughput alerts fire when request volume deviates by
more than 40 percent from the 7 day average.

## Logging Standards

Logs must be structured JSON with the fields timestamp, level, service, request
id and message. Plain-text logs are rejected by the collector. Request
identifiers must propagate across service boundaries unchanged.

## Distributed Tracing

Every request must carry a trace identifier that is forwarded to all downstream
services. A trace is considered broken when fewer than 80 percent of spans in
the critical path carry a parent span reference.

## Dashboard Ownership

Every production dashboard has a named owner and a review date. Dashboards
without an owner are archived after 90 days without review.