# Deployment and Release Management

## Deployment Windows

Production deployments are permitted only between 10:00 and 16:00 UTC on
weekdays. Deployments outside the permitted window require approval from the
Release Manager.

## Rollback Policy

Any deployment that increases the error rate above 2 percent must be rolled back
within 10 minutes. The rollback is executed by the on-call engineer without
additional approval.

## Release Approval

A release requires two approvals: one from the code owner and one from a
reviewer who is not the author. Releases with a single approval are blocked
automatically by the CI pipeline.

## Change Freeze

A change freeze is declared for the last 5 business days of each quarter. Only
severity 1 and severity 2 incident fixes are permitted during the freeze.

## Feature Flags

New functionality must ship behind a feature flag. Feature flags are reviewed
every 30 days and stale flags older than 90 days are removed automatically.