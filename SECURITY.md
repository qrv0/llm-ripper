# Security Policy

## Supported Versions
We support the latest minor release series.

## Reporting a Vulnerability
Please do not create a public issue for sensitive reports.

Email: qorvuscompany@gmail.com

Expected timeline: we aim to acknowledge within 72 hours.

## Threat model and risky operations

- Loading third-party models/code may execute untrusted code paths. The CLI exposes `--trust-remote-code` and requires explicit confirmations in risky flows.
- Prefer pinned model revisions and verified publishers.
- Run in sandboxed environments for evaluation of unknown artifacts.

## Reporting a Vulnerability

Please open a security advisory or contact maintainers privately. Provide:
- Affected versions and environment
- Steps to reproduce
- Impact assessment and proposed mitigation if available
