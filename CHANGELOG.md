# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Package skeleton: `decision_bakeoff` with schema and metrics modules.
- Decider protocol and members (RouterBaseline, CosineDecider, MockCLMDecider).
- Architectures A0–A5 with calib `fit()` where applicable.
- Synthetic sample JSONL under `data/samples/` (≥20/type, all splits).
