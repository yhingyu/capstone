# Monitoring, Versioning, and Rollback Plan

## Monitoring

The local app writes one JSON record per forecast to `data/monitoring/predictions.jsonl`. It records the model version, timestamp, input source, prediction, interval, baseline, threshold, advisory level, and fallback warnings. Raw uploaded readings are not written to the log.

For a pilot, review:

- request success/failure and latency;
- missing or nonconsecutive input days;
- voltage-fallback frequency;
- feature-range and seasonal drift;
- interval coverage when actual next-day consumption becomes available;
- MAE/RMSE overall and for high-use days;
- warning precision, recall, false-alert rate, and opt-out rate;
- performance by approved demographic groups only when lawful, consented data exists.

Trigger investigation when MAE rises materially above the 3.7232 kWh test reference, interval coverage falls below the nominal 90% target over an adequate sample, or warning behavior changes materially. These are review triggers, not automatic retraining commands.

## Versioning

- Tag container images, for example `household-energy-monitor-poc:1.0.0`.
- Store the model version in `config/app.yaml` and every prediction log.
- Keep model, feature schema, configuration, metrics, and training metadata together.
- Record the Git commit SHA used to build each image.
- Never overwrite a released model artifact; publish a new version.

## Rollback

1. Disable advisory notifications while leaving monitoring available.
2. Select the last validated Git tag and matching model/config bundle.
3. Rebuild or pull the prior immutable container image.
4. Start the previous image and verify `/health` plus a known sample forecast.
5. Record the reason, affected version, time window, and corrective action.

Because the POC is advisory-only, the safest fallback is to display historical consumption without a forecast rather than serve an unvalidated model.
