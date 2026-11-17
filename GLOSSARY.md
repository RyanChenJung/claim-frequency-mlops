# Motor claim frequency

The shared language of this project: predicting how often motor insurance policies produce claims, and operating that prediction as a monitored service.

## Insurance

**Policy**:
One motor third-party liability contract for one vehicle, observed over a period. One row of the dataset.
_Avoid_: customer, record

**Portfolio**:
The full set of policies an insurer holds.

**Exposure**:
The fraction of a year a policy was in force during observation (for example 0.5 = six months).
_Avoid_: duration, tenure

**Claim count**:
The number of claims a policy filed during its exposure (`ClaimNb`).

**Claim frequency**:
Expected claims per policy-year, claim count divided by exposure. This is what the model predicts.
_Avoid_: claim rate, risk score

**Rating factor**:
A policy attribute an insurer may price on: driver age, vehicle age, vehicle power, brand, fuel, bonus-malus, area, density, region.

**Bonus-malus**:
The French no-claims coefficient: 100 for a new driver, falling toward 50 with claim-free years and rising after at-fault claims.

**Adverse selection**:
Losing low-risk policies to competitors that price them more accurately, while keeping the high-risk ones.

## Models and monitoring

**Baseline model**:
A constant model that predicts the portfolio's mean claim frequency for every policy. Every candidate must beat it.
_Avoid_: using "baseline" for the monitoring reference

**Champion**:
The registered model version currently served to clients.

**Challenger**:
A newly registered model version that has been compared with the champion but not promoted.

**Promotion**:
A human decision that makes a challenger the new champion.

**Reference dataset**:
The validation split together with the champion's predictions on it. Monitoring compares live traffic against it.
_Avoid_: baseline, monitoring baseline

**Ground truth**:
The actual claim count of a scored policy. It becomes known only after the policy period, long after the prediction.
_Avoid_: label (in business discussions)

**Drift scenario**:
A named, reproducible corruption of the test set used to prove that monitoring catches a specific failure.

**Contract**:
A shared definition (data schema, model input/output, log tables, CLI commands) that lets roles build their parts independently.
