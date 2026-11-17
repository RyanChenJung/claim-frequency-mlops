# Version data with DVC and share it through a DagsHub remote

Status: accepted (2026-11-17)

DVC versions every data artifact: raw, cleaned, train / validation / test splits, and the drift datasets. These are stages in `dvc.yaml`, with hashes recorded in `dvc.lock`. Every MLflow run records the git commit and the DVC data hash, so any registered model can be traced back to the exact data and code that produced it.

The shared remote is DagsHub's S3-compatible endpoint (`s3://dvc` with `endpointurl https://dagshub.com/<owner>/<repo>.s3`). Each member authenticates with a personal DagsHub token stored only in `.dvc/config.local` (`dvc remote modify --local`).

## Considered options

- **Google Drive (`dvc-gdrive`):** Google blocks DVC's default OAuth app (treeverse/dvc#10516, closed as not planned), so the team would need its own GCP OAuth client. Other costs:
  - Refresh tokens expire every 7 days in testing mode.
  - University Workspace policies may block the app.
  - PyDrive2 requests full-Drive scope.
- **S3:** requires sharing AWS credentials among students.
- **Local remote only:** teammates could not share data versions.

## Consequences

- As of 2026-11-17, DagsHub denies anonymous reads even on public repositories (verified). Graders therefore reproduce the data with `dvc repro`, which rebuilds everything from the public OpenML source without any credentials. The README documents this path.
- Forgetting `--local` when setting the token would commit it to a public repository. The onboarding guide gives the exact commands.
- Automation never commits to git. A change to `dvc.lock` lands through a normal PR with the code change that caused it.
