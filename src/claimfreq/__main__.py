"""Allow ``python -m claimfreq``, equivalent to the ``claimfreq`` console script."""

from claimfreq.cli import main

raise SystemExit(main())
