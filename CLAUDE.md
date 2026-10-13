# Working in this repository with Claude Code

Read `README.md` for the layout, the checks and the measurements, and `TODO.md` for the open work.
This file holds the rules for a session.

## Running sub-agents

The owner's rules, given on 2026-10-13, for every session:

- Run at most three sub-agents at the same time. Start the next one only when one has finished.
- Stop a sub-agent as soon as its report is in. A sub-agent stops every server and background
  process that it started before it reports.
- Give every task its pull request first. Push the first commit early and open a draft pull
  request at once. Push each later commit to the same branch, so that no work stays only in the
  container.
