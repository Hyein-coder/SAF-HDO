---
name: verify-sa-refs
description: Verify the literature references behind TEA/LCA sensitivity-analysis parameter values for the SAF project. Use whenever the user wants to check, verify, audit, or re-run reference/citation checks on SA parameters — "check the SA references", "do the cited papers actually support these numbers", "verify the parameter sources", "re-run the reference check", "update ref verification", "which SA values are un-sourced". Opens each cited source (disk PDF or DOI), confirms the number matches, and records the exact paper + page + section for every value. Token- and memory-frugal by design.
---

# Verify SA literature references

Confirm that every parameter value swept in the SAF TEA/LCA sensitivity
analysis matches what its cited source actually says, and record exactly where
in each source the number lives. Built to be cheap on tokens and to carry state
across sessions.

Execution mode is **sub-agent fan-out/fan-in**: the verifiers are independent
(each reads one source), so there is no team messaging — just parallel
background agents returning compact verdicts. See `references/token-rules.md`
for the rules every stage must honour, and `references/schemas.md` for the JSON
shapes.

## Ground truth (locked)

- Scope: **SA-swept parameters only** (~7 TEA + ~7 LCA). Not the non-swept
  constants.
- Citation authority: the **202607 CEJ** manuscript set
  (`D:\SAF_Nurul\202607_Review_CEJ\Manuscript_ver14.docx` / `SupportingInformation_Ver8.docx`).
- Reference PDFs on disk:
  `D:\SAF_Nurul\20260304_Nurul_manuscript_and_references\260304\SAF references_manuscript\` and `...\SAF references_SI\`.
- Off-disk sources: WebFetch the DOI/URL.
- Working dir for all artifacts: `D:\saf_hdo\sa\ref_check\`.

## Workflow

### 0. Context check (initial vs re-run)

Look at `sa\ref_check\`:
- No `ref_map.json` → **initial run**: do steps 1–5.
- `ref_map.json` + `verdicts.json` present, user asks a plain re-run →
  **incremental**: re-map (mapper refreshes only changed rows), then re-verify
  only sources whose `value_used`/`source_hint` changed or whose prior verdict
  was `source_unavailable`/`mismatch`. Reuse cached verdicts for the rest.
- User asks to re-check one specific parameter → verify just that source.

Tell the user which mode you picked and why (one line).

### 1. Map

Invoke the **sa-ref-mapper** agent (`model: opus`). It writes
`sa\ref_check\ref_map.json` and returns a terse summary. Read the file
yourself; do not ask the agent to paste it.

### 2. Fan out verifiers

For each unique `source` in the map with a resolvable artifact, spawn one
**sa-ref-verifier** agent (`model: opus`, `run_in_background: true`) in
parallel — pass it that one source entry (its `resolved` + `params`). Skip
`source_type: "discarded"` and `resolved.kind: "unresolved"` sources (record
them straight to the report as not-verified). One verifier per source, never
one per parameter — that grouping is the token saving.

### 3. Aggregate

Collect the verdict arrays as agents finish. Merge into
`sa\ref_check\verdicts.json`. **Keep the full `location` + `quote` on every
row** — the record of where each number comes from is the deliverable, not just
pass/fail. A verifier that returns without a precise location for a `match` →
send it back once to pin the page/section.

### 4. Report

Write `sa\ref_check\REPORT.md`:
- **Two tables — one for TEA, one for LCA** (user preference): `param | value_used | source + exact location (paper · page · section/table) | quote | verdict | note`.
- **Flags** section, most important first: `mismatch` rows; values whose SA
  range is un-sourced ("standard/engineering"); citations that exist only in
  `LCA_sensitivity_variables.md` and not in code (`code_vs_md_only`); sources
  absent from the CEJ reference list (`in_cej_refs: false`);
  `source_unavailable` rows needing manual check.
- One-line headline: N verified / N mismatch / N unavailable.

### 5. Memory

Update the two memory files under
`C:\Users\user\.Codex\projects\D--saf-hdo\memory\` (add the one-line pointer to
`MEMORY.md`):
- `reference` type → pointer to `ref_map.json` + the unique-source list, so a
  future session skips re-discovery.
- `project` type → last-run date (absolute) + the headline flag summary.

## Error handling

Verifier fails after one retry → mark that source `source_unavailable`,
continue, note it in the report. Never drop a conflicting value: if two sources
disagree, keep both rows with both citations. Missing input file → map what
exists, flag the gap, do not fabricate.

## Test scenarios

**Normal:** "check the SA references" with no `ref_check/` present → initial
run → `REPORT.md` with a located verdict for every swept param + flags + two
memory files written; orchestrator context holds no raw PDF text.

**Error/edge:** a DOI is paywalled and a metal-price URL is dead → those rows
come back `source_unavailable` with reasons, the run still completes and
reports every other param; a re-run then retries only those two sources.
