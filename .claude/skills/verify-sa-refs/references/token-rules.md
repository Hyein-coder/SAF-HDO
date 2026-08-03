# Token & memory rules — verify-sa-refs

The whole harness exists to answer "do the cited papers support these numbers?"
without dragging heavy PDFs through the main context. Every stage honours these.

1. **Group by source, read once.** One verifier per *unique source*, never one
   per parameter. The NREL design report backs several values — it is read a
   single time and all its params are checked in that pass. The mapper's
   grouping is what makes this possible; keep it correct.

2. **PDF isolation.** Verifiers read/fetch freely inside their own context but
   return ONLY distilled verdicts (value found + location + short quote). Raw
   PDF pages, full abstracts, and long HTML never travel back to the
   orchestrator. One careless paste can cost more tokens than the entire rest of
   the run.

3. **Cache + incremental re-run.** `ref_map.json` and `verdicts.json` persist.
   A re-run re-verifies only sources whose value/citation changed or whose prior
   verdict was `mismatch`/`source_unavailable`; everything else is served from
   cache. Never re-read a heavy PDF whose inputs are unchanged.

4. **Memory across sessions.** After a run, persist a `reference` memory
   (pointer to `ref_map.json` + unique-source list) and a `project` memory
   (last-run date + flag headline). Next session starts from these instead of
   re-discovering the parameter map.

5. **Orchestrator reads files, not agents.** The mapper and verifiers write to
   `sa\ref_check\` and return terse summaries. The orchestrator reads the files
   itself rather than asking agents to echo contents back — the same bytes twice
   is twice the tokens.

6. **Targeted reads inside verifiers.** Jump to the number with Grep / offset
   Read (search the unit, the rounded value, the quantity name) instead of
   reading a 100-page PDF front to back.
