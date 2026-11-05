# JSON schemas — verify-sa-refs

Two files live in `D:\saf_hdo\sa\ref_check\`. Keep them small and flat.

## ref_map.json (written by sa-ref-mapper)

Parameters grouped under unique sources. One `source` object per real source;
its `params` array holds every SA parameter that cites it.

```json
{
  "generated": "<absolute date, filled by orchestrator after run>",
  "sources": [
    {
      "source_id": "nrel_design_2013",
      "source_hint": "NREL report p.71, 76.57 ¢/MMBtu",
      "source_type": "disk_pdf",
      "resolved": { "kind": "disk_pdf", "path": "D:\\SAF_Nurul\\...\\SAF references_manuscript\\12. ....pdf" },
      "in_cej_refs": true,
      "params": [
        {
          "param": "ng_price",
          "value_used": 0.0547,
          "unit": "$/kg",
          "sa_range": "x[0.50 0.75 1.00 1.25 1.50]",
          "code_vs_md_only": false,
          "note": "escalated from NG_2011=0.0363 via CEPCI 816/541.7"
        }
      ]
    }
  ]
}
```

Field notes:
- `source_type`: `disk_pdf` | `doi` | `url` | `standard` | `discarded`.
- `resolved.kind`: `disk_pdf` (+`path`) | `doi` (+`ref`) | `url` (+`ref`) | `unresolved`.
- `in_cej_refs`: `true` | `false` | `unknown` — present in the 202607 CEJ reference list.
- `code_vs_md_only`: `true` when the citation is only in `LCA_sensitivity_variables.md`, not in the code that sets the value.
- `standard` / `discarded` sources keep `params` but need no `resolved` artifact; the verifier skips them.

## verdicts.json (written by orchestrator from verifier returns)

Flat array, one object per (param, source) checked.

```json
[
  {
    "param": "ng_price",
    "source_id": "nrel_design_2013",
    "value_used": 0.0547,
    "unit": "$/kg",
    "value_in_source": "76.57 ¢/MMBtu",
    "location": "Davis et al. 2013 NREL/TP-5100-60223, Table 20, p.71",
    "quote": "Natural gas ... 76.57 ¢/MMBtu",
    "verdict": "match",
    "note": "converted ¢/MMBtu -> $/kg via NG HHV; consistent with escalation"
  }
]
```

Field notes:
- `verdict`: `match` | `mismatch` | `not_found` | `source_unavailable`.
- `location` and `quote` are **mandatory** whenever `verdict` is `match` or
  `mismatch` — they are the audit trail. `not_found`/`source_unavailable` record
  the reason in `note` instead.
- `value_in_source` is a string (keeps original units/wording); the conversion
  reasoning goes in `note`.
