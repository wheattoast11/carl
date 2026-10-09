---
last_updated: 2026-10-09
author: Intuition Labs LLC
applies_to: carl-studio, carl-core, carl-encoders
---
# Public and private boundaries

CARL's existing source is MIT. Retain its copyright and license notices.
The reusable encoder package follows the same public source boundary.

| Owner | Contents | Distribution |
| --- | --- | --- |
| carl-core | Lightweight contracts and primitives | Public MIT source |
| carl-encoders | Encoder workers, views and learning | Public MIT source |
| carl-studio | Sessions, tools, capture and prepared execution | Public MIT source |
| Private runtime | Proprietary implementation | Separate licensed installation |
| Operator harness | Launch approval, rights and private release records | Outside public CARL |
| Model bundle | Weights, heads, processor, notices and measurements | Artifact-specific terms |
| Local data | Personal episodes and corrections | Capture scope; not packaged |

Public modules keep optional runtime imports lazy. Private resolution retains
admin.py as its integration seam. A source check is not a substitute for
runtime authorization. Wheels and source archives must be inspected as exact
artifacts; source cleanliness alone does not establish their contents.

A model candidate binds its base checkpoint, adapter/head identities, processor,
source revision, dataset provenance, notices, evaluation population and budgets.
Promotion requires the declared task and retention measurements. Missing
measurements are inconclusive. GGUF also requires format/runtime correspondence
for the supported modalities and heads. Preserve the predecessor for rollback.

Capture grants do not authorize training or redistribution. Code, model, data,
fonts, artwork and brand permissions remain separately addressable. Public
metadata must omit personal payloads and private internal filesystem paths.
