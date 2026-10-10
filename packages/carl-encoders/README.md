# carl-encoders

Reusable encoder contracts, media processing, frozen heads and encoder adapters
for CARL. Source license: MIT, copyright Intuition Labs LLC.

```python
from carl_encoders.types import SemanticInput, SemanticPart
from carl_encoders.release import EncoderRelease
```

Install CARL integration with `pip install 'carl-studio[encoders]'`.
Standalone package: `pip install carl-encoders`.
Workers run in a separately bound interpreter; optional heavy dependencies stay
out of base imports. The initial qualified worker uses Transformers 5.19.0 and
SentenceTransformers 6.1.0. CARL binds actual interpreter, dependencies, source
bytes, processor and trainable modules before execution.

`EncoderRelease` describes an exact model bundle and validates its member bytes.
It does not evaluate task performance or authorize publication. Accepted and
GGUF references must identify hash-bound evaluation members of the bundle;
the existing evaluator and prepared activation owners decide acceptance.

Model weights, processors, datasets and external artwork have separate terms.
A package release does not establish model rights or model acceptance. Retain
upstream notices, provenance, evaluation receipts and rollback predecessors.
