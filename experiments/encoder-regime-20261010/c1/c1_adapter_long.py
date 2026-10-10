"""C1: rerun the adapter stage from step 0 with runtime_s 7200."""
import asyncio
import json
import os
import runpy
import sys
from pathlib import Path

BUNDLE = Path('/workspace/bundle')
OUTPUT = Path('/workspace/results')
manifest_sha = sys.argv[1]
run = runpy.run_path(str(BUNDLE / 'run.py'), run_name='bundle_run')
run['validate_bundle'](manifest_sha)
os.environ['CARL_HOME'] = str(OUTPUT / 'carl-home')
from carl_studio.experiment.manager import ExperimentManager
from carl_studio.training.pipeline import submit_training
from carl_studio.training.preparation import prepare_training
from carl_studio.types.config import TrainingConfig
from carl_studio.types.preparation import TrainingPreparation

original = TrainingPreparation.model_validate_json((OUTPUT / 'adapter-preparation.json').read_text())
values = dict(original.config)
values.update(run_name='a40-adapter-long', output_dir=str(OUTPUT / 'adapter-long'),
              encoder={**values['encoder'], 'runtime_s': 7200})
config = TrainingConfig.model_validate(values)
owner = ExperimentManager(OUTPUT / 'experiments')
plan_path = OUTPUT / 'adapter-long-preparation.json'
if plan_path.exists():
    prepared = TrainingPreparation.model_validate_json(plan_path.read_text())
    config = TrainingConfig.model_validate(prepared.config)
else:
    prepared = prepare_training(config, project_root=BUNDLE / 'evaluator', manager=owner)
    plan_path.write_text(prepared.model_dump_json(indent=2))
print(json.dumps({'stage': 'adapter-long', 'ready': prepared.ready, 'plan_id': prepared.plan_id,
                  'issues': [i.model_dump() for i in prepared.issues]}), flush=True)
if not prepared.ready:
    raise SystemExit(2)
result = asyncio.run(submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner))
(OUTPUT / 'adapter-long-result.json').write_text(result.model_dump_json(indent=2))
print(json.dumps({'stage': 'adapter-long', 'run': result.id, 'phase': result.phase.value,
                  'steps': result.current_step, 'acceptance': result.representation_acceptance,
                  'resources': result.resource_usage}), flush=True)
