from structured_response import parse_object
"""Bounded execution repair; no changes to query prompts or graph validator."""
import json
import time
from api_runtime import ExperimentPaused
from query_inputs import _json_object, _validate_graph

def generate_valid(client, messages, stage, question, root):
    budget = 400
    last_error = None
    for attempt in range(1, 5):
        # Once complete-schema validation fails, never replay its shallow-valid cache.
        raw, meta, hit = client.infer(messages, max_tokens=budget,
                                      _bypass_cache=attempt > 1)
        try:
            if meta.get('finish_reason') != 'stop':
                raise ValueError('Incomplete generation: '+str(meta.get('finish_reason')))
            obj = parse_object(raw)
            if stage == 'query_graph':
                value = _validate_graph(obj, question)
            else:
                if not isinstance(obj.get('sub_questions'), list) or not isinstance(obj.get('reasoning_chain'), list):
                    raise ValueError('Unrolling requires sub_questions and reasoning_chain lists')
                if not all(isinstance(x, str) for x in obj['sub_questions']):
                    raise ValueError('Invalid sub-question type')
                if not all(isinstance(x, list) and len(x) == 3 and all(isinstance(v, str) for v in x) for x in obj['reasoning_chain']):
                    raise ValueError('Invalid reasoning-chain triple type')
                value = obj
            error = None
        except (ValueError, TypeError, AttributeError) as exc:
            error = str(exc)
        with (root/'input_attempts.jsonl').open('a') as f:
            f.write(json.dumps({'time':time.time(),'stage':stage,'question':question,
                'attempt':attempt,'initial_stage_budget':budget,'requested_max_tokens':meta.get('requested_max_tokens',budget),'cache_hit':hit,
                'metadata':meta,'error':error,'raw_response':raw if error else None})+'\n')
        if error is None:
            return value, raw, meta, hit
        last_error = error
        if meta.get('finish_reason') == 'length':
            budget = min(3200, budget * 2)
    client.pause(f'{stage} invalid after four attempts: {last_error}')
