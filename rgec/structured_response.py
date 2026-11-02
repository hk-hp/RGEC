"""Extract one complete fenced JSON object, never complete broken JSON by guess."""
import json
import re


def _load_with_tail_repair(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as original:
        stripped = text.rstrip()
        if not stripped.endswith("}"):
            raise original

        # Only insert missing array closers before the final object brace.
        # Never remove, rewrite, or fabricate a triple or entity.
        for count in range(1, 4):
            candidate = stripped[:-1] + ("]" * count) + "}"
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            # Preserve an audit record of every syntactic repair.
            import os
            import time
            from pathlib import Path
            run_dir = os.environ.get("RGEC_RUN_DIR")
            if run_dir:
                try:
                    with (Path(run_dir) / "json_tail_repairs.jsonl").open("a") as f:
                        f.write(json.dumps({
                            "time": time.time(),
                            "missing_array_closers": count,
                            "original": text,
                            "repaired": candidate
                        }, ensure_ascii=False) + "\n")
                except OSError:
                    pass
            return obj
        raise original

def parse_object(raw):
    text=raw.strip()
    try:value=_load_with_tail_repair(text)
    except json.JSONDecodeError as original:
        blocks=list(re.finditer(r'```(?:json)?[ \t]*\r?\n(.*?)```',text,re.DOTALL|re.IGNORECASE))
        if len(blocks)!=1:raise original
        block=blocks[0]
        # A second partial JSON object outside the fence is not accepted.
        outside=text[:block.start()]+text[block.end():]
        if any(c in outside for c in '{}'):raise ValueError('Ambiguous JSON outside the single fenced object')
        value=_load_with_tail_repair(block.group(1).strip())
    if not isinstance(value,dict):raise ValueError('Expected one complete JSON object')
    return value

def repair_feedback(error,raw,stage=None,attempt=1):
    # One bounded, non-nested correction message. The original messages remain.
    previous=raw[:12000] if isinstance(raw,str) else repr(raw)[:12000]
    output=('Return the original output field [[ ## fact_after_filter ## ]] containing a complete '
        '{"fact": [...]} object, followed by [[ ## completed ## ]]. Do not return a bare JSON object. '
        if stage=='fact_filter' else 'Return exactly one complete JSON object with the original required fields. ')
    syntax=''
    if stage=='unrolling':
        syntax=('\nRequired JSON shape (replace the example strings with your decomposition, not these examples): '
            '{"sub_questions": ["single-hop question"], "reasoning_chain": [["head", "relation", "<FILL>"]]}\n'
            'The two arrays AND the outer object must be closed. Use JSON double quotes; '
            'no trailing comma, no Python literals. Do not supply a known answer for an unknown entity. ')
    return ('Formatting correction attempt '+str(attempt)+'. The previous response is not valid for the requested schema. Error: '+str(error)[:1500]+
        '\n'+output+
        syntax+
        'Correct formatting, including all closing brackets and braces. Preserve the original task constraints; '
        'do not answer the question or invent an entity, fact, or source. '
        'For a query with no supplied named entity, use null anchors, descriptive roles, '
        'and anchorless_query=true as instructed; do not invent an anchor. '
        'The previous untrusted response follows only as a formatting diagnostic:\n'+previous)


# RGEC_TAIL_SYNTAX_REPAIR_V2
# Only repair malformed terminal JSON punctuation.
# Semantic validation remains in api_runtime and input_retry.
_original_parse_object_v2 = parse_object

def parse_object(raw):
    try:
        return _original_parse_object_v2(raw)
    except (ValueError, TypeError):
        import re
        text = raw.strip()

        # Example: ... "reasoning_chain": [[...]]"}
        # Remove the extra quote immediately before the final brace.
        text = re.sub(r'(?<=\])"(?=\s*\}\s*$)', '', text)

        # Example: ... "reasoning_chain": [[...]]
        # Restore only a missing final object brace.
        if text.endswith(']'):
            text += '}'

        return _original_parse_object_v2(text)
