"""Query-only input generation helpers from the validated runner."""
import json
import re
import string

QUERY_GRAPH_PROMPT = """You convert a multi-hop question into its directed reasoning graph.

Return only valid JSON:
{
  "nodes": [{"id":"n0", "anchor":"exact named mention or null", "role":"type/role"}],
  "edges": [{"source":"n0", "target":"n1", "relation":"short relation"}]
}

Important rules:
1. A named work such as III, Labyrinth, or Till dom ensamma is a work node,
   not the performer/person mentioned by the question.
2. Preserve every explicit named anchor, including the final location such as
   Gulf of Mexico. Never connect the answer back to the first node to make a cycle.
3. Represent possessive/nested questions as a chain from the known entity to
   the unknown answer. Example: "county is Erik Hort's birthplace part of" is
   Erik Hort --born in--> location --part of--> county.
4. For "performer of III", use III --performed by--> person, then continue.
5. Keep relation direction explicit. Use null anchors for unknown variables.
6. Include only the reasoning chain and at least one edge. Preserve named anchors
   when supplied by the question. If the question supplies no specific named
   entity, keep its unknown entities as null anchors with descriptive roles and
   add the top-level field "anchorless_query": true. Do not guess an entity,
   use a gold answer, or mark an explicit named entity as unknown to avoid binding.
7. For a question asking the NAME of an unnamed category, make two
   unknown nodes connected by a naming relation; do not output an empty
   edges list. For example, for "what is the name of the X garment":
   {"nodes":[{"id":"n0","anchor":null,"role":"X garment category"},
   {"id":"n1","anchor":null,"role":"garment name"}],
   "edges":[{"source":"n0","target":"n1","relation":"called"}],
   "anchorless_query":true}
8. Do not answer the question and do not add markdown or explanation.
"""

def _json_object(text):
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {}

def _normalize_text(value):
    """Match the normalization used by the validated query-graph runner."""
    value = str(value).casefold().strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(value.split())

def _validate_graph(payload, question):
    nodes, edges = payload.get("nodes"), payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not edges:
        raise ValueError(f"Invalid query graph: {question}")
    cleaned, ids, anchored = [], set(), 0
    for node in nodes:
        node_id = str(node.get("id"))
        if not node_id or node_id == "None" or node_id in ids:
            raise ValueError(f"Invalid query node: {question}")
        ids.add(node_id)
        anchor = node.get("anchor")
        anchor = _normalize_text(anchor) if anchor else None
        anchored += int(bool(anchor))
        cleaned.append({"id": node_id, "anchor": anchor, "role": str(node.get("role", "variable"))})
    clean_edges = []
    for index, edge in enumerate(edges):
        endpoints = {}
        for side in ("source", "target"):
            value = edge.get(side)
            if value is None:
                value = f"unknown_{index}_{side}"
                ids.add(value)
                cleaned.append({"id": value, "anchor": None, "role": "answer variable"})
            value = str(value)
            if value not in ids:
                raise ValueError(f"Unknown query node {value}: {question}")
            endpoints[side] = value
        relation = _normalize_text(edge.get("relation", ""))
        if not relation:
            raise ValueError(f"Missing query relation: {question}")
        clean_edges.append({**endpoints, "relation": relation})
    anchorless=payload.get('anchorless_query') is True
    if not anchored and not anchorless:
        raise ValueError(f"Query graph has no anchor: {question}")
    result={"nodes": cleaned, "edges": clean_edges}
    if not anchored:
        if any(not n['role'].strip() for n in cleaned):raise ValueError('Anchorless query requires descriptive roles')
        result['anchorless_query']=True
    return result

def _gold_docs(samples):
    return [list({f"{p['title']}\n{p.get('text', p.get('paragraph_text', ''))}" for p in row["paragraphs"] if p.get("is_supporting", True)}) for row in samples]

def _gold_answers(samples):
    output = []
    for row in samples:
        values = {row["answer"]}; values.update(row.get("answer_aliases", [])); output.append(values)
    return output

UNROLL_PROMPT = """You decompose multi-hop questions for retrieval.
Return exactly one JSON object with keys sub_questions and reasoning_chain.
sub_questions is a list of self-contained single-hop questions.
reasoning_chain is a list of [head, relation, tail] triples in reasoning order.
Use <UNCERTAIN> for any entity you are not confident about. The last triple's
tail must be <FILL>. Do not use external search and do not output commentary.

Question: {question}"""
