import re

def split_sparql(sparql: str,
                 no_prefix: bool = True,
                 anonymize_entity: bool = True,
                 no_comment: bool = True,
                 no_white: bool = True,
                 ):
    """Try the best to split sparql into sequence with regular operations, do not rely on this for strict cases"""
    if no_prefix:
        sparql = re.sub(r"PREFIX[^\n]+\n", "", sparql)  # remove prefix
    if anonymize_entity:
        sparql = re.sub(r"ns:m\.[a-z0-9_]+", "?ent", sparql)  # anonymize entities
    if no_comment:
        sparql = re.sub(r"#[^\n]+\n", "", sparql)  # remove comments
    if no_white:
        sparql = re.sub(r"\n|\t", " ", sparql).strip()  # remove newlines and tabs

    inner_structures = re.split(r'("[^"]+")', sparql)
    for i, s in enumerate(inner_structures):
        if s.startswith('"') and s.endswith('"'):
            inner_structures[i] = s.replace(" ", "##space##")
    sparql = " ".join(inner_structures)

    toks = re.split(r" |([{}()|/]|\^\^)", sparql)
    valid_toks = []
    for t in toks:
        if t is None or len(t) == 0:
            continue

        if "##space##" in t:
            t = t.replace("##space##", " ")

        valid_toks.append(t)

    return valid_toks
