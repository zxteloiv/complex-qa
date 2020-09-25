import re

def split_sparql(sparql: str):
    sparql = re.sub(r"PREFIX[^\n]+\n", "", sparql)  # remove prefix
    sparql = re.sub(r"ns:m\.[a-z0-9_]+", "?ent", sparql)  # anonymize entities
    sparql = re.sub(r"#[^\n]+\n", "", sparql)  # remove comments
    sparql = re.sub(r"\n|\t", " ", sparql).strip()  # remove newlines and tabs

    inner_structures = re.split(r'("[^"]+")', sparql)
    for i, s in enumerate(inner_structures):
        if s.startswith('"') and s.endswith('"'):
            inner_structures[i] = s.replace(" ", "##space##")
    sparql = " ".join(inner_structures)

    toks = re.split(r" |([{}()]|\^\^)", sparql)
    valid_toks = []
    for t in toks:
        if t is None or len(t) == 0:
            continue

        if "##space##" in t:
            t = t.replace("##space##", " ")

        valid_toks.append(t)

    return valid_toks
