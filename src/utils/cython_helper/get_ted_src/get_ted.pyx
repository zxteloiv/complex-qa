def get_ted(c, ex, transform, ted_key):
    import apted
    from apted.helpers import Tree
    try:
        t1, t2 = c[ted_key][0], ex[ted_key]
        t1, t2 = list(map(transform, (t1, t2)))
        ted = apted.APTED(Tree.from_text(t1), Tree.from_text(t2),
                          config=apted.PerEditOperationConfig(1., 1., 1.))
        d = ted.compute_edit_distance()
    except:
        d = 2147483647
    return d

