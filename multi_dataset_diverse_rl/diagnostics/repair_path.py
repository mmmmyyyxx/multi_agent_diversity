"""Exact offline oracle-repair game. Never imported by the search runtime.

Inputs are existing, verified equivalence classes, not raw mathematical answers.
None denotes an invalid/no-vote output. Zero is the correct equivalence class.
Repairing an invalid member makes it valid and correct. No Solver is invoked.
"""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from math import factorial


def canonical(classes):
    if len(classes) != 5 or any(x is not None and (type(x) is not int or x < 0) for x in classes):
        raise ValueError('FIVE_NONNEGATIVE_CLASS_IDS_OR_NONE_REQUIRED')
    labels = {}; result = []
    for x in classes:
        if x is None or x == 0:
            result.append(x)
        else:
            labels.setdefault(x, len(labels) + 1)
            result.append(labels[x])
    return tuple(result)


def vote_correct(classes):
    counts = Counter(x for x in classes if x is not None)
    gold = counts.pop(0, 0)
    return int(gold > 0 and gold > max(counts.values(), default=0))


@lru_cache(None)
def _game(classes):
    wrong = tuple(i for i, x in enumerate(classes) if x != 0)
    m = len(wrong)
    values = tuple(vote_correct(tuple(0 if i in {
        wrong[j] for j in range(m) if mask & (1 << j)
    } else x for i, x in enumerate(classes))) for mask in range(1 << m))
    shapley = [Fraction(0)] * 5
    immediate = [Fraction(0)] * 5
    for j, member in enumerate(wrong):
        immediate[member] = Fraction(values[1 << j] - values[0])
        for mask in range(1 << m):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            weight = Fraction(factorial(k) * factorial(m-k-1), factorial(m))
            shapley[member] += weight * (values[mask | (1 << j)] - values[mask])
    immediate_path = tuple(x / m if m else Fraction(0) for x in immediate)
    return dict(shapley=tuple(shapley), immediate=tuple(immediate),
        immediate_path=immediate_path,
        enabling=tuple(a-b for a,b in zip(shapley, immediate_path)),
        count=tuple(Fraction((1,2,4,1,Fraction(1,2),0)[classes.count(0)]) if x != 0 else Fraction(0) for x in classes),
        values=values, wrong=wrong,
        distance=min(mask.bit_count() for mask,v in enumerate(values) if v))


def repair_game(classes):
    """Immutable tuples of exact Fractions; at most 32 virtual votes per game."""
    return dict(_game(canonical(classes)))


def classes_from_record(row):
    answers, valid, correct = (row[k] for k in ('team_answers','team_validity','team_correctness'))
    if any(len(v) != 5 for v in (answers,valid,correct)) or any(type(b) is not bool for b in (*valid,*correct)):
        raise ValueError('MALFORMED_OBSERVATION')
    if any(c and not v for c,v in zip(correct,valid)):
        raise ValueError('CORRECT_INVALID_OBSERVATION')
    labels = {}; semantics = {}; out = []
    correct_classes = set()
    for a,v,c in zip(answers,valid,correct):
        if not isinstance(a,str) or (v and not a):
            raise ValueError('MALFORMED_CLASS')
        if v:
            if a in semantics and semantics[a] != c:
                raise ValueError('INCONSISTENT_CLASS_CORRECTNESS')
            semantics[a] = c
        if c:
            correct_classes.add(a)
        if not v:out.append(None)
        elif c:out.append(0)
        else:
            labels.setdefault(a,len(labels)+1);out.append(labels[a])
    if len(correct_classes)>1:raise ValueError('MULTIPLE_CORRECT_CLASSES')
    result=canonical(out)
    if type(row['vote_correct']) is not bool or vote_correct(result) != row['vote_correct']:
        raise ValueError('HISTORICAL_VOTE_MISMATCH')
    return result


def exhaustive_audit():
    """All labeled five-member classes, quotient by wrong-class renaming.

    Six valid classes suffice: gold plus five distinct wrong answers. Include
    invalid/no-vote separately. Verify every coalition edge and independently
    average all repair permutations for each canonical state.
    """
    seen=set(); edges=0
    for classes in product((0,1,2,3,4,5,None),repeat=5):
        seen.add(canonical(classes))
    for classes in seen:
        g=repair_game(classes);w=g['wrong'];m=len(w);v=g['values']
        assert sum(g['shapley']) == 1-v[0]
        assert all(x>=0 for x in (*g['shapley'],*g['enabling']))
        assert all(a==b+c for a,b,c in zip(g['shapley'],g['immediate_path'],g['enabling']))
        path=[Fraction(0)]*5
        for order in permutations(range(m)):
            mask=0
            for j in order:
                nxt=mask | (1<<j);path[w[j]]+=v[nxt]-v[mask];mask=nxt
        assert tuple(x/factorial(m) for x in path)==g['shapley']
        for j,i in enumerate(w):
            marginal=[]
            for mask in range(1<<m):
                if mask & (1<<j):continue
                delta=v[mask | (1<<j)]-v[mask];edges+=1
                assert delta>=0;marginal.append(delta)
            if not any(marginal):assert g['shapley'][i]==0
            for k in w:
                if classes[i]==classes[k]:assert g['shapley'][i]==g['shapley'][k]
        if v[0]:assert not any(g['shapley'])
        if not v[0]:assert all(g['shapley'][i]>0 for i in w)
    return dict(labeled_states=7**5,canonical_states=len(seen),coalition_edges=edges,
        efficiency=True,nonnegativity=True,dummy=True,same_class_symmetry=True,
        permutation_formula_agreement=True,immediate_enabling_decomposition=True,
        failed_vote_every_wrong_member_positive=True,correct_vote_all_zero=True)
