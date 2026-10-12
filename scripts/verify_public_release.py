"""Recompute the public snapshot using only Python's standard library; no API calls.

Covers both the pre-adjudication label metrics and the post-adjudication claims:
the human adjudication of the 75 correctness disagreements, the resulting 450-row
distribution, the human annotation coverage (90 spot checks + 75 adjudications with
16 in common), the "which judge is closer to the human" hit counts, and the cause
adjudication. The judge comparison is descriptive only: the design is paired, the 75
rows were selected because the judges disagreed, and no difference test is run, so
interval overlap is never used as a significance criterion. Anything whose source
data is not shipped prints an explicit "not recomputed" notice instead of being
silently treated as passing.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent.parent
C = {'一致', '部分一致', '不一致'}
F = {'忠实', '部分无依据', '无依据'}
Z95 = 1.959963984540054
# Claims whose source data is not part of the public snapshot.
NOT_RECOMPUTED = []


def read(name):
    return [json.loads(line) for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines() if line.strip()]


def unique(rows, fields):
    keys = [tuple(r[f] for f in fields) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'Duplicate records: {fields}')
    return set(keys)


def kappa(pairs):
    n = len(pairs)
    a = collections.Counter(x for x, _ in pairs)
    b = collections.Counter(y for _, y in pairs)
    agree = sum(x == y for x, y in pairs)
    pe = sum(a[x] * b[x] for x in C) / n ** 2
    return agree, (agree / n - pe) / (1 - pe) if pe != 1 else None


def wilson(k, n, z=Z95):
    """95% Wilson score interval; returns (point, low, high)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def confirmed_error_type(rationale, memo):
    """Human-confirmed error type, classified from the reviewer's own words."""
    text = (rationale or '') + ' || ' + (memo or '')
    rules = (
        ('跨研究混用', ('未在原文', '来自于文献', '另一研究', '混用', '混入')),
        ('证据使用不当', ('并不应该用于此题', '多此一举', '不应该用于')),
        ('要素列举不完整', ('未能列出完整', '列出了部分')),
        ('核心处理建议缺失', ('未明确给出', '核心处理建议')),
        ('未给所问层面结论', ('未给所问', '未给技术差异', '未说明具体原因')),
        ('文献有据但未检索到', ('未检索到', '未进入模型上下文')),
    )
    for name, keys in rules:
        if any(k in text for k in keys):
            return name
    return '其他/未归类'


def adjudication_metrics(judges):
    """Post-adjudication layer: human verdicts, final distribution, judge comparison."""
    adjudicated = [r for r in judges if 'human_adjudication' in r]
    base = [r for r in judges if 'human_adjudication' not in r]
    out = {}

    # --- 1. the 75 rows must be exactly the two-judge disagreements ---
    assert base, 'no base rows found'
    assert all(r.get('human_adjudicated') is False for r in base), \
        'rows without human_adjudication must be explicitly marked human_adjudicated: false'
    assert all(r['judge1_correctness'] == r['judge2_correctness'] for r in base), \
        'only two-judge disagreements may carry human_adjudication'
    assert all(r['judge1_correctness'] != r['judge2_correctness'] for r in adjudicated), \
        'every human_adjudication row must be a two-judge disagreement'
    assert len(adjudicated) == 75 and len(base) == 375, (len(adjudicated), len(base))
    assert all(r['human_adjudication'] in C for r in adjudicated), 'invalid human_adjudication value'

    human = collections.Counter(r['human_adjudication'] for r in adjudicated)
    out['adjudication_distribution'] = {k: human[k] for k in ('一致', '部分一致', '不一致')}
    assert (human['一致'], human['部分一致'], human['不一致']) == (56, 10, 9), human

    # --- 2. P0 priority layer: 18 rows = 16 / 2 / 0 ---
    if all('priority_p0' in r for r in adjudicated):
        p0 = [r for r in adjudicated if r['priority_p0']]
        d = collections.Counter(r['human_adjudication'] for r in p0)
        out['p0'] = {'n': len(p0), **{k: d[k] for k in ('一致', '部分一致', '不一致')}}
        assert (len(p0), d['一致'], d['部分一致'], d['不一致']) == (18, 16, 2, 0), out['p0']
    else:
        NOT_RECOMPUTED.append(
            'P0 split (18 = 16 / 2 / 0): no priority field in the public snapshot -> NOT recomputed')

    # --- 3. final 450-row distribution: the human verdict replaces the disagreement label ---
    final = collections.Counter()
    baseline = collections.Counter()
    for r in judges:
        if 'human_adjudication' in r:
            final[r['human_adjudication']] += 1
        else:
            baseline[r['judge1_correctness']] += 1
            final[r['judge1_correctness']] += 1
    out['final_distribution'] = {k: final[k] for k in ('一致', '部分一致', '不一致')}
    out['final_distribution']['total'] = sum(final.values())
    out['presumed_baseline'] = {'rows': sum(baseline.values()),
                                **{k: baseline[k] for k in ('一致', '部分一致', '不一致')}}
    assert sum(final.values()) == 450, final
    assert sum(baseline.values()) == 375, baseline
    assert (final['一致'], final['部分一致'], final['不一致']) == (334, 19, 97), final
    assert (baseline['一致'], baseline['部分一致'], baseline['不一致']) == (278, 9, 88), baseline

    # --- 4. "which judge is closer to the human", on the 75 rows only ---
    # Descriptive only: the two judges scored the same answers (paired), same-question
    # answers across models are correlated, and these 75 rows were selected BECAUSE the
    # judges disagreed. No difference test is run here, and interval overlap is NOT used
    # as a significance criterion (overlapping intervals do not imply "not significant").
    hit1 = sum(r['judge1_correctness'] == r['human_adjudication'] for r in adjudicated)
    hit2 = sum(r['judge2_correctness'] == r['human_adjudication'] for r in adjudicated)
    p1, lo1, hi1 = wilson(hit1, len(adjudicated))
    p2, lo2, hi2 = wilson(hit2, len(adjudicated))
    out['closer_to_human'] = {
        'judge1': {'hit': hit1, 'n': len(adjudicated), 'rate': round(p1, 4)},
        'judge2_v3': {'hit': hit2, 'n': len(adjudicated), 'rate': round(p2, 4)},
        'descriptive_ci_not_a_test': {
            'judge1': [round(lo1, 4), round(hi1, 4)],
            'judge2_v3': [round(lo2, 4), round(hi2, 4)],
            'caveat': ('unadjusted Wilson intervals; same-question correlation and the paired '
                       'design are NOT handled, so they are reported for description only and '
                       'must not be used to claim a significant or non-significant difference'),
        },
        'reading': ('descriptive result within the disagreement sample: judge1 agrees with the human '
                    'verdict on 30 of 75 rows (40.0%), judge2 v3 on 44 of 75 (58.7%). No difference '
                    'test accounting for the paired structure and the same-question correlation was '
                    'run, so this does not rank the two judges on the full answer set'),
    }
    assert (hit1, hit2) == (30, 44), (hit1, hit2)
    assert round(p1 * 100, 1) == 40.0 and round(p2 * 100, 1) == 58.7, (p1, p2)
    # the intervals are descriptive only; they must reproduce the published figures
    assert (round(lo1 * 100, 1), round(hi1 * 100, 1)) == (29.7, 51.3), (lo1, hi1)
    assert (round(lo2 * 100, 1), round(hi2 * 100, 1)) == (47.4, 69.1), (lo2, hi2)
    assert 'not significant' not in json.dumps(out['closer_to_human']), 'no significance reading allowed'

    # --- 4b. human annotation coverage: 90 spot checks + 75 adjudications - 16 overlap ---
    spot = read('human_spot_checks.jsonl')
    spot_keys = {(r['model'], r['question_id']) for r in spot}
    adj_keys = {(r['model'], r['question_id']) for r in adjudicated}
    base_spot = [r for r in base if (r['model'], r['question_id']) in spot_keys]
    covered = spot_keys | adj_keys
    assert all(r.get('human_spot_checked') is ((r['model'], r['question_id']) in spot_keys)
               for r in judges), 'human_spot_checked must match the 90-row spot-check file'
    out['human_annotation_coverage'] = {
        'spot_checked_rows': len(spot_keys),
        'adjudicated_rows': len(adj_keys),
        'spot_checked_and_adjudicated': len(spot_keys & adj_keys),
        'rows_with_a_human_correctness_record': len(covered),
        'presumed_baseline_rows': len(base),
        'presumed_baseline_spot_checked': len(base_spot),
        'presumed_baseline_never_human_reviewed': len(base) - len(base_spot),
        'note': ('human_adjudicated: false means "not part of the disagreement-adjudication '
                 'workflow", NOT "never reviewed by a human"; use human_spot_checked'),
    }
    assert len(spot_keys) == 90 and len(adj_keys) == 75, (len(spot_keys), len(adj_keys))
    assert len(spot_keys & adj_keys) == 16, len(spot_keys & adj_keys)
    assert len(covered) == 149, len(covered)
    assert (len(base_spot), len(base) - len(base_spot)) == (74, 301), (len(base_spot), len(base))
    spot_label = {(r['model'], r['question_id']): r['human_correctness'] for r in spot}
    assert all(spot_label[(r['model'], r['question_id'])] == r['judge1_correctness'] for r in base_spot), \
        'the 74 presumed baseline rows that were spot checked must carry the same label'

    # --- 4c. the 97 "不一致" rows: 9 from the adjudication + 88 from the presumed baseline ---
    non_agree_adj = sum(1 for r in adjudicated if r['human_adjudication'] == '不一致')
    non_agree_base = sum(1 for r in base if r['judge1_correctness'] == '不一致')
    non_agree_base_spot = sum(1 for r in base
                              if r['judge1_correctness'] == '不一致'
                              and (r['model'], r['question_id']) in spot_keys)
    out['disagreement_scope'] = {
        'total': non_agree_adj + non_agree_base,
        'from_adjudicated_disagreements': non_agree_adj,
        'from_presumed_two_judge_agreement': non_agree_base,
        'of_which_spot_checked': non_agree_base_spot,
        'with_human_non_agreement_label': non_agree_adj + non_agree_base_spot,
        'without_any_human_label': non_agree_base - non_agree_base_spot,
    }
    assert (non_agree_adj, non_agree_base) == (9, 88), (non_agree_adj, non_agree_base)
    assert (non_agree_base_spot, non_agree_adj + non_agree_base_spot,
            non_agree_base - non_agree_base_spot) == (18, 27, 70)
    assert non_agree_adj + non_agree_base == out['final_distribution']['不一致'] == 97

    # --- 5. human-classified error types among the 19 non-"一致" rows ---
    # The type name is assigned by keyword rules over the human-written adjudication
    # rationale; it is not a category the reviewer filled in item by item.
    non_agree = [r for r in adjudicated if r['human_adjudication'] != '一致']
    out['error_type_scope'] = len(non_agree)
    assert len(non_agree) == 19, len(non_agree)
    assert len(non_agree) == sum(1 for r in adjudicated if r['human_adjudication'] == '部分一致') \
        + sum(1 for r in adjudicated if r['human_adjudication'] == '不一致'), \
        'the 19 error-type rows are the 10 partial + 9 non-agreement rows, not a subset of 97'
    if any(r.get('human_rationale') for r in non_agree):
        dist = collections.Counter(confirmed_error_type(r.get('human_rationale'), r.get('human_comment'))
                                   for r in non_agree)
        out['error_types_classified_from_human_rationale'] = dict(dist)
        expected = {'要素列举不完整': 5, '文献有据但未检索到': 5, '未给所问层面结论': 5,
                    '跨研究混用': 2, '证据使用不当': 1, '核心处理建议缺失': 1}
        assert dict(dist) == expected, dist
    else:
        NOT_RECOMPUTED.append(
            'error types classified from the human rationale (n=19): no rationale text in the '
            'package -> NOT recomputed')
    return out


def cause_metrics():
    """Cause adjudication: the 16-row confirmation table and the 18-row refused-answer scope."""
    path = ROOT / 'data' / 'cause_adjudication.jsonl'
    if not path.is_file():
        NOT_RECOMPUTED.append('cause adjudication: data/cause_adjudication.jsonl is absent -> NOT recomputed')
        return {'status': 'not recomputed (file absent)'}
    rows = read('cause_adjudication.jsonl')
    out = {}
    assert unique(rows, ['model', 'question_id']) == {(r['model'], r['question_id']) for r in rows}
    scoped = [r for r in rows if r.get('in_refused_answer_scope')]
    table_rows = [r for r in rows if r['cause_source'].startswith('qa/成因待确认.csv')]
    prefilled = [r for r in rows if not r['cause_source'].startswith('qa/成因待确认.csv')]
    out['records_in_file'] = len(rows)
    out['cause_table_rows'] = len(table_rows)
    out['prefilled_rows'] = len(prefilled)
    out['scope_rows'] = len(scoped)
    assert len(table_rows) == 16, len(table_rows)
    assert len(prefilled) == 4, len(prefilled)
    assert len(scoped) == 18, len(scoped)
    assert {r['confirmed_cause'] for r in table_rows} == {'库内存在但未进上下文'}, \
        'qa/成因待确认.csv must be 16/16 库内存在但未进上下文'
    dist = collections.Counter(r['confirmed_cause'] for r in scoped)
    out['scope_breakdown'] = dict(dist)
    assert dict(dist) == {'库内存在但未进上下文': 14, '检索没给到': 4}, dist
    retrieval_side = sum(1 for r in scoped
                         if r['confirmed_cause'] in ('库内存在但未进上下文', '检索没给到'))
    model_side = sum(1 for r in scoped if r['confirmed_cause'] == '模型没读出证据')
    out['retrieval_or_context_side'] = retrieval_side
    out['model_side'] = model_side
    assert (retrieval_side, model_side) == (18, 0), (retrieval_side, model_side)
    assert all(r.get('registered_chunk_in_kb') for r in scoped), \
        'every scoped row must record that the registered chunk was present in the KB'
    out['limit'] = ('scope: the chunk registered at question-writing time, NOT proof that the fragment the '
                    'model received held no equivalent evidence; the quantitative split is not done')
    NOT_RECOMPUTED.append(
        'cause split "retrieval did not supply it" vs "the model did not read it": needs the controlled '
        're-answer experiment, which is not shipped -> NOT recomputed (limit retained)')
    return out


def calculate():
    q = read('questions.jsonl')
    answers = read('answer_metadata.jsonl')
    judges = read('judgements.jsonl')
    human = read('human_spot_checks.jsonl')
    retrieval = read('retrieval_review.jsonl')
    traps = read('trap_review.jsonl')
    qids = {r['question_id'] for r in q}
    if len(unique(q, ['question_id'])) != 90:
        raise ValueError('Expected 90 questions in this snapshot')
    models = sorted({r['model'] for r in answers})
    assert len(models) == 5
    expected = {(m, mode, qid) for m in models for mode in ['open', 'closed'] for qid in qids}
    assert unique(answers, ['model', 'mode', 'question_id']) == expected
    expected_open = {(m, qid) for m in models for qid in qids}
    assert unique(judges, ['model', 'question_id']) == expected_open
    assert unique(human, ['model', 'question_id']) <= expected_open and len(human) == 90
    assert {x[0] for x in unique(retrieval, ['question_id'])} == qids
    assert unique(traps, ['model', 'question_id']) <= expected_open and len(traps) == 50
    for r in judges:
        assert r['judge1_correctness'] in C and r['judge2_correctness'] in C
        assert r['judge1_faithfulness'] in F and r['judge2_faithfulness'] in F
    assert all(r['human_correctness'] in C for r in human)
    assert all(not r.get('error') for r in answers)
    om = {(r['model'], r['question_id']): r for r in answers if r['mode'] == 'open'}
    jm = {(r['model'], r['question_id']): r for r in judges}
    assert all(r['answer_input_fp'] == om[(r['model'], r['question_id'])]['fp'] for r in judges)
    agree_c = sum(r['judge1_correctness'] == r['judge2_correctness'] for r in judges)
    agree_f = sum(r['judge1_faithfulness'] == r['judge2_faithfulness'] for r in judges)
    h1 = [(jm[(r['model'], r['question_id'])]['judge1_correctness'], r['human_correctness']) for r in human]
    h2 = [(jm[(r['model'], r['question_id'])]['judge2_correctness'], r['human_correctness']) for r in human]
    ans_ret = [r for r in retrieval if r['answerable']]
    hit = sum(r['source_in_top6'] for r in ans_ret)
    sufficient = sum(r['context_sufficient'] for r in ans_ret)
    assert (agree_c, agree_f, kappa(h1)[0], kappa(h2)[0], hit, sufficient) == (375, 382, 82, 82, 24, 50)
    kappa_c = kappa([(r['judge1_correctness'], r['judge2_correctness']) for r in judges])[1]
    assert round(kappa_c, 3) == 0.650, f'correctness kappa should be 0.650, got {kappa_c}'
    stats = []
    for model in models:
        rs = [r for r in answers if r['model'] == model]
        js = [r for r in judges if r['model'] == model]
        faithful = sum(r['judge1_faithfulness'] == '忠实' for r in js)
        stats.append({'model': model, 'faithful': faithful, 'n': len(js), 'cost': sum(r['cost'] for r in rs),
                      'open_latency_s': sum(r['latency_s'] for r in rs if r['mode'] == 'open') / 90})
    faithful_rates = {s['model']: round(s['faithful'] / s['n'] * 100, 1) for s in stats}
    total_cost = round(sum(s['cost'] for s in stats), 4)
    assert round(min(faithful_rates.values()), 1) == 85.6, faithful_rates
    assert round(max(faithful_rates.values()), 1) == 95.6, faithful_rates
    assert total_cost == 9.3473, total_cost
    mcq = json.loads((ROOT / 'data' / 'batch_b_mcq_summary.json').read_text(encoding='utf-8'))
    mcq_rates = sorted(round(v / mcq['n_per_model'] * 100, 1) for v in mcq['correct'].values())
    assert mcq_rates == [90.0, 93.3, 93.3, 93.3, 93.3], mcq_rates
    out = {'correctness_agreement': [agree_c, len(judges)], 'faithfulness_agreement': [agree_f, len(judges)],
           'correctness_kappa': round(kappa_c, 3),
           'human_judge1': list(kappa(h1)), 'human_judge2': list(kappa(h2)),
           'strict_retrieval': [hit, len(ans_ret)], 'context_sufficiency': [sufficient, len(ans_ret)],
           'judge1_faithfulness_labels': dict(collections.Counter(r['judge1_faithfulness'] for r in judges)),
           'trap_labels': dict(collections.Counter(r['judge'] for r in traps)), 'models': stats,
           'faithfulness_rate_by_model': faithful_rates, 'total_cost_estimate': total_cost,
           'batch_b_mcq_rates_pct': mcq_rates,
           'answer_records': len(answers), 'questions': len(q)}
    out['adjudication'] = adjudication_metrics(judges)
    out['cause_adjudication'] = cause_metrics()
    return out


def verify_manifest():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        p = ROOT / name
        assert p.is_file(), f'Missing file: {name}'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Changed file: {name}'
    print(f"File hashes verified: {len(manifest['sha256'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true', help='For release assembly: skip the file manifest check')
    args = parser.parse_args()
    if not args.data_only:
        verify_manifest()
    metrics = calculate()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print()
    print('Recomputed in this run:')
    adj = metrics['adjudication']
    print(f"  adjudication 75 rows            : {adj['adjudication_distribution']}")
    if isinstance(adj.get('p0'), dict):
        print(f"  P0 18 rows                      : {adj['p0']}")
    else:
        print(f"  P0 18 rows                      : {adj['p0']}")
    print(f"  final 450 distribution          : {adj['final_distribution']}")
    print(f"  375-row presumed baseline       : {adj['presumed_baseline']}")
    closer = adj['closer_to_human']
    print(f"  who is closer to the human      : descriptive only - judge1 {closer['judge1']['hit']}/"
          f"{closer['judge1']['n']} ({closer['judge1']['rate'] * 100:.1f}%), judge2 v3 "
          f"{closer['judge2_v3']['hit']}/{closer['judge2_v3']['n']} "
          f"({closer['judge2_v3']['rate'] * 100:.1f}%); paired design and same-question correlation "
          f"not handled -> no difference test, no significance reading")
    cov = adj['human_annotation_coverage']
    print(f"  human annotation coverage       : {cov['rows_with_a_human_correctness_record']}/450 = "
          f"{cov['spot_checked_rows']} spot checks + {cov['adjudicated_rows']} adjudications - "
          f"{cov['spot_checked_and_adjudicated']} in both; presumed baseline "
          f"{cov['presumed_baseline_spot_checked']} spot checked / "
          f"{cov['presumed_baseline_never_human_reviewed']} never reviewed")
    ds = adj['disagreement_scope']
    print(f"  97 '不一致' = {ds['from_adjudicated_disagreements']} adjudicated + "
          f"{ds['from_presumed_two_judge_agreement']} presumed ({ds['of_which_spot_checked']} of those "
          f"spot checked) -> {ds['with_human_non_agreement_label']} with a human label, "
          f"{ds['without_any_human_label']} without")
    if 'error_types_classified_from_human_rationale' in adj:
        print(f"  error types classified from the human rationale (n={adj['error_type_scope']}): "
              f"{adj['error_types_classified_from_human_rationale']}")
    cause = metrics['cause_adjudication']
    if 'scope_breakdown' in cause:
        print(f"  cause adjudication              : {cause['scope_rows']} in scope of "
              f"{cause['records_in_file']} records; {cause['scope_breakdown']}; "
              f"retrieval/context side {cause['retrieval_or_context_side']}/{cause['scope_rows']}, "
              f"model side {cause['model_side']}")
    print()
    if NOT_RECOMPUTED:
        print('NOT RECOMPUTED (source data is not part of this snapshot - not a pass):')
        for item in NOT_RECOMPUTED:
            print(f'  - {item}')
        print()
    print('PASS: public snapshot structure and arithmetic verified; no network/API calls.')
    if NOT_RECOMPUTED:
        print(f'NOTE: {len(NOT_RECOMPUTED)} claim(s) above could not be recomputed from the shipped data.')


if __name__ == '__main__':
    main()
