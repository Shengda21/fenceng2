#!/bin/bash
# Pipeline smoke test on A episodes only (never read): one LLM arm per prompt family, plus a MAgent fixed arm whose
# return must equal the locked calibration return of that episode.
mkdir -p /root/calib/tmp /root/calib/smoke
cd /root/calib/tmp
while IFS= read -r line; do
  [ -z "$line" ] && continue
  d=$(mktemp -d -p /root/calib/tmp); (cd "$d" && bash -c "$line"); echo "EXIT $? :: ${line:0:160}"
done < /root/calib/tasks/smoke_tasks.txt
R=/root/calib
$R/../envs/oghp/bin/python -c "print(1)" 2>/dev/null
/root/envs/oghp/bin/python $R/code/release/oghp/experiments0106b/scripts/rs/run_rs_combined.py --type-set base4 --M 4 --encoding semantic --schedule single --delta-scale 1.0 --type-hints --blue scripted --k 20 --map-size 16 --max-cycles 120 --n-melee 6 --n-ranged 6 --permutation-seed 0 --default HOLD_LINE --pool combined_pool6 --arm fixed:ALL_ATTACK --T 1 --lock $R/locks/smoke_combined-base4.json --out $R/smoke/magent/fixed_ALL_ATTACK.json
/root/envs/oghp/bin/python - <<'PY'
import json, glob
d = json.load(open('/root/calib/smoke/magent/fixed_ALL_ATTACK.json'))
rows = d['rows'] if isinstance(d, dict) else d
lock = json.load(open('/root/calib/locks/calib_table_combined-base4.json'))
ep = [e for e in lock['episodes'] if e['seed'] == 1000][0]
print('MAGENT_REPLAY_CHECK run', rows[0]['total_reward'], 'lock', ep['fixed']['ALL_ATTACK'])
for f in sorted(glob.glob('/root/calib/smoke/**/*.json*', recursive=True)):
    if f.endswith('.log') or f.endswith('meta.json'):
        continue
    try:
        if f.endswith('.jsonl'):
            recs = [json.loads(l) for l in open(f)]
        else:
            x = json.load(open(f)); recs = x['rows'] if isinstance(x, dict) else x
        for r in recs:
            p = (r.get('provenance') or [{}])[0]
            ex = p.get('extra', {})
            print('SMOKE', f.split('/smoke/')[1], 'seed', r['seed'], 'type', r['type'], 'tau', p.get('tau'), 'fb', p.get('fallback'),
                  'src', ex.get('source') or ex.get('parse_rule'), 'ct', (ex.get('usage') or {}).get('completion_tokens'), 'reward', r['total_reward'])
            print('   PROMPT_HEAD', repr((ex.get('prompt') or '')[:300]))
            print('   RAW', repr((ex.get('raw') or ex.get('completion') or '')[:200]))
    except Exception as e:
        print('ERR', f, e)
PY
echo SMOKE_DONE
