"""PDP-CR phase-1 pilot: GA vs TSR vs multi-start local descent (MLS).

- Five independent seeds / method / instance (one process per seed).
- Exactly the same objective, time limit, worker count and seeds per method.
- Wall time is measured inside each solver; CPU seconds and overruns recorded.
- Resume mode persists each seed as it finishes; results survive interruption.
- There is NO BKS/OPT for these converted instances. 'best_found' is empirical.

CLI:
 python prepare_instances.py
 python run_pilot.py --sizes 50 100 200 --families LC1 LR1 LRC1 \
   --seconds50 600 --seconds100 1800 --seconds200 3600 --workers 5
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter, process_time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

ROOT = Path(__file__).resolve().parent
METHODS = ('GA','TSR','MLS')


def _seed_worker(method, path, seed, seconds, population):
    """Defined at module scope for Windows/macOS ProcessPoolExecutor spawn."""
    from pdpcr.core import Instance, evaluate, replay_routes, decompose_routes
    instance = Instance.load(path)
    cpu0=process_time()
    if method=='GA':
        from pdpcr.genetic import run_ga
        result=run_ga(instance,seed=seed,population_size=population,
                      time_limit=seconds)
        iterations=int(result['generations_completed'])
        offspring=int(result['offspring_completed'])
        initial=float(result['statistics']['initial_best'])
        diagnostics={key:result['statistics'].get(key) for key in
                     ('parent_pairs','novel_crossover_children','order_changed',
                      'adaptive_triggers','local_search_improved','local_search_seconds',
                      'offspring_novel_after_intensify')}
    elif method=='TSR':
        from pdpcr.search import tabu_search
        result=tabu_search(instance,seed=seed,iterations=10**9,
                           time_limit=seconds,beta=0.5,
                           tiny_min_candidates=False,
                           restart_best=300,restart_new=600)
        iterations=int(result['iterations_completed'])
        offspring=None
        initial=float(result['initial_objective'])
        diagnostics={'candidate_budget':result['budgets'],
                     'generated':result['generated']}
    elif method=='MLS':
        from pdpcr.mls import multi_start_descent
        result=multi_start_descent(instance,seed=seed,time_limit=seconds,
                                   local_search_steps=2,local_search_candidates=8)
        iterations=int(result['iterations_completed'])
        offspring=None
        initial=float(result['initial_objective'])
        diagnostics={'local_search_calls':result['local_search_calls'],
                     'improvements':result['improvements']}
    else:
        raise ValueError(method)
    cpu_elapsed=process_time()-cpu0
    # Lightweight internal validity check (not a notebook cell, never alters search).
    schedule=evaluate(instance,result['flow'])
    if abs(schedule.makespan-float(result['objective']))>1e-6:
        raise AssertionError(f'Objective mismatch: {path}, {method}, {seed}')
    routes=decompose_routes(instance,result['flow'])
    replay=replay_routes(instance,routes,schedule.starts)
    if not replay['valid'] or abs(replay['actual_return_max']-schedule.makespan)>1e-6:
        raise AssertionError(f'Invalid robot routes: {path}, {method}, {seed}')
    return {
        'instance':instance.name,'n':instance.n,'m':instance.robots,
        'method':method,'seed':seed,'objective':float(result['objective']),
        'initial_objective':initial,'solver_wall_seconds':float(result['seconds']),
        'solver_cpu_seconds':float(cpu_elapsed),'time_limit_seconds':seconds,
        'time_limit_overrun_seconds':max(0.0,float(result['seconds'])-seconds),
        'iterations_completed':iterations,'offspring_completed':offspring,
        'stop_reason':result.get('stop_reason'),
        'valid':True,'diagnostics':diagnostics,
        'history':result['history'],'flow':result['flow'].tolist(),
        'routes':routes}


def _csv(path,records,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)


def _load_manifest(path):
    records=json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(records,list) or not records:
        raise ValueError('Empty manifest; run prepare_instances.py first')
    return records


def _select(records,sizes,families):
    selected=[r for r in records if int(r['n']) in sizes and r['family'] in families]
    for n in sizes:
        for family in families:
            count=sum(r['n']==n and r['family']==family for r in selected)
            if count!=1:
                raise ValueError(f'Expected one converted file for n={n} {family}, found {count}')
    for r in selected:
        if not (ROOT/r['instance_file']).exists():
            raise FileNotFoundError(f"Missing converted instance {r['instance_file']}")
    return sorted(selected,key=lambda r:(r['n'],families.index(r['family'])))


def _summarize(output,selected,methods,seeds):
    runs=[]; method_summary=[]; comparisons=[]
    for spec in selected:
        records={}
        for method in methods:
            group=[]
            for seed in seeds:
                fn=output/'seeds'/f"{spec['name']}__{method}__seed{seed}.json"
                rec=json.loads(fn.read_text(encoding='utf8'))
                group.append(rec)
            records[method]=group
        best_found=min(x['objective'] for group in records.values() for x in group)
        winners={meth:min(records[meth],key=lambda r:(r['objective'],r['seed'])) for meth in methods}
        line={'instance':spec['name'],'family':spec['family'],'n':spec['n'],
              'm':spec['m'],'best_found':best_found,'reference_type':'best_found_in_this_pilot_not_OPT'}
        for method in methods:
            group=records[method]
            win=winners[method]
            line[f'{method}_best_of_5']=win['objective']
            line[f'{method}_mean']=mean(x['objective'] for x in group)
            line[f'{method}_best_RPD_pct']=100*(win['objective']/best_found-1)
            line[f'{method}_best_seed']=win['seed']
            item={k:v for k,v in win.items() if k!='flow'}
            item['flow']=win['flow']
            (output/'best'/f"{spec['name']}__{method}.json").write_text(
                json.dumps(item,indent=2),encoding='utf8')
            method_summary.append({'instance':spec['name'],'family':spec['family'],
                  'n':spec['n'],'method':method,'best_of_5':win['objective'],
                  'best_seed':win['seed'],'mean':mean(x['objective'] for x in group),
                  'std':pstdev(x['objective'] for x in group),
                  'best_found':best_found,
                  'best_RPD_pct':100*(win['objective']/best_found-1),
                  'mean_RPD_pct':100*(mean(x['objective'] for x in group)/best_found-1),
                  'cpu_total_seconds':sum(x['solver_cpu_seconds'] for x in group),
                  'mean_solver_wall_seconds':mean(x['solver_wall_seconds'] for x in group),
                  'max_time_overrun_seconds':max(x['time_limit_overrun_seconds'] for x in group),
                  'number_seeds':len(seeds)})
            for x in group:
                runs.append({k:v for k,v in x.items() if k not in ('history','flow','routes','diagnostics')}
                           | {'family':spec['family'],'best_found':best_found,
                              'RPD_pct':100*(x['objective']/best_found-1),
                              'best_seed_for_method':x['seed']==win['seed']})
        if 'GA' in methods and 'TSR' in methods:
            line['GA_improvement_vs_TSR_pct']=100*(winners['TSR']['objective']-winners['GA']['objective'])/winners['TSR']['objective']
        if 'GA' in methods and 'MLS' in methods:
            line['GA_improvement_vs_MLS_pct']=100*(winners['MLS']['objective']-winners['GA']['objective'])/winners['MLS']['objective']
        comparisons.append(line)
    _csv(output/'runs.csv',runs,[
        'instance','family','n','m','method','seed','objective','best_found','RPD_pct',
        'initial_objective','best_seed_for_method','solver_wall_seconds','solver_cpu_seconds',
        'time_limit_seconds','time_limit_overrun_seconds','iterations_completed',
        'offspring_completed','valid','stop_reason'])
    _csv(output/'summary.csv',method_summary,[
        'instance','family','n','method','best_of_5','best_seed','mean','std','best_found',
        'best_RPD_pct','mean_RPD_pct','number_seeds','cpu_total_seconds',
        'mean_solver_wall_seconds','max_time_overrun_seconds'])
    line_keys=list(dict.fromkeys(k for row in comparisons for k in row))
    _csv(output/'comparison.csv',comparisons,line_keys)
    overall=[]
    for n in sorted({r['n'] for r in selected}):
        for method in methods:
            group=[r for r in method_summary if r['n']==n and r['method']==method]
            overall.append({'n':n,'method':method,'instance_count':len(group),
                            'avg_best_RPD_pct':mean(r['best_RPD_pct'] for r in group),
                            'avg_mean_RPD_pct':mean(r['mean_RPD_pct'] for r in group),
                            'mean_cpu_total_seconds':mean(r['cpu_total_seconds'] for r in group)})
    _csv(output/'overall.csv',overall,['n','method','instance_count',
                                     'avg_best_RPD_pct','avg_mean_RPD_pct','mean_cpu_total_seconds'])
    return comparisons


def run_pilot(sizes=(50,100,200),families=('LC1','LR1','LRC1'),seeds=(0,1,2,3,4),
              seconds_by_n=None,workers=5,population=16,
              methods=METHODS,output_dir=None,manifest=None,resume=True):
    if seconds_by_n is None:
        seconds_by_n={50:600.0,100:1800.0,200:3600.0}
    if len(seeds)!=5 or len(set(seeds))!=5:
        raise ValueError('Use exactly five distinct seeds for each method and instance')
    if not (1<=workers<=5 and population>=2):
        raise ValueError('workers in 1..5 and population >= 2 required')
    methods=tuple(methods)
    if not methods or any(x not in METHODS for x in methods):
        raise ValueError('Unknown methods')
    for n in sizes:
        if seconds_by_n[n]<=0:
            raise ValueError('time budget must be positive')
    manifest=Path(manifest or ROOT/'instances'/'manifest.json')
    selected=_select(_load_manifest(manifest),list(sizes),list(families))
    output=Path(output_dir or ROOT/'results'/'pilot_9_10_30_60min')
    for folder in ('seeds','best'):(output/folder).mkdir(parents=True,exist_ok=True)
    config={'sizes':list(sizes),'families':list(families),'seeds':list(seeds),
            'methods':list(methods),'seconds_by_n':{str(k):seconds_by_n[k] for k in sizes},
            'workers':workers,'GA_population_per_process':population,
            'TSR_options':{'beta':0.5,'tiny_min_candidates':False,
                           'restart_best':300,'restart_new':600},
            'MLS_options':{'local_search_steps':2,'local_search_candidates':8},
            'reference':'best_found computed from all methods/seeds, NOT certified optimum',
            'source':'Li-Lim adapted for PDP-CR (not original PDPTW objective)',
            'software':{'python':sys.version,'platform':platform.platform()},
            'timing':'solver time budget is per independent process; startup not charged; soft cutoff inside moves; CPU time and any overrun recorded'}
    cfg=output/'config.json'
    if cfg.exists() and resume:
        previous=json.loads(cfg.read_text(encoding='utf8'))
        # Machine paths/versions not part of the experimental parameter identity.
        keys=['sizes','families','seeds','methods','seconds_by_n','workers','GA_population_per_process']
        if any(previous.get(k)!=config[k] for k in keys):
            raise ValueError('Existing output has different configuration. Use another output_dir or --no-resume.')
    else:
        if any((output/'seeds').glob('*.json')):
            raise ValueError('Output dir already contains run files; choose empty dir.')
        cfg.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    for spec in selected:
        path=ROOT/spec['instance_file']
        for method in methods:
            missing=[seed for seed in seeds if not (output/'seeds'/f"{spec['name']}__{method}__seed{seed}.json").exists()]
            if not missing:
                print(f"Already done: n={spec['n']} {spec['family']} {method}",flush=True)
                continue
            t0=perf_counter()
            print(f"n={spec['n']} {spec['family']} | {method}: {len(missing)} seeds, "
                  f"{seconds_by_n[spec['n']]}s/seed, {workers} workers",flush=True)
            with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
                future={pool.submit(_seed_worker,method,str(path),seed,
                          float(seconds_by_n[spec['n']]),population):seed for seed in missing}
                for result_future in as_completed(future):
                    rec=result_future.result()
                    # Atomic per-seed commit: safe to resume after interruption.
                    target=output/'seeds'/f"{spec['name']}__{method}__seed{rec['seed']}.json"
                    temp=target.with_suffix('.json.tmp')
                    temp.write_text(json.dumps(rec,ensure_ascii=False),encoding='utf8')
                    temp.replace(target)
                    print(f"  seed {rec['seed']}: objective={rec['objective']:.4f}; "
                          f"wall={rec['solver_wall_seconds']:.2f}s cpu={rec['solver_cpu_seconds']:.2f}s",flush=True)
            print(f"  finished in {perf_counter()-t0:.2f}s total wall",flush=True)
        # Summaries are regenerated after *all* runs, but seed runs already saved.
    comparisons=_summarize(output,selected,methods,seeds)
    print('RESULTS:',output,flush=True)
    return comparisons


def main():
    p=argparse.ArgumentParser(description='Phase 1: GA vs TSR vs multi-start local search')
    p.add_argument('--sizes',nargs='+',type=int,default=[50,100,200])
    p.add_argument('--families',nargs='+',default=['LC1','LR1','LRC1'])
    p.add_argument('--seeds',type=int,nargs=5,default=[0,1,2,3,4])
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--population',type=int,default=16)
    p.add_argument('--seconds50',type=float,default=600.0)
    p.add_argument('--seconds100',type=float,default=1800.0)
    p.add_argument('--seconds200',type=float,default=3600.0)
    p.add_argument('--output',default=None)
    p.add_argument('--manifest',default=None)
    p.add_argument('--no-resume',action='store_true')
    a=p.parse_args()
    run_pilot(a.sizes,a.families,a.seeds,
              seconds_by_n={50:a.seconds50,100:a.seconds100,200:a.seconds200},
              workers=a.workers,population=a.population,
              output_dir=a.output,manifest=a.manifest,resume=not a.no_resume)

if __name__=='__main__':
    main()
