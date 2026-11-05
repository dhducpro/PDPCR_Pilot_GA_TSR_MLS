"""Prepare public Li-Lim data as reproducible PDP-CR instances.

Example:
    python prepare_instances.py --families LC1 LR1 LRC1 --sizes 50 100 200
    python prepare_instances.py --families LC1 LR1 LRC1 LC2 LR2 --sizes 50 100 200

Downloads source node data, not PDPTW solutions. Offline users can place .txt or
.pdp files under data/li_lim/. No BKS/OPT from PDPTW is ever copied into PDP-CR.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from pdpcr.li_lim import convert_li_lim, prefix_instance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'li_lim'
OUT = ROOT / 'instances'
FAMILIES = ('LC1','LR1','LRC1','LC2','LR2','LRC2')
SIZES = (50,100,200)


def source_identity(n, family, index=1):
    """n is the target pair count; source size labels are nominal node counts."""
    if n not in SIZES or family not in FAMILIES or index < 1 or index > 10:
        raise ValueError(f'Unsupported n/family/index combination: {n}, {family}, {index}')
    if n == 50:
        # File naming for the original 100-node Li-Lim data: lc101, lr201 etc.
        stem = f'{family.lower()}{index:02d}'
    else:
        stem = f'{family.lower()}_{2*n//100}_{index}'
    return str(2*n), stem


def _fetch(url, seconds=30):
    request = Request(url, headers={'User-Agent':'PDPCR-LiLim-Research-Benchmark/1.0',
                                    'Accept':'text/plain, */*'})
    with urlopen(request, timeout=seconds) as f:
        raw = f.read()
    if len(raw) < 200 or b'\x00' in raw:
        raise ValueError(f'Bad/empty data downloaded from {url}')
    return raw


def get_source(n, family, index=1, download=True):
    DATA.mkdir(parents=True,exist_ok=True)
    folder, stem = source_identity(n, family, index)
    for file in (DATA / (stem+'.txt'), DATA / (stem+'.pdp'),
                 DATA / (stem.upper()+'.pdp')):
        if file.is_file():
            return file, 'local_cache'
    if not download:
        raise FileNotFoundError(f'Missing {stem}.txt/.pdp in {DATA}; allow download or place raw data manually.')
    links = [
        f'https://raw.githubusercontent.com/zhu-he/pdptw-data/main/{folder}/{stem}.txt',
        f'https://github.com/zhu-he/pdptw-data/raw/refs/heads/main/{folder}/{stem}.txt'
    ]
    errors=[]
    for link in links:
        try:
            raw=_fetch(link)
            file=DATA/(stem+'.txt')
            file.write_bytes(raw)
            return file, link
        except (OSError, ValueError, URLError) as e:
            errors.append(f'{link}: {e}')
    raise RuntimeError('Cannot download Li-Lim raw instance '+stem+'; '
                       'manually obtain the corresponding original Li-Lim file '
                       'from https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/ '
                       'and place it in data/li_lim/ as '+stem+'.pdp.\n'+'\n'.join(errors))


def prepare(sizes=(50,100,200),families=('LC1','LR1','LRC1'),robots=10,
            requirement_seed=0, download=True):
    if robots < 5:
        raise ValueError('robots >= 5 is required for Uniform{1,...,5} task requirements')
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for n in sizes:
        for family in families:
            path, url=get_source(n, family, download=download)
            # Validate every original pair before selecting complete tasks.
            # Li-Lim's nominal node count need not equal twice the pair count.
            instance,metadata=convert_li_lim(path,robots=robots,
                                requirement_seed=requirement_seed)
            source_task_count=instance.n
            if source_task_count < n:
                raise ValueError(f'{path.name}: requested {n} complete tasks, '
                                 f'but source contains only {source_task_count}')
            selection='all complete requests'
            if source_task_count > n:
                instance=prefix_instance(instance,count=n,robots=robots,
                    name=f'lilim_{path.stem}_n{n}_m{robots}_req{requirement_seed}_first{n}')
                selection=f'first {n} complete requests ordered by original pickup ID'
            target=OUT/(instance.name+'.json')
            instance.save(target)
            record={'name':instance.name,'n':n,'m':robots,'family':family,
                    'source_id':path.stem,'source_file':path.name,
                    'source_url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'source_task_count':source_task_count,'selection':selection,
                    'selected_pairs':[[t['pickup_id'],t['delivery_id']] for t in metadata['tasks'][:n]],
                    'requirement_seed':requirement_seed,'conversion':'pickup-delivery -> task; Euclidean processing/travel; r_i=randint(1,5)',
                    'instance_file':str(target.relative_to(ROOT)),'optimal_reference':None}
            records.append(record)
            print(f'Prepared {target.name} [{family}; {n}/{source_task_count} source tasks; {selection}]',flush=True)
    (OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    return records


def make_local_demo():
    """A bundled n=100 LR2 file permits an offline smoke run (NOT part of pilot9)."""
    original=DATA/'LR2_2_6.pdp'
    instance,_=convert_li_lim(original,robots=10,requirement_seed=0,expected_tasks=100)
    target=OUT/(instance.name+'.json')
    instance.save(target)
    from pdpcr.li_lim import prefix_instance
    demo50=prefix_instance(instance,count=50,robots=10,
                           name='demo_LR2_2_6_n50_m10_req0_first50')
    demo50.save(OUT/(demo50.name+'.json'))
    return target,OUT/(demo50.name+'.json')


def main():
    p=argparse.ArgumentParser(description='Download and convert Li-Lim instances for PDP-CR pilot')
    p.add_argument('--sizes',type=int,nargs='+',choices=SIZES,default=list(SIZES))
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=['LC1','LR1','LRC1'])
    p.add_argument('--robots',type=int,default=10)
    p.add_argument('--req-seed',type=int,default=0)
    p.add_argument('--offline',action='store_true',help='Use only locally cached .txt/.pdp sources')
    p.add_argument('--demo',action='store_true',help='Only prepare local bundled demonstration instances')
    a=p.parse_args()
    if a.demo:
        print(make_local_demo())
    else:
        prepare(a.sizes,a.families,a.robots,a.req_seed,not a.offline)

if __name__=='__main__':
    main()
