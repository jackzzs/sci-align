#!/usr/bin/env python

# Copyright Zhesheng Zhou, Zhejiang University
# Email: zhouzzs@zju.edu.cn

import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from concurrent import futures

from whoswho import who
from pypinyin import pinyin
from pinyinsplit import PinyinSplit
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn

pys = PinyinSplit()

# configs
cpus = 12
cites_dir = "./data/cites"
name_list_files = {
    "Chinese Academicians": "./data/chinese_academicians.txt",
    "Foreign Academicians": "./data/foreign_academicians.txt",
    "Foreign Academicians (Chinese)": "./data/foreign_academicians_chinese.txt",
}

def parse(file):
    results = []
    df = pd.read_excel(file, header=None)
    dfs = np.split(df, df[df.isnull().all(1)].index)
    for dfc in dfs:
        result = {}
        dfc[0].fillna(method="pad", inplace=True)
        # process authors
        dfn = dfc.loc[dfc[0] == "AU"].iloc[:, 1:3]
        dfnf = dfn.iloc[:, 0] + dfn.iloc[:, 1]
        result["AU"] = dfnf.dropna().to_list()
        # process title
        titles_segs = []
        dft = dfc.loc[dfc[0] == "TI"].iloc[:, 1:3]
        for _, r in dft.iterrows():
            titles_segs.extend(r.dropna().to_list())
        result["TI"] = " ".join([s.strip() for s in titles_segs if isinstance(s, str)])
        if not result["AU"] or not result["TI"]:
            continue
        else:
            result["CI"] = file
            results.append(result)
    return results


def is_all_chinese(s):
    for _char in s:
        if not "\u4e00" <= _char <= "\u9fa5":
            return False
    return True


def mutate_comb(comb):
    results = [" ".join(comb).title()]
    if len(comb) == 2:
        results.append(f"{comb[1]} {comb[0]}".title())
    elif len(comb) == 3:
        results.append(f"{comb[1]} {comb[2]} {comb[0]}".title())
        results.append(f"{comb[1]}{comb[2]} {comb[0]}".title())
        results.append(f"{comb[0]} {comb[1]}{comb[2]}".title())
    elif len(comb) == 4:
        results.append(f"{comb[2]} {comb[3]} {comb[0]} {comb[1]}".title())
        results.append(f"{comb[0]}{comb[1]} {comb[2]}{comb[3]} ".title())
    return results


def get_alternatives(s):
    if not isinstance(s, str):
        raise ValueError(f"'{s}' is not a valid string to get alternatives.")
    results = [s]
    words = s.split()
    if is_all_chinese(s):
        combs = pinyin(s, heteronym=True, style=0)
        for comb in itertools.product(*combs):
            results.extend(mutate_comb(comb))
    elif len(words) == 2:
        splits = [pys.split(x) for x in reversed(words)]
        if all(splits):
            for comb in itertools.product(*splits):
                comb = [item for sublist in comb for item in sublist]
                results.extend(mutate_comb(comb))
    if words[0].isupper():
        result = ""
        for ss in words[0]:
            result += f"{ss}. "
        result += " ".join(words[1:])
        results.append(result)
    return results


def pinyin_sanity_check(a, b):
    a = "".join(filter(str.isalpha, a.lower()))
    b = "".join(filter(str.isalpha, b.lower()))
    a = [set(x) for x in pys.split(a)]
    b = [set(x) for x in pys.split(b)]
    if not a or not b:
        return True
    for ai in a:
        if ai in b:
            return True
    return False


def align(ns, name_lists):
    results = []
    for n in ns:
        for nln, nlf in name_lists.items():
            for nf in nlf:
                alters = itertools.product(get_alternatives(n), get_alternatives(nf))
                if any((who.match(*args) and pinyin_sanity_check(*args) for args in alters)):
                    results.append((n, nln, nf))
    return results

def main():
    name_lists = {}
    for nln, nlf in name_list_files.items():
        with open(nlf, 'r') as f:
            name_lists[nln] = [line.rstrip('\n') for line in f]

    targets = list(Path(cites_dir).glob("*.xlsx"))
    results = []
    progress = Progress(
        *Progress.get_default_columns(), 
        TimeElapsedColumn(),
        SpinnerColumn(),
        "[progress.download]{task.completed}/{task.total}",
    )
    with progress:
        with futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            task1 = progress.add_task("Loading cite files...", total=len(targets))
            future_to_target = {executor.submit(parse, t): t for t in targets}
            future_to_input = {}
            for future in futures.as_completed(future_to_target):
                target = future_to_target[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f'Error occors when processing "{target}":\n\t{exc}')
                else:
                    for c in data:
                        future_to_input[executor.submit(align, c["AU"], name_lists)] = c
                finally:
                    progress.update(task1, advance=1)
            task2 = progress.add_task("Processing...", total=len(future_to_input))
            for future in futures.as_completed(future_to_input):
                inp = future_to_input[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f'Error occors when processing "{inp["TI"]}":\n\t{exc}')
                else:
                    for m in data:
                        results.append({
                            "Title": inp["TI"],
                            "Cited by": inp["CI"],
                            "Author": m[0],
                            "Author Match Catalog": m[1],
                            "Author Match": m[2],
                        })
                finally:
                    progress.update(task2, advance=1)
    pd.DataFrame(results).to_excel("results.xlsx")

if __name__ == '__main__':
    main()