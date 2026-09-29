"""Bind a dataset to the existing runner for one CLI process; never rewrite the model/trainer."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

from external_baselines.inpformer_btad import run as shared
from .data import Adapter


@contextmanager
def dataset_context(dataset):
    """Process-scoped CLI binding; restored for tests. Do not call concurrently in threads."""
    adapter=Adapter(dataset)
    values={name:getattr(adapter,name) for name in ('CATEGORIES','TITLE','PROTOCOL','find_root','records','image_records','load_gt')}
    values.update(COMPACT_FINAL=True, DATASET_NAME='MVTec AD' if dataset=='mvtec' else 'VisA',
                  FULL_MACRO_SCOPE=f'{dataset}_{len(adapter.CATEGORIES)}_category_equal_macro')
    previous={k:getattr(shared,k) for k in values}
    try:
        for k,v in values.items(): setattr(shared,k,v)
        yield adapter
    finally:
        for k,v in previous.items(): setattr(shared,k,v)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset',choices=('mvtec','visa'),required=True)
    args,remaining=parser.parse_known_args()
    with dataset_context(args.dataset):
        stage=shared.parser().parse_args(remaining)
        if stage.workers<0: raise ValueError('workers must be nonnegative')
        shared.__dict__[stage.command.replace('-','_')](stage)


if __name__=='__main__': main()
