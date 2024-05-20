"""Evaluate estimators (Naru or others) on queries."""
import argparse
import ast
import collections
import glob
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import torch

import common
import datasets
import estimators as estimators_lib
import made
import transformer

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', DEVICE)

parser = argparse.ArgumentParser()

parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')

parser.add_argument('--num-queries', type=int, default=20, help='# queries.')
parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument('--psample',
                    type=int,
                    default=2000,
                    help='# of progressive samples to use per query.')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Turn on wildcard skipping.  Requires checkpoints be trained with ' \
         'column masking.')
parser.add_argument('--order',
                    nargs='+',
                    type=int,
                    help='Use a specific order?')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=128,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order' \
         'lists natural indices, e.g., [0 2 1] means variable 2 appears second.' \
         'MADE, however, is implemented to take in an argument the inverse ' \
         'semantics (element i indicates the position of variable i).  Transformer' \
         ' does not have this issue and thus should not have this flag on.')
parser.add_argument(
    '--input-encoding',
    type=str,
    default='binary',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='one_hot',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
         'then input encoding should be set to embed as well.')

# Transformer.
parser.add_argument(
    '--heads',
    type=int,
    default=0,
    help='Transformer: num heads.  A non-zero value turns on Transformer' \
         ' (otherwise MADE/ResMADE).'
)
parser.add_argument('--blocks',
                    type=int,
                    default=2,
                    help='Transformer: num blocks.')
parser.add_argument('--dmodel',
                    type=int,
                    default=32,
                    help='Transformer: d_model.')
parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
parser.add_argument('--transformer-act',
                    type=str,
                    default='gelu',
                    help='Transformer activation.')

# Estimators to enable.
parser.add_argument('--run-sampling',
                    action='store_true',
                    help='Run a materialized sampler?')
parser.add_argument('--run_maxdiff',
                    action='store_true',
                    help='Run the MaxDiff histogram?')
parser.add_argument('--run-bn',
                    action='store_true',
                    help='Run Bayes nets? If enabled, run BN only.')

# Bayes nets.
parser.add_argument('--bn-samples',
                    type=int,
                    default=200,
                    help='# samples for each BN inference.')
parser.add_argument('--bn-root',
                    type=int,
                    default=0,
                    help='Root variable index for chow liu tree.')
# Maxdiff
parser.add_argument(
    '--maxdiff-limit',
    type=int,
    default=30000,
    help='Maximum number of partitions of the Maxdiff histogram.')

args = parser.parse_args()


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeTable():
    assert args.dataset in ['dmv-tiny', 'dmv', 'badges', 'comments', 'postHistory', 'postLinks', 'posts', 'tags',
                            'users', 'votes']
    if args.dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif args.dataset == 'dmv':
        table = datasets.LoadDmv()

    oracle_est = estimators_lib.Oracle(table)
    if args.run_bn:
        return table, common.TableDataset(table), oracle_est
    return table, None, oracle_est


def MakeTableStats():
    assert args.dataset in ['badges', 'comments', 'postHistory', 'postLinks', 'posts', 'tags', 'users', 'votes']
    if args.dataset == 'badges':
        table = datasets.LoadStats_badges()
    elif args.dataset == 'comments':
        table = datasets.LoadStats_comments()
    elif args.dataset == 'postHistory':
        table = datasets.LoadStats_postHistory()
    elif args.dataset == 'postLinks':
        table = datasets.LoadStats_postLinks()
    elif args.dataset == 'posts':
        table = datasets.LoadStats_posts()
    elif args.dataset == 'tags':
        table = datasets.LoadStats_tags()
    elif args.dataset == 'users':
        table = datasets.LoadStats_users()
    elif args.dataset == 'votes':
        table = datasets.LoadStats_votes()

    return table


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          return_col_idx=False):
    s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values

    if args.dataset in ['dmv', 'dmv-tiny']:
        # Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def GenerateQuery(all_cols, rng, table, return_col_idx=False):
    """Generate a random query."""
    num_filters = rng.randint(5, 12)
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            return_col_idx=return_col_idx)
    return cols, ops, vals


def Query(estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query
    print("query is", query)

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    card = oracle_est.Query(cols, ops,
                            vals) if oracle_card is None else oracle_card
    if card == 0:
        return

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    pprint('): ', end='')

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        est_card = est.Query(cols, ops, vals)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()


def Query_stats(estimators,
                card,
                query=None,
                table=None,
                oracle_est=None):
    assert query is not None
    cols, ops, vals = query
    print('cols are', cols)
    print('ops are', ops)
    print('vals are', vals)
    print("query is", query)

    def pprint(*args, **kwargs):
        print(*args, **kwargs)

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c, o, float(v)), end='')
    pprint('): ', end='')

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        if est.table == table:
            est_card = est.Query(cols, ops, vals)
            err = ErrorMetric(est_card, card)
            est.AddError(err, est_card, card)
            pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
            break
    pprint()


def ReportEsts(estimators):
    v = -1
    for est in estimators:
        print(est.name, 'max', np.max(est.errs), '99th',
              np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
              'median', np.quantile(est.errs, 0.5))
        v = max(v, np.max(est.errs))
    return v


def RunN(table,
         cols,
         estimators,
         rng=None,
         num=20,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None):
    if rng is None:
        rng = np.random.RandomState(1234)

    last_time = None
    for i in range(num):
        do_print = False
        if i % log_every == 0:
            if last_time is not None:
                print('{:.1f} queries/sec'.format(log_every /
                                                  (time.time() - last_time)))
            do_print = True
            print('Query {}:'.format(i), end=' ')
            last_time = time.time()
        query = GenerateQuery(cols, rng, table)
        Query(estimators,
              do_print,
              oracle_card=oracle_cards[i]
              if oracle_cards is not None and i < len(oracle_cards) else None,
              query=query,
              table=table,
              oracle_est=oracle_est)

        max_err = ReportEsts(estimators)
    return False


def RunN_stats(table,
               cols,
               estimators,
               queries_file_name,
               rng=None):
    with open(queries_file_name) as f:
        queries = f.readlines()

    for query_no, query_str in enumerate(queries):
        cardinality_true = int(query_str.split("||")[-1])
        query_str = query_str.split("||")[0][:-1]
        table_name = query_str.split(" ")[3]
        if table_name != table.name:
            continue
        query = parse_query_single_table(query_str, table, cols)

        begin_time = time.time()
        Query_stats(estimators,
                    cardinality_true,
                    query=query,
                    table=table)
        end_time = time.time()
        print('latency is {} ms'.format((end_time - begin_time) * 1000))
    max_err = ReportEsts(estimators)
    return False


def parse_query_single_table(query, table, all_cols):
    """
    将SQL语句转换为 cols, ops, vals
    """
    attrs = []
    cols = []
    opss = []
    vals = []
    useful = query.split(' WHERE ')[-1].strip()
    result = dict()
    if 'AND' not in useful:
        # 无查询条件
        if 'SELECT' in useful:
            print('all_cols', all_cols)
            return cols, opss, vals
        else:
            attr, ops, value = str_pattern_matching(useful.strip())
            if "Date" in attr:
                assert "::timestamp" in value
                value = timestamp_transform(value.strip().split("::timestamp")[0])
            attr = attr.split('.')[-1]

            opss.append(ops)
            vals.append(value)

            for _ in all_cols:
                if _.name in attrs:
                    cols.append(_)
            cols = np.array(cols)
            print('cols', cols)
            return cols, opss, vals
    else:
        for sub_query in useful.split(' AND '):
            print("sub_query is ", sub_query)
            attr, ops, value = str_pattern_matching(sub_query.strip())
            if "Date" in attr:
                assert "::timestamp" in value
                value = timestamp_transform(value.strip().split("::timestamp")[0])
            attr = attr.split('.')[-1]
            attrs.append(attr)
            opss.append(ops)
            vals.append(value)

    for _ in all_cols:
        if _.name in attrs:
            cols.append(_)
    cols = np.array(cols)
    print('cols', cols)
    return cols, opss, vals


def str_pattern_matching(s):
    # split the string "attr==value" to ('attr', '=', 'value')
    op_start = 0
    if len(s.split(' IN ')) != 1:
        s = s.split(' IN ')
        attr = s[0].strip()
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            temp_value = s[1].strip()[1:][:-1].split(',')
            value = []
            for v in temp_value:
                value.append(v.strip())
        return attr, 'in', value

    for i in range(len(s)):
        if s[i] in OPS:
            op_start = i
            if i + 1 < len(s) and s[i + 1] in OPS:
                op_end = i + 1
            else:
                op_end = i
            break
    attr = s[:op_start]
    value = s[(op_end + 1):].strip()
    ops = s[op_start:(op_end + 1)]
    try:
        value = list(ast.literal_eval(s[1].strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value
    return attr.strip(), ops.strip(), value


def timestamp_transform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "'%Y-%m-%d %H:%M:%S'")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def RunNParallel(estimator_factory,
                 parallelism=2,
                 rng=None,
                 num=20,
                 num_filters=11,
                 oracle_cards=None):
    """RunN in parallel with Ray.  Useful for slow estimators e.g., BN."""
    import ray
    ray.init(redis_password='xxx')

    @ray.remote
    class Worker(object):

        def __init__(self, i):
            self.estimators, self.table, self.oracle_est = estimator_factory()
            self.columns = np.asarray(self.table.columns)
            self.i = i

        def run_query(self, query, j):
            col_idxs, ops, vals = pickle.loads(query)
            Query(self.estimators,
                  do_print=True,
                  oracle_card=oracle_cards[j]
                  if oracle_cards is not None else None,
                  query=(self.columns[col_idxs], ops, vals),
                  table=self.table,
                  oracle_est=self.oracle_est)

            print('=== Worker {}, Query {} ==='.format(self.i, j))
            for est in self.estimators:
                est.report()

        def get_stats(self):
            return [e.get_stats() for e in self.estimators]

    print('Building estimators on {} workers'.format(parallelism))
    workers = []
    for i in range(parallelism):
        workers.append(Worker.remote(i))

    print('Building estimators on driver')
    estimators, table, _ = estimator_factory()
    cols = table.columns

    if rng is None:
        rng = np.random.RandomState(1234)
    queries = []
    for i in range(num):
        col_idxs, ops, vals = GenerateQuery(cols,
                                            rng,
                                            table=table,
                                            return_col_idx=True)
        queries.append((col_idxs, ops, vals))

    cnts = 0
    for i in range(num):
        query = queries[i]
        print('Queueing execution of query', i)
        workers[i % parallelism].run_query.remote(pickle.dumps(query), i)

    print('Waiting for queries to finish')
    stats = ray.get([w.get_stats.remote() for w in workers])

    print('Merging and printing final results')
    for stat_set in stats:
        for e, s in zip(estimators, stat_set):
            e.merge_stats(s)
    time.sleep(1)

    print('=== Merged stats ===')
    for est in estimators:
        est.report()
    return estimators


def MakeBnEstimators():
    table, train_data, oracle_est = MakeTable()
    estimators = [
        estimators_lib.BayesianNetwork(train_data,
                                       args.bn_samples,
                                       'chow-liu',
                                       topological_sampling_order=True,
                                       root=args.bn_root,
                                       max_parents=2,
                                       use_pgm=False,
                                       discretize=100,
                                       discretize_method='equal_freq')
    ]

    for est in estimators:
        est.name = str(est)
    return estimators, table, oracle_est


def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
                     args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))
    if return_df:
        return results
    results.to_csv(path, index=False)


def LoadOracleCardinalities():
    ORACLE_CARD_FILES = {
        'dmv': 'datasets/dmv-2000queries-oracle-cards-seed1234.csv'
    }
    path = ORACLE_CARD_FILES.get(args.dataset, None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 2000, len(df)
        return df.values.reshape(-1)
    return None


def Main():
    queries_file = ('/home/lrr/Documents/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries'
                    '/stats_CEB_single_table_sub_query.sql')
    all_ckpts = glob.glob('./models/{}'.format(args.glob))
    if args.blacklist:
        all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]

    selected_ckpts = all_ckpts
    oracle_cards = LoadOracleCardinalities()
    print('ckpts', selected_ckpts)

    # tables of STATS
    tables = ['badges', 'comments', 'postHistory', 'postLinks', 'posts', 'tags', 'users', 'votes']

    # Estimators to run.
    if args.run_bn:
        estimators = RunNParallel(estimator_factory=MakeBnEstimators,
                                  parallelism=50,
                                  rng=np.random.RandomState(1234),
                                  num=args.num_queries,
                                  num_filters=None,
                                  oracle_cards=oracle_cards)
    else:
        estimators = [
            # estimators_lib.ProgressiveSampling(c.loaded_model,
            #                                    table,
            #                                    args.psample,
            #                                    device=DEVICE,
            #                                    shortcircuit=args.column_masking)
            # for c in parsed_ckpts
        ]
        # for est, ckpt in zip(estimators, parsed_ckpts):
        #     est.name = str(est) + '_{}_{:.3f}'.format(ckpt.seed, ckpt.bits_gap)

        if args.inference_opts:
            print('Tracing forward_with_encoded_input()...')
            for est in estimators:
                encoded_input = est.model.EncodeInput(
                    torch.zeros(args.psample, est.model.nin, device=DEVICE))

                # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
                # The 1.2 version changes the API to
                # torch.jit.script(est.model) and requires an annotation --
                # which was found to be slower.
                est.traced_fwd = torch.jit.trace(
                    est.model.forward_with_encoded_input, encoded_input)

        # if args.run_sampling:
        #     SAMPLE_RATIO = {'dmv': [0.0013]}  # ~1.3MB.
        #     for p in SAMPLE_RATIO.get(args.dataset, [0.01]):
        #         estimators.append(estimators_lib.Sampling(table, p=p))

        table = MakeTableStats()
        cols_to_train = table.columns
        all_table = {}
        if args.run_maxdiff:

            # dmv
            estimators.append(
                estimators_lib.MaxDiffHistogram(table, args.maxdiff_limit))

            # stats
            # for t in tables:
            #     table = MakeTableStats(t)
            #     all_table[t] = table
            #     cols_to_train = table.columns
            #
            #     estimators.append(
            #         estimators_lib.MaxDiffHistogram(table, args.maxdiff_limit))
        # Other estimators can be appended as well.

        if len(estimators):
            # RunN(table,
            #      cols_to_train,
            #      estimators,
            #      rng=np.random.RandomState(1234),
            #      num=args.num_queries,
            #      log_every=1,
            #      num_filters=None,
            #      oracle_cards=oracle_cards,
            #      oracle_est=oracle_est)

            RunN_stats(table,
                       cols_to_train,
                       estimators,
                       queries_file)

    SaveEstimators(args.err_csv, estimators)
    print('...Done, result:', args.err_csv)


if __name__ == '__main__':
    Main()
