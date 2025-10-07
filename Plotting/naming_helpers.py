import pandas as pd

#!!Hard coded part!! This is for changing the full, class parameter based identifiers into the expressive naming conventiones used in the paper.
def unordered_lookup(query, original_map = None, sep= '|'):
    if original_map is None:
        original_map  =  {
    'bagging_method:bag | pre_smooth:False | post_smooth:False': 'Simple bagging',
    'bagging_method:bag | pre_smooth:False | post_smooth:True': 'Simple bagging with post-smoothing',
    'bagging_method:bag | pre_smooth:True | post_smooth:False': 'Simple bagging with pre-smoothing',
    'bagging_method:bag | pre_smooth:True | post_smooth:True': 'Simple bagging with pre-smoothing and post-smoothing',
    'bagging_method:None | pre_smooth:False | post_smooth:False': 'Baseline',
    'bagging_method:None | pre_smooth:False | post_smooth:True': 'Baseline with smoothing'}
    def build_canonical_map(original: dict[str, str], sep: str = '|') -> dict[tuple[str, ...], str]:
        return {
            tuple(sorted(part.strip() for part in key.split(sep))): value
            for key, value in original.items()
        }
    canonical_map = build_canonical_map(original_map)
    signature = tuple(sorted(part.strip() for part in query.split(sep)))
    return canonical_map.get(signature)

#!!Hard coded part!! This is for changing the full, class parameter based identifiers into the expressive naming conventiones used in the paper.
def modify_label(label):
    if label == 'bagging_method:bag':
        label = 'Simple bagging'
    elif label == 'bagging_method:bagw':
        label = 'Bagging with out-of-bag weights'
    elif label == 'bagging_method:bagwth':
        label = 'Bagging with out-of-bag weights (adjust)'
    elif label == 'bagging_method:approx_bagwth':
        label = 'Bagging with out-of-bag weights (approximate adjust)'
    elif label == 'bagging_method:None':
        label = 'Baseline'
    else:
        label = unordered_lookup(label)
    return label

#!!Hard coded part!! This if for reordering experiments, so that different plots get the same colored spider chart lines (based on method variants), and they are arranged in a logical order in the table, instead of random.
def reorder_sorted_experiments(df, order=None, keep_rest=True):
    order_mle = [(('Nbag', 10),
                  ('bagging_method', None),
                  ('estimator_name', 'mle'),
                  ('post_smooth', False),
                  ('pre_smooth', False),
                  ('submethod_0', '0'),
                  ('submethod_error', 'log_diff'),
                  ('t', 1)), (('Nbag', 10),
                              ('bagging_method', None),
                              ('estimator_name', 'mle'),
                              ('post_smooth', True),
                              ('pre_smooth', False),
                              ('submethod_0', '0'),
                              ('submethod_error', 'log_diff'),
                              ('t', 1)), (('Nbag', 10),
                                          ('bagging_method', 'bag'),
                                          ('estimator_name', 'mle'),
                                          ('post_smooth', False),
                                          ('pre_smooth', False),
                                          ('submethod_0', '0'),
                                          ('submethod_error', 'log_diff'),
                                          ('t', 1)), (('Nbag', 10),
                                                      ('bagging_method', 'bag'),
                                                      ('estimator_name', 'mle'),
                                                      ('post_smooth', True),
                                                      ('pre_smooth', False),
                                                      ('submethod_0', '0'),
                                                      ('submethod_error', 'log_diff'),
                                                      ('t', 1)), (('Nbag', 10),
                                                                  ('bagging_method', 'bag'),
                                                                  ('estimator_name', 'mle'),
                                                                  ('post_smooth', False),
                                                                  ('pre_smooth', True),
                                                                  ('submethod_0', '0'),
                                                                  ('submethod_error', 'log_diff'),
                                                                  ('t', 1)), (('Nbag', 10),
                                                                              ('bagging_method', 'bag'),
                                                                              ('estimator_name', 'mle'),
                                                                              ('post_smooth', True),
                                                                              ('pre_smooth', True),
                                                                              ('submethod_0', '0'),
                                                                              ('submethod_error', 'log_diff'),
                                                                              ('t', 1)), (('Nbag', 10),
                                                                                          ('bagging_method', 'bagw'),
                                                                                          ('estimator_name', 'mle'),
                                                                                          ('post_smooth', False),
                                                                                          ('pre_smooth', False),
                                                                                          ('submethod_0', '0'),
                                                                                          ('submethod_error',
                                                                                           'log_diff'),
                                                                                          ('t', 1)), (('Nbag', 10),
                                                                                                      ('bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                      ('estimator_name',
                                                                                                       'mle'),
                                                                                                      ('post_smooth',
                                                                                                       False),
                                                                                                      ('pre_smooth',
                                                                                                       False),
                                                                                                      ('submethod_0',
                                                                                                       '0'),
                                                                                                      (
                                                                                                      'submethod_error',
                                                                                                      'log_diff'),
                                                                                                      ('t', 1))]

    order_tle = [(('Nbag', 10),
                  ('bagging_method', None),
                  ('estimator_name', 'tle'),
                  ('post_smooth', False),
                  ('pre_smooth', False),
                  ('submethod_0', '0'),
                  ('submethod_error', 'log_diff'),
                  ('t', 1)), (('Nbag', 10),
                              ('bagging_method', None),
                              ('estimator_name', 'tle'),
                              ('post_smooth', True),
                              ('pre_smooth', False),
                              ('submethod_0', '0'),
                              ('submethod_error', 'log_diff'),
                              ('t', 1)), (('Nbag', 10),
                                          ('bagging_method', 'bag'),
                                          ('estimator_name', 'tle'),
                                          ('post_smooth', False),
                                          ('pre_smooth', False),
                                          ('submethod_0', '0'),
                                          ('submethod_error', 'log_diff'),
                                          ('t', 1)), (('Nbag', 10),
                                                      ('bagging_method', 'bag'),
                                                      ('estimator_name', 'tle'),
                                                      ('post_smooth', True),
                                                      ('pre_smooth', False),
                                                      ('submethod_0', '0'),
                                                      ('submethod_error', 'log_diff'),
                                                      ('t', 1)), (('Nbag', 10),
                                                                  ('bagging_method', 'bag'),
                                                                  ('estimator_name', 'tle'),
                                                                  ('post_smooth', False),
                                                                  ('pre_smooth', True),
                                                                  ('submethod_0', '0'),
                                                                  ('submethod_error', 'log_diff'),
                                                                  ('t', 1)), (('Nbag', 10),
                                                                              ('bagging_method', 'bag'),
                                                                              ('estimator_name', 'tle'),
                                                                              ('post_smooth', True),
                                                                              ('pre_smooth', True),
                                                                              ('submethod_0', '0'),
                                                                              ('submethod_error', 'log_diff'),
                                                                              ('t', 1)), (('Nbag', 10),
                                                                                          ('bagging_method', 'bagw'),
                                                                                          ('estimator_name', 'tle'),
                                                                                          ('post_smooth', False),
                                                                                          ('pre_smooth', False),
                                                                                          ('submethod_0', '0'),
                                                                                          ('submethod_error',
                                                                                           'log_diff'),
                                                                                          ('t', 1)), (('Nbag', 10),
                                                                                                      ('bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                      ('estimator_name',
                                                                                                       'tle'),
                                                                                                      ('post_smooth',
                                                                                                       False),
                                                                                                      ('pre_smooth',
                                                                                                       False),
                                                                                                      ('submethod_0',
                                                                                                       '0'),
                                                                                                      (
                                                                                                      'submethod_error',
                                                                                                      'log_diff'),
                                                                                                      ('t', 1))]

    order_mada = [(('Nbag', 10),
                   ('bagging_method', None),
                   ('estimator_name', 'mada'),
                   ('post_smooth', False),
                   ('pre_smooth', False),
                   ('submethod_0', '0'),
                   ('submethod_error', 'log_diff'),
                   ('t', 1)), (('Nbag', 10),
                               ('bagging_method', None),
                               ('estimator_name', 'mada'),
                               ('post_smooth', True),
                               ('pre_smooth', False),
                               ('submethod_0', '0'),
                               ('submethod_error', 'log_diff'),
                               ('t', 1)), (('Nbag', 10),
                                           ('bagging_method', 'bag'),
                                           ('estimator_name', 'mada'),
                                           ('post_smooth', False),
                                           ('pre_smooth', False),
                                           ('submethod_0', '0'),
                                           ('submethod_error', 'log_diff'),
                                           ('t', 1)), (('Nbag', 10),
                                                       ('bagging_method', 'bag'),
                                                       ('estimator_name', 'mada'),
                                                       ('post_smooth', True),
                                                       ('pre_smooth', False),
                                                       ('submethod_0', '0'),
                                                       ('submethod_error', 'log_diff'),
                                                       ('t', 1)), (('Nbag', 10),
                                                                   ('bagging_method', 'bag'),
                                                                   ('estimator_name', 'mada'),
                                                                   ('post_smooth', False),
                                                                   ('pre_smooth', True),
                                                                   ('submethod_0', '0'),
                                                                   ('submethod_error', 'log_diff'),
                                                                   ('t', 1)), (('Nbag', 10),
                                                                               ('bagging_method', 'bag'),
                                                                               ('estimator_name', 'mada'),
                                                                               ('post_smooth', True),
                                                                               ('pre_smooth', True),
                                                                               ('submethod_0', '0'),
                                                                               ('submethod_error', 'log_diff'),
                                                                               ('t', 1)), (('Nbag', 10),
                                                                                           ('bagging_method', 'bagw'),
                                                                                           ('estimator_name', 'mada'),
                                                                                           ('post_smooth', False),
                                                                                           ('pre_smooth', False),
                                                                                           ('submethod_0', '0'),
                                                                                           ('submethod_error',
                                                                                            'log_diff'),
                                                                                           ('t', 1)), (('Nbag', 10),
                                                                                                       (
                                                                                                       'bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                       (
                                                                                                       'estimator_name',
                                                                                                       'mada'),
                                                                                                       ('post_smooth',
                                                                                                        False),
                                                                                                       ('pre_smooth',
                                                                                                        False),
                                                                                                       ('submethod_0',
                                                                                                        '0'),
                                                                                                       (
                                                                                                       'submethod_error',
                                                                                                       'log_diff'),
                                                                                                       ('t', 1))]
    default_order = order_mle + order_tle + order_mada
    order = default_order if order is None else order

    ordered = [c for c in order if c in df.columns]
    the_rest = [c for c in df.columns if c not in order] if keep_rest else []

    # 👇 prevent pandas from interpreting the key as a 3D array
    key = pd.Index(ordered + the_rest, dtype=object)
    return df.loc[:, key]

def reassing_placeholder_value(experiments):
    def _get(exp, attr, default=None):
        return getattr(exp, attr, default)

    for i in range(len(experiments)):
        if _get(experiments[i], 'pre_smooth') is None:
            experiments[i].pre_smooth = False
        if _get(experiments[i], 'post_smooth') is None:
            experiments[i].post_smooth = False
    return experiments