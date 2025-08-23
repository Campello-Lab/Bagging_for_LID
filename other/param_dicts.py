param_dicts_compare_mle1_k = {'dataset_name': all,
                              'n': 2500,
                              'lid': None,
                              'dim': None,
                              'estimator_name': 'mle',
                              'bagging_method': [None, 'bag'],
                              'submethod_0': '0',
                              'submethod_error': 'diff',
                              'k': k_progression,
                              'sr': 0.3,
                              'Nbag': 10,
                              'pre_smooth': False,
                              'post_smooth': False,
                              't': 1}

param_dicts_compare_mada1_k = {'dataset_name': all,
                               'n': 2500,
                               'lid': None,
                               'dim': None,
                               'estimator_name': 'mada',
                               'bagging_method': [None, 'bag'],
                               'submethod_0': '0',
                               'submethod_error': 'diff',
                               'k': k_progression,
                               'sr': 0.3,
                               'Nbag': 10,
                               'pre_smooth': False,
                               'post_smooth': False,
                               't': 1}

param_dicts_compare_tle1_k = {'dataset_name': all,
                              'n': 2500,
                              'lid': None,
                              'dim': None,
                              'estimator_name': 'tle',
                              'bagging_method': [None, 'bag'],
                              'submethod_0': '0',
                              'submethod_error': 'diff',
                              'k': k_progression,
                              'sr': 0.3,
                              'Nbag': 10,
                              'pre_smooth': False,
                              'post_smooth': False,
                              't': 1}

param_dicts_compare_mle2_k = {'dataset_name': all,
                              'n': 2500,
                              'lid': None,
                              'dim': None,
                              'estimator_name': 'mle',
                              'bagging_method': ['bagw', 'bagwth'],
                              'submethod_0': '0',
                              'submethod_error': ['diff', 'log_diff'],
                              'k': k_progression,
                              'sr': 0.3,
                              'Nbag': 10,
                              'pre_smooth': False,
                              'post_smooth': False,
                              't': 1}

param_dicts_compare_mada2_k = {'dataset_name': all,
                               'n': 2500,
                               'lid': None,
                               'dim': None,
                               'estimator_name': 'mada',
                               'bagging_method': ['bagw', 'bagwth'],
                               'submethod_0': '0',
                               'submethod_error': ['diff', 'log_diff'],
                               'k': k_progression,
                               'sr': 0.3,
                               'Nbag': 10,
                               'pre_smooth': False,
                               'post_smooth': False,
                               't': 1}

param_dicts_compare_tle2_k = {'dataset_name': all,
                              'n': 2500,
                              'lid': None,
                              'dim': None,
                              'estimator_name': 'tle',
                              'bagging_method': ['bagw', 'bagwth'],
                              'submethod_0': '0',
                              'submethod_error': ['diff', 'log_diff'],
                              'k': k_progression,
                              'sr': 0.3,
                              'Nbag': 10,
                              'pre_smooth': False,
                              'post_smooth': False,
                              't': 1}

param_dicts_compare_mle_k = [param_dicts_compare_mle1_k, param_dicts_compare_mle2_k]
param_dicts_compare_mada_k = [param_dicts_compare_mada1_k, param_dicts_compare_mada2_k]
param_dicts_compare_tle_k = [param_dicts_compare_tle1_k, param_dicts_compare_tle2_k]

###########################################################################################################

param_dicts_compare_mle1_sr = {'dataset_name': all,
                               'n': 2500,
                               'lid': None,
                               'dim': None,
                               'estimator_name': 'mle',
                               'bagging_method': [None, 'bag'],
                               'submethod_0': '0',
                               'submethod_error': 'diff',
                               'k': 10,
                               'sr': sr_progression,
                               'Nbag': 10,
                               'pre_smooth': False,
                               'post_smooth': False,
                               't': 1}

param_dicts_compare_mada1_sr = {'dataset_name': all,
                                'n': 2500,
                                'lid': None,
                                'dim': None,
                                'estimator_name': 'mada',
                                'bagging_method': [None, 'bag'],
                                'submethod_0': '0',
                                'submethod_error': 'diff',
                                'k': 10,
                                'sr': sr_progression,
                                'Nbag': 10,
                                'pre_smooth': False,
                                'post_smooth': False,
                                't': 1}

param_dicts_compare_tle1_sr = {'dataset_name': all,
                               'n': 2500,
                               'lid': None,
                               'dim': None,
                               'estimator_name': 'tle',
                               'bagging_method': [None, 'bag'],
                               'submethod_0': '0',
                               'submethod_error': 'diff',
                               'k': 10,
                               'sr': sr_progression,
                               'Nbag': 10,
                               'pre_smooth': False,
                               'post_smooth': False,
                               't': 1}

param_dicts_compare_mle2_sr = {'dataset_name': all,
                               'n': 2500,
                               'lid': None,
                               'dim': None,
                               'estimator_name': 'mle',
                               'bagging_method': ['bagw', 'bagwth'],
                               'submethod_0': '0',
                               'submethod_error': ['diff', 'log_diff'],
                               'k': 10,
                               'sr': sr_progression,
                               'Nbag': 10,
                               'pre_smooth': False,
                               'post_smooth': False,
                               't': 1}

param_dicts_compare_mada2_sr = {'dataset_name': all,
                                'n': 2500,
                                'lid': None,
                                'dim': None,
                                'estimator_name': 'mada',
                                'bagging_method': ['bagw', 'bagwth'],
                                'submethod_0': '0',
                                'submethod_error': ['diff', 'log_diff'],
                                'k': 10,
                                'sr': sr_progression,
                                'Nbag': 10,
                                'pre_smooth': False,
                                'post_smooth': False,
                                't': 1}

param_dicts_compare_tle2_sr = {'dataset_name': all,
                               'n': 2500,
                               'lid': None,
                               'dim': None,
                               'estimator_name': 'tle',
                               'bagging_method': ['bagw', 'bagwth'],
                               'submethod_0': '0',
                               'submethod_error': ['diff', 'log_diff'],
                               'k': 10,
                               'sr': sr_progression,
                               'Nbag': 10,
                               'pre_smooth': False,
                               'post_smooth': False,
                               't': 1}

param_dicts_compare_mle_sr = [param_dicts_compare_mle1_sr, param_dicts_compare_mle2_sr]
param_dicts_compare_mada_sr = [param_dicts_compare_mada1_sr, param_dicts_compare_mada2_sr]
param_dicts_compare_tle_sr = [param_dicts_compare_tle1_sr, param_dicts_compare_tle2_sr]

######################################################################################################################

param_dicts_compare_mle1_lid = {'dataset_name': 'uniform',
                                'n': 2500,
                                'lid': lid_progression,
                                'dim': 40,
                                'estimator_name': 'mle',
                                'bagging_method': [None, 'bag'],
                                'submethod_0': '0',
                                'submethod_error': 'diff',
                                'k': 10,
                                'sr': 0.3,
                                'Nbag': 10,
                                'pre_smooth': False,
                                'post_smooth': False,
                                't': 1}

param_dicts_compare_mada1_lid = {'dataset_name': 'uniform',
                                 'n': 2500,
                                 'lid': lid_progression,
                                 'dim': 40,
                                 'estimator_name': 'mada',
                                 'bagging_method': [None, 'bag'],
                                 'submethod_0': '0',
                                 'submethod_error': 'diff',
                                 'k': 10,
                                 'sr': 0.3,
                                 'Nbag': 10,
                                 'pre_smooth': False,
                                 'post_smooth': False,
                                 't': 1}

param_dicts_compare_tle1_lid = {'dataset_name': 'uniform',
                                'n': 2500,
                                'lid': lid_progression,
                                'dim': 40,
                                'estimator_name': 'tle',
                                'bagging_method': [None, 'bag'],
                                'submethod_0': '0',
                                'submethod_error': 'diff',
                                'k': 10,
                                'sr': 0.3,
                                'Nbag': 10,
                                'pre_smooth': False,
                                'post_smooth': False,
                                't': 1}

param_dicts_compare_mle2_lid = {'dataset_name': 'uniform',
                                'n': 2500,
                                'lid': lid_progression,
                                'dim': 40,
                                'estimator_name': 'mle',
                                'bagging_method': ['bagw', 'bagwth'],
                                'submethod_0': '0',
                                'submethod_error': ['diff', 'log_diff'],
                                'k': 10,
                                'sr': 0.3,
                                'Nbag': 10,
                                'pre_smooth': False,
                                'post_smooth': False,
                                't': 1}

param_dicts_compare_mada2_lid = {'dataset_name': 'uniform',
                                 'n': 2500,
                                 'lid': lid_progression,
                                 'dim': 40,
                                 'estimator_name': 'mada',
                                 'bagging_method': ['bagw', 'bagwth'],
                                 'submethod_0': '0',
                                 'submethod_error': ['diff', 'log_diff'],
                                 'k': 10,
                                 'sr': 0.3,
                                 'Nbag': 10,
                                 'pre_smooth': False,
                                 'post_smooth': False,
                                 't': 1}

param_dicts_compare_tle2_lid = {'dataset_name': 'uniform',
                                'n': 2500,
                                'lid': lid_progression,
                                'dim': 40,
                                'estimator_name': 'tle',
                                'bagging_method': ['bagw', 'bagwth'],
                                'submethod_0': '0',
                                'submethod_error': ['diff', 'log_diff'],
                                'k': 10,
                                'sr': 0.3,
                                'Nbag': 10,
                                'pre_smooth': False,
                                'post_smooth': False,
                                't': 1}

param_dicts_compare_mle_lid = [param_dicts_compare_mle1_lid, param_dicts_compare_mle2_lid]
param_dicts_compare_mada_lid = [param_dicts_compare_mada1_lid, param_dicts_compare_mada2_lid]
param_dicts_compare_tle_lid = [param_dicts_compare_tle1_lid, param_dicts_compare_tle2_lid]

##################################################################################################################

param_dicts_compare_mle_k_sm = {'dataset_name': all,
                                'n': 2500,
                                'lid': None,
                                'dim': None,
                                'estimator_name': 'mle',
                                'bagging_method': [None, 'bag'],
                                'submethod_0': '0',
                                'submethod_error': 'diff',
                                'k': k_progression,
                                'sr': 0.3,
                                'Nbag': 10,
                                'pre_smooth': [True, False],
                                'post_smooth': [True, False],
                                't': 1}

param_dicts_compare_mada_k_sm = {'dataset_name': all,
                                 'n': 2500,
                                 'lid': None,
                                 'dim': None,
                                 'estimator_name': 'mada',
                                 'bagging_method': [None, 'bag'],
                                 'submethod_0': '0',
                                 'submethod_error': 'diff',
                                 'k': k_progression,
                                 'sr': 0.3,
                                 'Nbag': 10,
                                 'pre_smooth': [True, False],
                                 'post_smooth': [True, False],
                                 't': 1}

param_dicts_compare_tle_k_sm = {'dataset_name': all,
                                'n': 2500,
                                'lid': None,
                                'dim': None,
                                'estimator_name': 'tle',
                                'bagging_method': [None, 'bag'],
                                'submethod_0': '0',
                                'submethod_error': 'diff',
                                'k': k_progression,
                                'sr': 0.3,
                                'Nbag': 10,
                                'pre_smooth': [True, False],
                                'post_smooth': [True, False],
                                't': 1}

###########################################################################################################

param_dicts_compare_mle_sr_sm = {'dataset_name': all,
                                 'n': 2500,
                                 'lid': None,
                                 'dim': None,
                                 'estimator_name': 'mle',
                                 'bagging_method': [None, 'bag'],
                                 'submethod_0': '0',
                                 'submethod_error': 'diff',
                                 'k': 10,
                                 'sr': sr_progression,
                                 'Nbag': 10,
                                 'pre_smooth': [True, False],
                                 'post_smooth': [True, False],
                                 't': 1}

param_dicts_compare_mada_sr_sm = {'dataset_name': all,
                                  'n': 2500,
                                  'lid': None,
                                  'dim': None,
                                  'estimator_name': 'mada',
                                  'bagging_method': [None, 'bag'],
                                  'submethod_0': '0',
                                  'submethod_error': 'diff',
                                  'k': 10,
                                  'sr': sr_progression,
                                  'Nbag': 10,
                                  'pre_smooth': [True, False],
                                  'post_smooth': [True, False],
                                  't': 1}

param_dicts_compare_tle_sr_sm = {'dataset_name': all,
                                 'n': 2500,
                                 'lid': None,
                                 'dim': None,
                                 'estimator_name': 'tle',
                                 'bagging_method': [None, 'bag'],
                                 'submethod_0': '0',
                                 'submethod_error': 'diff',
                                 'k': 10,
                                 'sr': sr_progression,
                                 'Nbag': 10,
                                 'pre_smooth': [True, False],
                                 'post_smooth': [True, False],
                                 't': 1}

######################################################################################################################

param_dicts_compare_mle_lid_sm = {'dataset_name': 'uniform',
                                  'n': 2500,
                                  'lid': lid_progression,
                                  'dim': 40,
                                  'estimator_name': 'mle',
                                  'bagging_method': [None, 'bag'],
                                  'submethod_0': '0',
                                  'submethod_error': 'diff',
                                  'k': 10,
                                  'sr': 0.3,
                                  'Nbag': 10,
                                  'pre_smooth': [True, False],
                                  'post_smooth': [True, False],
                                  't': 1}

param_dicts_compare_mada_lid_sm = {'dataset_name': 'uniform',
                                   'n': 2500,
                                   'lid': lid_progression,
                                   'dim': 40,
                                   'estimator_name': 'mada',
                                   'bagging_method': [None, 'bag'],
                                   'submethod_0': '0',
                                   'submethod_error': 'diff',
                                   'k': 10,
                                   'sr': 0.3,
                                   'Nbag': 10,
                                   'pre_smooth': [True, False],
                                   'post_smooth': [True, False],
                                   't': 1}

param_dicts_compare_tle_lid_sm = {'dataset_name': 'uniform',
                                  'n': 2500,
                                  'lid': lid_progression,
                                  'dim': 40,
                                  'estimator_name': 'tle',
                                  'bagging_method': [None, 'bag'],
                                  'submethod_0': '0',
                                  'submethod_error': 'diff',
                                  'k': 10,
                                  'sr': 0.3,
                                  'Nbag': 10,
                                  'pre_smooth': [True, False],
                                  'post_smooth': [True, False],
                                  't': 1}

param_dicts_compare_mle_k
param_dicts_compare_mada_k
param_dicts_compare_tle_k
param_dicts_compare_mle_sr
param_dicts_compare_mada_sr
param_dicts_compare_tle_sr
param_dicts_compare_mle_lid
param_dicts_compare_mada_lid
param_dicts_compare_tle_lid

param_dicts_compare_mle_k_sm
param_dicts_compare_mada_k_sm
param_dicts_compare_tle_k_sm
param_dicts_compare_mle_sr_sm
param_dicts_compare_mada_sr_sm
param_dicts_compare_tle_sr_sm
param_dicts_compare_mle_lid_sm
param_dicts_compare_mada_lid_sm
param_dicts_compare_tle_lid_sm