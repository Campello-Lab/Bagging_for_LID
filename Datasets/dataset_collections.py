#Used datasets in the paper
#-----------------------------------------------------------------------------------------------------------------------
dataset_name_strings = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M5b_Helix2d', 'M6_Nonlinear',
       'M7_Roll', 'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M11_Moebius',
       'M12_Norm', 'M13a_Scurve', 'Mn1_Nonlinear', 'Mn2_Nonlinear', 'lollipop_', 'uniform']

#-----------------------------------------------------------------------------------------------------------------------
#Old ones
keylist = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M5b_Helix2d', 'M6_Nonlinear', 'M7_Roll',
           'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M11_Moebius', 'M12_Norm',
           'M13a_Scurve',
           'Mn1_Nonlinear', 'Mn2_Nonlinear', 'lollipop_', 'swiss_roll', 'affine_1000D_100d_uniform',
           'affine_1000D_900d_gaussian', 'affine_1000D_900d_uniform', 'affine_2000D_100d_200d_uniform',
           'affine_800D_10d_80d_200d_uniform', 'affine_800D_200d_uniform', 'squiggly_05_freq_2000D_200d_uniform',
           'squiggly_10_freq_2000D_200d_uniform', 'squiggly_1_freq_2000D_200d_uniform',
           'squiggly_5_freq_2000D_200d_uniform', 'affine_100D_10d_25d_50d_gaussian', 'affine_100D_10d_25d_50d_uniform',
           'affine_100D_10d_30d_90d_uniform',
           'affine_100D_10d_uniform', 'affine_100D_30d_gaussian', 'affine_100D_30d_uniform', 'affine_100D_90d_gaussian',
           'affine_100D_90d_uniform', 'squiggly_05_freq_100D_10d_25d_50d_uniform', 'squiggly_05_freq_100D_50d_uniform',
           'squiggly_10_freq_100D_10d_25d_50d_uniform',
           'squiggly_10_freq_100D_50d_uniform', 'squiggly_1_freq_100D_10d_25d_50d_uniform',
           'squiggly_1_freq_100D_50d_uniform', 'squiggly_5_freq_100D_10d_25d_50d_uniform',
           'squiggly_5_freq_100D_50d_uniform', 'affine_10D_2d_4d_8d_gaussian',
           'affine_10D_2d_4d_8d_laplace', 'affine_10D_2d_4d_8d_uniform', 'affine_10D_5d_gaussian',
           'affine_10D_5d_laplace', 'affine_10D_5d_uniform',
           'squiggly_05_freq_10D_2d_4d_8d_uniform', 'squiggly_10_freq_10D_2d_4d_8d_uniform',
           'squiggly_1_freq_10D_2d_4d_8d_uniform', 'squiggly_5_freq_10D_2d_4d_8d_uniform', 'torus_circle']

original = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M5b_Helix2d', 'M6_Nonlinear',
            'M7_Roll', 'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M11_Moebius', 'M12_Norm',
            'M13a_Scurve', 'Mn1_Nonlinear', 'Mn2_Nonlinear']

dgm1 = ['affine_1000D_100d_uniform', 'affine_1000D_900d_gaussian', 'affine_1000D_900d_uniform',
        'affine_2000D_100d_200d_uniform', 'affine_800D_10d_80d_200d_uniform', 'affine_800D_200d_uniform',
        'squiggly_05_freq_2000D_200d_uniform', 'squiggly_10_freq_2000D_200d_uniform',
        'squiggly_1_freq_2000D_200d_uniform',
        'squiggly_5_freq_2000D_200d_uniform', 'affine_100D_10d_25d_50d_gaussian', 'affine_100D_10d_25d_50d_uniform',
        'affine_100D_10d_30d_90d_uniform',
        'affine_100D_10d_uniform', 'affine_100D_30d_gaussian', 'affine_100D_30d_uniform', 'affine_100D_90d_gaussian',
        'affine_100D_90d_uniform']

dgm2 = ['squiggly_05_freq_100D_10d_25d_50d_uniform', 'squiggly_05_freq_100D_50d_uniform',
        'squiggly_10_freq_100D_10d_25d_50d_uniform',
        'squiggly_10_freq_100D_50d_uniform', 'squiggly_1_freq_100D_10d_25d_50d_uniform',
        'squiggly_1_freq_100D_50d_uniform', 'squiggly_5_freq_100D_10d_25d_50d_uniform',
        'squiggly_5_freq_100D_50d_uniform', 'affine_10D_2d_4d_8d_gaussian',
        'affine_10D_2d_4d_8d_laplace', 'affine_10D_2d_4d_8d_uniform', 'affine_10D_5d_gaussian', 'affine_10D_5d_laplace',
        'affine_10D_5d_uniform',
        'squiggly_05_freq_10D_2d_4d_8d_uniform', 'squiggly_10_freq_10D_2d_4d_8d_uniform',
        'squiggly_1_freq_10D_2d_4d_8d_uniform', 'squiggly_5_freq_10D_2d_4d_8d_uniform', 'torus_circle']


interesting_low_dim = ['M7_Roll', 'M11_Moebius', 'M13a_Scurve', 'swiss_roll', 'torus_circle', 'M5b_Helix2d', 'lollipop_']

medium_uniform_lid = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M6_Nonlinear', 'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M12_Norm',
                      'Mn1_Nonlinear', 'Mn2_Nonlinear', 'affine_10D_5d_uniform', 'affine_10D_5d_laplace', 'affine_10D_5d_gaussian']

medium_lid_unions = ['affine_10D_2d_4d_8d_gaussian', 'affine_10D_2d_4d_8d_laplace', 'affine_10D_2d_4d_8d_uniform', 'squiggly_05_freq_10D_2d_4d_8d_uniform', 'squiggly_10_freq_10D_2d_4d_8d_uniform',
                     'squiggly_1_freq_10D_2d_4d_8d_uniform', 'squiggly_5_freq_10D_2d_4d_8d_uniform']

very_large_lid = ['affine_1000D_100d_uniform', 'affine_1000D_900d_gaussian','affine_1000D_900d_uniform','affine_2000D_100d_200d_uniform',
                  'affine_800D_10d_80d_200d_uniform', 'affine_800D_200d_uniform', 'squiggly_05_freq_2000D_200d_uniform','squiggly_10_freq_2000D_200d_uniform',
                  'squiggly_1_freq_2000D_200d_uniform', 'squiggly_5_freq_2000D_200d_uniform']

large_lid = ['affine_100D_10d_25d_50d_gaussian', 'affine_100D_10d_25d_50d_uniform', 'affine_100D_10d_30d_90d_uniform', 'affine_100D_10d_uniform', 'affine_100D_30d_gaussian',
             'affine_100D_30d_uniform', 'affine_100D_90d_gaussian', 'affine_100D_90d_uniform', 'squiggly_05_freq_100D_10d_25d_50d_uniform', 'squiggly_05_freq_100D_50d_uniform',
             'squiggly_10_freq_100D_10d_25d_50d_uniform', 'squiggly_10_freq_100D_50d_uniform', 'squiggly_1_freq_100D_10d_25d_50d_uniform', 'squiggly_1_freq_100D_50d_uniform',
             'squiggly_5_freq_100D_10d_25d_50d_uniform', 'squiggly_5_freq_100D_50d_uniform']