import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau, spearmanr, cramervonmises, kstest, weibull_min

# *** TWO-NN LID (LOCAL INTRINSIC DIMENSIONALITY) ESTIMATOR ***:

def two_NN_LID_Estimator(data_array, subsample_indexes = None, *, neighbourhood_size = 10, perc_deleted = 0, return_smoothed = True, **kwargs):
    '''
    Perform Two-NN LID Estimation (optionally, w.r.t. a reference subsample)
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . subsample_indexes:  Row indexes of a subset of the dataset observations to be used as a reference subsample for computation of LIDs.
                              If subsample_indexes is None (default), then LIDs are computed in the conventional way, using the whole dataset.
                              Otherwise, the LID of each observation is computed w.r.t. the subsample only. Out of subsample observations also
                              have their LIDs computed, but they do NOT affect LIDs of any other observation (within or outside the subsample);
        . neighbourhood_size: Number of neighbours used to determine the local region around a query observation, within which the LID of
                              the query is computed by assuming the local distribution to be approximately uniform (neighbourhood_size >= 3);
        . perc_deleted:       Percentage of largest mu values to be discarded. The value suggested by the authors of the 2-NN (Global) ID
                              Estimator is 10%. Here, the default is zero, but the largest value is removed regardless, to avoid NaN/Inf;
        . return_smoothed:    Whether or not a smoothed version of the estimates (averaging across the neighbours) should be computed/returned.
      OUTPUT:
        . If return_smoothed == True: A single 2D numpy array concatenating: (i) the raw (non-smoothed) LID estimates [index 0 along axis 0 of
          the returned array]; and (ii) the smoothed LID estimates, where the LID of a query is the mean LID within the query's neighbourhood
          [index 1 along axis 0 of the returned array]. The returned array has shape (2, data_size);
        . If return_smoothed == False: A 1D numpy array of shape (data_size, ) with item (i) above only.
      NOTE:
        . This estimator does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function.
    '''
    # Preliminaries:
    data_size = data_array.shape[0] # Number of observations in the dataset
    if ( (subsample_indexes is None) and not (3 <= neighbourhood_size < data_size) ):
        print("-----------------------------------------------------------------------------------------------------------")
        print("!!! Abort - 2NN Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < data size !!!")
        print("-----------------------------------------------------------------------------------------------------------")
        return None
    elif ( (subsample_indexes is not None) and not (3 <= neighbourhood_size < len(subsample_indexes)) ):
        print("----------------------------------------------------------------------------------------------------------------")
        print("!!! Abort - 2NN Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < subsample size !!!")
        print("----------------------------------------------------------------------------------------------------------------")
        return None
    else: # Neighbourhood size properly provided
        neighbourhood_size = np.ceil(neighbourhood_size).astype(int) 
    if (0 < perc_deleted < 100):
        no_deleted = np.ceil((perc_deleted/100)*neighbourhood_size).astype(int) # No. of largest mu values to be discarded (guaranteed to be at least 1)
    else:
        no_deleted = 1 # We need to discard at least the largest mu value, which corresponds to F_emp = 1, causing log(1-F_emp) = -Inf
    # Compute LIDs:
    if subsample_indexes is None: # Compute LIDs w.r.t. the whole dataset
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size, algorithm='auto').fit(data_array) # Nearest Neighbours Oracle
        distances, KNN_indices = nbrs.kneighbors(data_array)    # Since query observations are provided as input, and they are all contained in the reference database used by the
                                                                # NN Oracle (actually, in this case, the query set and the reference database are exactly the same), then the 1st NN
                                                                # of each query observation is the observation itself (with dist = 0). Should the input argument had been ommitted,
                                                                # i.e, nbrs.kneighbors(), then the distances and indices of the neighbours of the reference data themselves would
                                                                # be similarly returned; however, in that case, each observation would not appear as its own neighbour in the result
        NN_ratio_mu = distances[:,2]/distances[:,1] # Ratio between the 2nd and 1st nearest neighbour distances, mu = r2/r1 (NB. 1st and 2nd NN excluded the query observation itself!)
        Two_NN_LID_estimates = np.zeros(data_size)  # Raw (non-smoothed) LID estimates to be computed and returned
        for obs in range(data_size):
            indices_mu_sorted = np.argsort(NN_ratio_mu[KNN_indices[obs]])   # Indices corresponding to the permutation that would sort the mu ratios for observations within the neighbourhood in ascending order.
                                                                            # NB. Since the NN indices here DO include the query observations themselves (as their own "neighbour"), the mu ratios within the
                                                                            # neighbourhood of each observation DO include the mu ratio for that observation    
            F_emp = np.zeros(neighbourhood_size)
            F_emp[indices_mu_sorted] = np.arange(1,neighbourhood_size+1)/neighbourhood_size # Empirical cumulative distribution (of NN_ratio_mu values for the subset of observations within the neighbourhood)
            indices_mu_filtered_out = indices_mu_sorted[-no_deleted:] # We need to discard at least the largest mu value, which corresponds to F_emp = 1, causing log(1-F_emp) = -Inf
            NN_ratio_mu_filtered = np.delete(NN_ratio_mu[KNN_indices[obs]], indices_mu_filtered_out)
            F_emp_filtered = np.delete(F_emp, indices_mu_filtered_out)
            x = np.log(NN_ratio_mu_filtered)[np.newaxis].T
            y = -np.log(1-F_emp_filtered)[np.newaxis].T
            reg = LinearRegression(fit_intercept = False).fit(x, y) # Regression with no intercept
            Two_NN_LID_estimates[obs] = reg.coef_ # Store the regression coefficient (slope) as a coarse LID estimate
        if return_smoothed == True:
            Two_NN_LID_smoothed_estimates = np.array([np.mean(Two_NN_LID_estimates[KNN_indices[obs]]) for obs in range(data_size)]) # LID smoothing: LID of query as the mean of the LIDs in the query's neighbourhood
    else: # subsample_indexes is not None, so, compute LIDs w.r.t. the subsample provided as input
        data_array_subs = data_array[subsample_indexes] # Data subsample
        mask_subsample = False*np.ones(data_size, dtype = bool)
        mask_subsample[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the subsample, False otherwise (out-of-subsample)
        out_of_subsample_indexes = np.array([x for x in range(data_size)])[mask_subsample == False] # Indexes of observations not contained in the subsample
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size, algorithm='auto').fit(data_array_subs) # Reference database for the NN Oracle is the subsample
        distances_insub, KNN_Sindices_insub = nbrs.kneighbors(data_array_subs)  # Since these query observations are in the reference database, then the
                                                                                # returned 1st NN of each observation is the observation itself (with dist = 0)
        distances_outsub, KNN_Sindices_outsub = nbrs.kneighbors(data_array[out_of_subsample_indexes]) # Since these query observations are NOT in the reference database, then the
                                                                                                      # returned 1st NN of each observation is NOT the observation itself
        NN_ratio_mu = np.zeros(data_size)
        NN_ratio_mu[subsample_indexes] = distances_insub[:,2]/distances_insub[:,1] # Ratio between the 2nd and 1st nearest neighbour distances, mu = r2/r1 (excluded the query observation itself)
        NN_ratio_mu[out_of_subsample_indexes] = distances_outsub[:,1]/distances_outsub[:,0] # Ratio between the 2nd and 1st nearest neighbour distances, mu = r2/r1 (query is NOT in the subsample)
        KNN_Sindices = np.zeros([data_size,neighbourhood_size]).astype(int) # Indexes for each row of the dataset, POINTING TO ROWS OF THE SUBSAMPLE (corresponding to the observation's NNs in the subsample, indexed as such)
        KNN_Sindices[subsample_indexes] = KNN_Sindices_insub # Notice that in this subset, which corresponds to the observations in the subsample, the first element of each row, corresponding to the
                                                             # 1st subsampled NN of the respective data observation, corresponds to the index/row of the observation itself in the subsample. These 
                                                             # elements (indexes) will not be removed, because the local region within the subsample and around a given query observation should NOT
                                                             # exclude that observation AS LONG AS THAT OBSERVATION IS PART OF THE SUBSAMPLE ITSELF 
        KNN_Sindices[out_of_subsample_indexes] = KNN_Sindices_outsub # This subset corresponds to the observations outside the subsample
        Two_NN_LID_estimates = np.zeros(data_size)
        for obs in range(data_size):
            KNN_data_indices_obs = subsample_indexes[KNN_Sindices[obs]] # Convert the list of subsample indexes corresponding to the within-subsample NN of the current query observation into dataset indexes  
            indices_mu_sorted = np.argsort(NN_ratio_mu[KNN_data_indices_obs]) # Permutation indices that would sort the mu ratios for observations within the subsampling neighbourhood in ascending order.
                                                                              # NB. Here, the NN indices do NOT include the query observations themselves, therefore, the mu ratios within the neighbourhood of each
                                                                              # observation do NOT include the mu ratio for that observation. This is in conformity with the fact that the query observation may have
                                                                              # NOT been considered when computing the mu ratios of the observations in its subsampling neighbourhood, if it wasn't in the subsample  
            F_emp = np.zeros(neighbourhood_size)
            F_emp[indices_mu_sorted] = np.arange(1,neighbourhood_size+1)/neighbourhood_size # Empirical cumulative distribution (of NN_ratio_mu values for the subset of observations within the neighbourhood)
            indices_mu_filtered_out = indices_mu_sorted[-no_deleted:] # We need to discard at least the largest mu value, which corresponds to F_emp = 1, causing log(1-F_emp) = -Inf
            NN_ratio_mu_filtered = np.delete(NN_ratio_mu[KNN_data_indices_obs], indices_mu_filtered_out)
            F_emp_filtered = np.delete(F_emp, indices_mu_filtered_out)
            x = np.log(NN_ratio_mu_filtered)[np.newaxis].T
            y = -np.log(1-F_emp_filtered)[np.newaxis].T
            reg = LinearRegression(fit_intercept = False).fit(x, y) # Regression with no intercept
            Two_NN_LID_estimates[obs] = reg.coef_ # Store the regression coefficient (slope) as a coarse LID estimate
        if return_smoothed == True:
            Two_NN_LID_smoothed_estimates = np.array([np.mean(Two_NN_LID_estimates[subsample_indexes[KNN_Sindices[obs]]]) for obs in range(data_size)]) # LID of query as the mean of the LIDs in the query's subsample
                                                                                                                                                        # neighbourhood. Notice that the neighbours of each query
                                                                                                                                                        # observation here (whose mean LID is the smoothed LID estimate
                                                                                                                                                        # of the query) are the within-subsample neighbours (rather than
                                                                                                                                                        # the global dataset neighbours) 
    if return_smoothed == True:
        return np.stack((Two_NN_LID_estimates, Two_NN_LID_smoothed_estimates))
    else:
        return Two_NN_LID_estimates


# *** MLE LID ESTIMATOR ***:

def MLE_LID_Estimator(data_array, subsample_indexes = None, *, neighbourhood_size = 100, return_smoothed=False, **kwargs):
    '''
    Perform MLE LID Estimation (optionally, w.r.t. a reference subsample)
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . subsample_indexes:  Row indexes of a subset of the dataset observations to be used as a reference subsample for computation of LIDs.
                              If subsample_indexes is None (default), then LIDs are computed in the conventional way, using the whole dataset.
                              Otherwise, the LID of each observation is computed w.r.t. the subsample only. Out of subsample observations also
                              have their LIDs computed, but they do NOT affect LIDs of any other observation (within or outside the subsample);
        . neighbourhood_size: Number of neighbours used to determine the MLE estimate;
        . return_smoothed:    Whether or not a smoothed version of the estimates (averaging across the neighbours) should be computed/returned.
      OUTPUT:
        . If return_smoothed == True: A single 2D numpy array concatenating: (i) the raw (non-smoothed) LID estimates [index 0 along axis 0 of
          the returned array]; and (ii) the smoothed LID estimates, where the LID of a query is the mean LID within the query's neighbourhood
          [index 1 along axis 0 of the returned array]. The returned array has shape (2, data_size);
        . If return_smoothed == False: A 1D numpy array of shape (data_size, ) with item (i) above only.
      NOTE:
        . This estimator does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function.
    '''
  # Preliminaries:
    data_size = data_array.shape[0] # Number of observations in the dataset
    if ( (subsample_indexes is None) and not (3 <= neighbourhood_size < data_size) ):
        print("-------------------------------------------------------------------------------------------------------")
        print("Abort - MLE Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < data size !!!")
        print("-------------------------------------------------------------------------------------------------------")
        return None
    elif ( (subsample_indexes is not None) and not (3 <= neighbourhood_size < len(subsample_indexes)) ):
        print("------------------------------------------------------------------------------------------------------------")
        print("Abort - MLE Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < subsample size !!!")
        print("------------------------------------------------------------------------------------------------------------")
        return None
    else: # Neighbourhood size properly provided
        neighbourhood_size = np.ceil(neighbourhood_size).astype(int)
        
    # Compute LIDs:
    if subsample_indexes is None: # Compute LIDs w.r.t. the whole dataset
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size, algorithm='auto').fit(data_array) # Nearest Neighbours Oracle
        nn_dist, KNN_indices = nbrs.kneighbors()  # Since the input argument of nbrs.kneighbors() has been ommitted, then the distances and indices of the neighbours of the reference data themselves
                                                  # (used to fit the NN Oracle) are returned. In this particular setting, each observation does NOT appear as its own neighbour in the result
        if return_smoothed == True:
            KNN_indices = np.concatenate((np.arange(0,data_size)[np.newaxis].T, KNN_indices), axis=1) # Include each observation as its own neighbour for the sake of smoothing (if any)
    else: # subsample_indexes is not None, so, compute LIDs w.r.t. the subsample provided as input
        data_array_subs = data_array[subsample_indexes] # Data subsample
        mask_subsample = False*np.ones(data_size, dtype = bool)
        mask_subsample[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the subsample, False otherwise (out-of-subsample)
        out_of_subsample_indexes = np.array([x for x in range(data_size)])[mask_subsample == False] # Indexes of observations not contained in the subsample
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size+1, algorithm='auto').fit(data_array_subs) # Reference database for the NN Oracle is the subsample
        distances_insub, KNN_Sindices_insub = nbrs.kneighbors(data_array_subs)  # Since these query observations are in the reference database, then the
                                                                                # returned 1st NN of each observation is the observation itself (with dist = 0)
        distances_outsub, KNN_Sindices_outsub = nbrs.kneighbors(data_array[out_of_subsample_indexes]) # Since these query observations are NOT in the reference database, then the
                                                                                                      # returned 1st NN of each observation is NOT the observation itself
        nn_dist = np.zeros((data_size, neighbourhood_size))
        nn_dist[subsample_indexes] = distances_insub[:, 1:] # Discard own observation as its 1st NN to keep the real 1st NN as such (keeps k NNs indexed from 1 to k)
        nn_dist[out_of_subsample_indexes] = distances_outsub[:, :-1] # Out of subsample observations are not their own 1st neighbour (keeps k NNs indexed from 0 to k-1)
        if return_smoothed == True:
            KNN_Sindices = np.zeros([data_size,neighbourhood_size]).astype(int) # Indexes for each row of the dataset, POINTING TO ROWS OF THE SUBSAMPLE (corresponding to the observation's NNs in the subsample, indexed as such)
            KNN_Sindices[subsample_indexes] = KNN_Sindices_insub[:, :-1]  # Keep each observation in the subsample as its own neighbour FOR THE SAKE OF SMOOTHING (if any) - Discard last (extra) neighbour instead 
            KNN_Sindices[out_of_subsample_indexes] = KNN_Sindices_outsub[:, :-1] # This subset corresponds to the observations outside the subsample
    MLE_LID_estimates = -neighbourhood_size / np.sum(np.log(nn_dist[:, 0:] / nn_dist[:, -1].reshape([-1, 1])), axis=1)
    if return_smoothed == True:
        # Compute Smoothed LIDs:
        if subsample_indexes is None:
            MLE_LID_smoothed_estimates = np.array([np.mean(MLE_LID_estimates[KNN_indices[obs]]) for obs in range(data_size)]) # LID smoothing: LID of query as the mean of the LIDs in the query's neighbourhood (including itself)
        else:
            MLE_LID_smoothed_estimates = np.array([np.mean(MLE_LID_estimates[subsample_indexes[KNN_Sindices[obs]]]) for obs in range(data_size)]) # LID of query as the mean of the LIDs in the query's subsample neighbourhood (which may or not include itself)
        return np.stack((MLE_LID_estimates, MLE_LID_smoothed_estimates))
    else:
        return MLE_LID_estimates


# *** MLE LID ESTIMATOR - WEIBULL HYPOTHESIS ***:

def Weibull_MLE_LID_Estimator(data_array, subsample_indexes = None, *, neighbourhood_size = 100, return_smoothed = True, **kwargs):
    '''
    Perform MLE LID Estimation (optionally, w.r.t. a reference subsample) under the hypothesis of a WEIBULL (rather than tail) distance distribution
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . subsample_indexes:  Row indexes of a subset of the dataset observations to be used as a reference subsample for computation of LIDs.
                              If subsample_indexes is None (default), then LIDs are computed in the conventional way, using the whole dataset.
                              Otherwise, the LID of each observation is computed w.r.t. the subsample only. Out of subsample observations also
                              have their LIDs computed, but they do NOT affect LIDs of any other observation (within or outside the subsample);
        . neighbourhood_size: Number of neighbours used to determine the MLE estimate;
        . return_smoothed:    Whether or not a smoothed version of the estimates (averaging across the neighbours) should be computed/returned.
      OUTPUT:
        . If return_smoothed == True: A single 2D numpy array concatenating: (i) the raw (non-smoothed) LID estimates [index 0 along axis 0 of
          the returned array]; and (ii) the smoothed LID estimates, where the LID of a query is the mean LID within the query's neighbourhood
          [index 1 along axis 0 of the returned array]. The returned array has shape (2, data_size);
        . If return_smoothed == False: A 1D numpy array of shape (data_size, ) with item (i) above only.
      NOTE:
        . This estimator does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function.
    '''
  # Preliminaries:
    data_size = data_array.shape[0] # Number of observations in the dataset
    if ( (subsample_indexes is None) and not (3 <= neighbourhood_size < data_size) ):
        print("---------------------------------------------------------------------------------------------------------------")
        print("Abort - Weibull MLE Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < data size !!!")
        print("---------------------------------------------------------------------------------------------------------------")
        return None
    elif ( (subsample_indexes is not None) and not (3 <= neighbourhood_size < len(subsample_indexes)) ):
        print("--------------------------------------------------------------------------------------------------------------------")
        print("Abort - Weibull MLE Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < subsample size !!!")
        print("--------------------------------------------------------------------------------------------------------------------")
        return None
    else: # Neighbourhood size properly provided
        neighbourhood_size = np.ceil(neighbourhood_size).astype(int)
        
    # Compute LIDs:
    if subsample_indexes is None: # Compute LIDs w.r.t. the whole dataset
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size, algorithm='auto').fit(data_array) # Nearest Neighbours Oracle
        nn_dist, KNN_indices = nbrs.kneighbors()  # Since the input argument of nbrs.kneighbors() has been ommitted, then the distances and indices of the neighbours of the reference data themselves
                                                  # (used to fit the NN Oracle) are returned. In this particular setting, each observation does NOT appear as its own neighbour in the result
        if return_smoothed == True:
            KNN_indices = np.concatenate((np.arange(0,data_size)[np.newaxis].T, KNN_indices), axis=1) # Include each observation as its own neighbour for the sake of smoothing (if any)
    else: # subsample_indexes is not None, so, compute LIDs w.r.t. the subsample provided as input
        data_array_subs = data_array[subsample_indexes] # Data subsample
        mask_subsample = False*np.ones(data_size, dtype = bool)
        mask_subsample[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the subsample, False otherwise (out-of-subsample)
        out_of_subsample_indexes = np.array([x for x in range(data_size)])[mask_subsample == False] # Indexes of observations not contained in the subsample
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size+1, algorithm='auto').fit(data_array_subs) # Reference database for the NN Oracle is the subsample
        distances_insub, KNN_Sindices_insub = nbrs.kneighbors(data_array_subs)  # Since these query observations are in the reference database, then the
                                                                                # returned 1st NN of each observation is the observation itself (with dist = 0)
        distances_outsub, KNN_Sindices_outsub = nbrs.kneighbors(data_array[out_of_subsample_indexes]) # Since these query observations are NOT in the reference database, then the
                                                                                                      # returned 1st NN of each observation is NOT the observation itself
        nn_dist = np.zeros((data_size, neighbourhood_size))
        nn_dist[subsample_indexes] = distances_insub[:, 1:] # Discard own observation as its 1st NN to keep the real 1st NN as such (keeps k NNs indexed from 1 to k)
        nn_dist[out_of_subsample_indexes] = distances_outsub[:, :-1] # Out of subsample observations are not their own 1st neighbour (keeps k NNs indexed from 0 to k-1)
        if return_smoothed == True:
            KNN_Sindices = np.zeros([data_size,neighbourhood_size]).astype(int) # Indexes for each row of the dataset, POINTING TO ROWS OF THE SUBSAMPLE (corresponding to the observation's NNs in the subsample, indexed as such)
            KNN_Sindices[subsample_indexes] = KNN_Sindices_insub[:, :-1]  # Keep each observation in the subsample as its own neighbour FOR THE SAKE OF SMOOTHING (if any) - Discard last (extra) neighbour instead 
            KNN_Sindices[out_of_subsample_indexes] = KNN_Sindices_outsub[:, :-1] # This subset corresponds to the observations outside the subsample
    MLE_LID_estimates = np.zeros(data_size)
    for obs in range(data_size):
        shape_par, location_par, scale_par = weibull_min.fit(nn_dist[obs, 0:], floc = 0, method = "MLE") # Note:  floc = 0 fixes (constrains) the location parameter in such a way that the generalised distribution reduces to the 2-parameter Weibull distribution.
                                                                                                         #        This is different from loc = 0, which just initialises the optimiser with an initial value of the location argument, without constraining it.    
        MLE_LID_estimates[obs] = shape_par # LID can be theoretically shown to correspond to the shape parameter of a 2-parameter Weibull distribution, under the assumption that distances from the query follow such a distribution
    if return_smoothed == True:
        # Compute Smoothed LIDs:
        if subsample_indexes is None:
            MLE_LID_smoothed_estimates = np.array([np.mean(MLE_LID_estimates[KNN_indices[obs]]) for obs in range(data_size)]) # LID smoothing: LID of query as the mean of the LIDs in the query's neighbourhood (including itself)
        else:
            MLE_LID_smoothed_estimates = np.array([np.mean(MLE_LID_estimates[subsample_indexes[KNN_Sindices[obs]]]) for obs in range(data_size)]) # LID of query as the mean of the LIDs in the query's subsample neighbourhood (which may or not include itself)
        return np.stack((MLE_LID_estimates, MLE_LID_smoothed_estimates))
    else:
        return MLE_LID_estimates


# *** MM (METHOD OF MOMENTS) LID ESTIMATOR ***:

def MM_LID_Estimator(data_array, subsample_indexes = None, *, neighbourhood_size = 100, order_m = 2, return_smoothed = True, **kwargs):
    '''
    Perform Method of Moments (MM) LID Estimation (optionally, w.r.t. a reference subsample)
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . subsample_indexes:  Row indexes of a subset of the dataset observations to be used as a reference subsample for computation of LIDs.
                              If subsample_indexes is None (default), then LIDs are computed in the conventional way, using the whole dataset.
                              Otherwise, the LID of each observation is computed w.r.t. the subsample only. Out of subsample observations also
                              have their LIDs computed, but they do NOT affect LIDs of any other observation (within or outside the subsample);
        . neighbourhood_size: Number of neighbours used to determine the MM estimate;
        . order_m:            Order of the moment (arbitrary small positive integer);
        . return_smoothed:    Whether or not a smoothed version of the estimates (averaging across the neighbours) should be computed/returned.
      OUTPUT:
        . If return_smoothed == True: A single 2D numpy array concatenating: (i) the raw (non-smoothed) LID estimates [index 0 along axis 0 of
          the returned array]; and (ii) the smoothed LID estimates, where the LID of a query is the mean LID within the query's neighbourhood
          [index 1 along axis 0 of the returned array]. The returned array has shape (2, data_size);
        . If return_smoothed == False: A 1D numpy array of shape (data_size, ) with item (i) above only.
      NOTE:
        . This estimator does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function.
    '''
  # Preliminaries:
    data_size = data_array.shape[0] # Number of observations in the dataset
    if ( (subsample_indexes is None) and not (3 <= neighbourhood_size < data_size) ):
        print("-------------------------------------------------------------------------------------------------------")
        print("Abort - MM Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < data size !!!")
        print("-------------------------------------------------------------------------------------------------------")
        return None
    elif ( (subsample_indexes is not None) and not (3 <= neighbourhood_size < len(subsample_indexes)) ):
        print("------------------------------------------------------------------------------------------------------------")
        print("Abort - MM Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < subsample size !!!")
        print("------------------------------------------------------------------------------------------------------------")
        return None
    else: # Neighbourhood size properly provided
        neighbourhood_size = np.ceil(neighbourhood_size).astype(int)
        
    # Compute LIDs:
    if subsample_indexes is None: # Compute LIDs w.r.t. the whole dataset
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size, algorithm='auto').fit(data_array) # Nearest Neighbours Oracle
        nn_dist, KNN_indices = nbrs.kneighbors()  # Since the input argument of nbrs.kneighbors() has been ommitted, then the distances and indices of the neighbours of the reference data themselves
                                                  # (used to fit the NN Oracle) are returned. In this particular setting, each observation does NOT appear as its own neighbour in the result
        if return_smoothed == True:
            KNN_indices = np.concatenate((np.arange(0,data_size)[np.newaxis].T, KNN_indices), axis=1) # Include each observation as its own neighbour for the sake of smoothing (if any)
    else: # subsample_indexes is not None, so, compute LIDs w.r.t. the subsample provided as input
        data_array_subs = data_array[subsample_indexes] # Data subsample
        mask_subsample = False*np.ones(data_size, dtype = bool)
        mask_subsample[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the subsample, False otherwise (out-of-subsample)
        out_of_subsample_indexes = np.array([x for x in range(data_size)])[mask_subsample == False] # Indexes of observations not contained in the subsample
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size+1, algorithm='auto').fit(data_array_subs) # Reference database for the NN Oracle is the subsample
        distances_insub, KNN_Sindices_insub = nbrs.kneighbors(data_array_subs)  # Since these query observations are in the reference database, then the
                                                                                # returned 1st NN of each observation is the observation itself (with dist = 0)
        distances_outsub, KNN_Sindices_outsub = nbrs.kneighbors(data_array[out_of_subsample_indexes]) # Since these query observations are NOT in the reference database, then the
                                                                                                      # returned 1st NN of each observation is NOT the observation itself
        nn_dist = np.zeros((data_size, neighbourhood_size))
        nn_dist[subsample_indexes] = distances_insub[:, 1:] # Discard own observation as its 1st NN to keep the real 1st NN as such (keeps k NNs indexed from 1 to k)
        nn_dist[out_of_subsample_indexes] = distances_outsub[:, :-1] # Out of subsample observations are not their own 1st neighbour (keeps k NNs indexed from 0 to k-1)
        if return_smoothed == True:
            KNN_Sindices = np.zeros([data_size,neighbourhood_size]).astype(int) # Indexes for each row of the dataset, POINTING TO ROWS OF THE SUBSAMPLE (corresponding to the observation's NNs in the subsample, indexed as such)
            KNN_Sindices[subsample_indexes] = KNN_Sindices_insub[:, :-1]  # Keep each observation in the subsample as its own neighbour FOR THE SAKE OF SMOOTHING (if any) - Discard last (extra) neighbour instead 
            KNN_Sindices[out_of_subsample_indexes] = KNN_Sindices_outsub[:, :-1] # This subset corresponds to the observations outside the subsample
    mu_m = np.mean(np.power(nn_dist,int(order_m)), axis=1)
    w_power_m = np.power(nn_dist[:,-1],int(order_m)) 
    MM_LID_estimates = -order_m * mu_m / (mu_m - w_power_m)
    if return_smoothed == True:
        # Compute Smoothed LIDs:
        if subsample_indexes is None:
            MM_LID_smoothed_estimates = np.array([np.mean(MM_LID_estimates[KNN_indices[obs]]) for obs in range(data_size)]) # LID smoothing: LID of query as the mean of the LIDs in the query's neighbourhood (including itself)
        else:
            MM_LID_smoothed_estimates = np.array([np.mean(MM_LID_estimates[subsample_indexes[KNN_Sindices[obs]]]) for obs in range(data_size)]) # LID of query as the mean of the LIDs in the query's subsample neighbourhood (which may or not include itself)
        return np.stack((MM_LID_estimates, MM_LID_smoothed_estimates))
    else:
        return MM_LID_estimates


def LEO_LID_Estimator(data_array, subsample_indexes = None, *, neighbourhood_size = 100, return_smoothed = True, **kwargs):
    '''
    Perform Local Estimated-to-Observed Nearest Neighbour Relative Differences [LEONNARD] LID Estimation (optionally, w.r.t. a ref. subsample)
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . subsample_indexes:  Row indexes of a subset of the dataset observations to be used as a reference subsample for computation of LIDs.
                              If subsample_indexes is None (default), then LIDs are computed in the conventional way, using the whole dataset.
                              Otherwise, the LID of each observation is computed w.r.t. the subsample only. Out of subsample observations also
                              have their LIDs computed, but they do NOT affect LIDs of any other observation (within or outside the subsample);
        . neighbourhood_size: Number of neighbours used to determine the LEONNARD (LEO for short) estimate;
        . return_smoothed:    Whether or not a smoothed version of the estimates (averaging across the neighbours) should be computed/returned.
      OUTPUT:
        . If return_smoothed == True: A single 2D numpy array concatenating: (i) the raw (non-smoothed) LID estimates [index 0 along axis 0 of
          the returned array]; and (ii) the smoothed LID estimates, where the LID of a query is the mean LID within the query's neighbourhood
          [index 1 along axis 0 of the returned array]. The returned array has shape (2, data_size);
        . If return_smoothed == False: A 1D numpy array of shape (data_size, ) with item (i) above only.
      NOTE:
        . This estimator does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function.
    '''
  # Preliminaries:
    data_size = data_array.shape[0] # Number of observations in the dataset
    if ( (subsample_indexes is None) and not (3 <= neighbourhood_size < data_size) ):
        print("-------------------------------------------------------------------------------------------------------")
        print("Abort - LEO Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < data size !!!")
        print("-------------------------------------------------------------------------------------------------------")
        return None
    elif ( (subsample_indexes is not None) and not (3 <= neighbourhood_size < len(subsample_indexes)) ):
        print("------------------------------------------------------------------------------------------------------------")
        print("Abort - LEO Estimator: Neighbourhood size needs to be such that 3 <= neighbourhood_size < subsample size !!!")
        print("------------------------------------------------------------------------------------------------------------")
        return None
    else: # Neighbourhood size properly provided
        neighbourhood_size = np.ceil(neighbourhood_size).astype(int)
        
    # Compute LIDs:
    if subsample_indexes is None: # Compute LIDs w.r.t. the whole dataset
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size, algorithm='auto').fit(data_array) # Nearest Neighbours Oracle
        nn_dist, KNN_indices = nbrs.kneighbors()  # Since the input argument of nbrs.kneighbors() has been ommitted, then the distances and indices of the neighbours of the reference data themselves
                                                  # (used to fit the NN Oracle) are returned. In this particular setting, each observation does NOT appear as its own neighbour in the result
        if return_smoothed == True:
            KNN_indices = np.concatenate((np.arange(0,data_size)[np.newaxis].T, KNN_indices), axis=1) # Include each observation as its own neighbour for the sake of smoothing (if any)
    else: # subsample_indexes is not None, so, compute LIDs w.r.t. the subsample provided as input
        data_array_subs = data_array[subsample_indexes] # Data subsample
        mask_subsample = False*np.ones(data_size, dtype = bool)
        mask_subsample[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the subsample, False otherwise (out-of-subsample)
        out_of_subsample_indexes = np.array([x for x in range(data_size)])[mask_subsample == False] # Indexes of observations not contained in the subsample
        nbrs = NearestNeighbors(n_neighbors=neighbourhood_size+1, algorithm='auto').fit(data_array_subs) # Reference database for the NN Oracle is the subsample
        distances_insub, KNN_Sindices_insub = nbrs.kneighbors(data_array_subs)  # Since these query observations are in the reference database, then the
                                                                                # returned 1st NN of each observation is the observation itself (with dist = 0)
        distances_outsub, KNN_Sindices_outsub = nbrs.kneighbors(data_array[out_of_subsample_indexes]) # Since these query observations are NOT in the reference database, then the
                                                                                                      # returned 1st NN of each observation is NOT the observation itself
        nn_dist = np.zeros((data_size, neighbourhood_size))
        nn_dist[subsample_indexes] = distances_insub[:, 1:] # Discard own observation as its 1st NN to keep the real 1st NN as such (keeps k NNs indexed from 1 to k)
        nn_dist[out_of_subsample_indexes] = distances_outsub[:, :-1] # Out of subsample observations are not their own 1st neighbour (keeps k NNs indexed from 0 to k-1)
        if return_smoothed == True:
            KNN_Sindices = np.zeros([data_size,neighbourhood_size]).astype(int) # Indexes for each row of the dataset, POINTING TO ROWS OF THE SUBSAMPLE (corresponding to the observation's NNs in the subsample, indexed as such)
            KNN_Sindices[subsample_indexes] = KNN_Sindices_insub[:, :-1]  # Keep each observation in the subsample as its own neighbour FOR THE SAKE OF SMOOTHING (if any) - Discard last (extra) neighbour instead 
            KNN_Sindices[out_of_subsample_indexes] = KNN_Sindices_outsub[:, :-1] # This subset corresponds to the observations outside the subsample
    array_of_ks = np.tile(np.arange(1, neighbourhood_size+1), (data_size, 1))
    num_LID_est = np.sum(np.power(np.log(array_of_ks[:, 0:-1] / array_of_ks[:, -1].reshape([-1, 1])),2), axis=1)
    den_LID_est = np.sum(np.log(nn_dist[:, 0:-1] / nn_dist[:, -1].reshape([-1, 1]))*np.log(array_of_ks[:, 0:-1] / array_of_ks[:, -1].reshape([-1, 1])), axis=1)
    LEO_LID_estimates = num_LID_est / den_LID_est
    if return_smoothed == True:
        # Compute Smoothed LIDs:
        if subsample_indexes is None:
            LEO_LID_smoothed_estimates = np.array([np.mean(LEO_LID_estimates[KNN_indices[obs]]) for obs in range(data_size)]) # LID smoothing: LID of query as the mean of the LIDs in the query's neighbourhood (including itself)
        else:
            LEO_LID_smoothed_estimates = np.array([np.mean(LEO_LID_estimates[subsample_indexes[KNN_Sindices[obs]]]) for obs in range(data_size)]) # LID of query as the mean of the LIDs in the query's subsample neighbourhood (which may or not include itself)
        return np.stack((LEO_LID_estimates, LEO_LID_smoothed_estimates))
    else:
        return LEO_LID_estimates


# *** OUT-OF-BAG LIKELIHOOD OR GOODNESS OF FIT MEASURES AS INTERNAL EVALUATION CRITERIA FOR BAGGED CANDIDATE LIDs ***:

def Weibull_Scale_Estimator(data_array, lid_estimates, subsample_indexes = None, *, NN_size_for_w = 7, **kwargs):
    '''
    Perform MLE estimation of the scale parameter of an assumed WEIBULL (rather than tail) distance distribution w.r.t. a reference data subsample
    (BAG) that is supposed to have been used to pre-estimate the fixed shape parameter, which is provided as input as the LID of each observation 
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . lid_estimates:      1D numpy array of float type containing the candidate LID estimates for every observation in the dataset computed 
                              w.r.t. a subsample of the data (IN-BAG) whose observations are indexed by <subsample_indexes>;
        . subsample_indexes:  Row indexes of a subset of the dataset observations to be used as a reference subsample for scale parameter estimation.
                              This is supposed to contain the indexes of the subset of observations that have been used to produce <lid_estimates>.
                              If subsample_indexes is None (default), then estimations will be performed using the whole dataset as a reference;
        . NN_size_for_w:      Number of neighbours used to determine the MLE estimate.
      OUTPUT:
        . A 1D numpy array of shape (data_size, ) with the MLE of the scale parameter of each query given its fixed candidate LID.
      NOTE:
        . This estimator does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function.
    '''
    # Preliminaries:
    data_size = data_array.shape[0] # Number of observations in the dataset
    if ( (subsample_indexes is None) and not (3 <= NN_size_for_w < data_size) ):
        print("----------------------------------------------------------------------------------------------------------------")
        print("Abort - Weibull MLE Scale Estimator: Neighbourhood size needs to be such that 3 <= NN_size_for_w < data size !!!")
        print("----------------------------------------------------------------------------------------------------------------")
        return None
    elif ( (subsample_indexes is not None) and not (3 <= NN_size_for_w < len(subsample_indexes)) ):
        print("---------------------------------------------------------------------------------------------------------------------")
        print("Abort - Weibull MLE Scale Estimator: Neighbourhood size needs to be such that 3 <= NN_size_for_w < subsample size !!!")
        print("---------------------------------------------------------------------------------------------------------------------")
        return None
    else: # Neighbourhood size properly provided
        NN_size_for_w = np.ceil(NN_size_for_w).astype(int)
    lid_estimates = lid_estimates.flatten() # Make sure lid_est is a 1D numpy array (rather than a row or column array)
    # Compute MLE (scale only):
    if subsample_indexes is None: # Compute MLE w.r.t. the whole dataset
        nbrs = NearestNeighbors(n_neighbors=NN_size_for_w, algorithm='auto').fit(data_array) # Nearest Neighbours Oracle
        nn_dist, KNN_indices = nbrs.kneighbors()  # Since the input argument of nbrs.kneighbors() has been ommitted, then the distances and indices of the neighbours of the reference data themselves
                                                  # (used to fit the NN Oracle) are returned. In this particular setting, each observation does NOT appear as its own neighbour in the result
    else: # <subsample_indexes> is not None, so, compute MLE w.r.t. the subsample provided as input
        data_array_subs = data_array[subsample_indexes] # Data subsample
        mask_subsample = False*np.ones(data_size, dtype = bool)
        mask_subsample[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the subsample, False otherwise (out-of-subsample)
        out_of_subsample_indexes = np.array([x for x in range(data_size)])[mask_subsample == False] # Indexes of observations not contained in the subsample
        nbrs = NearestNeighbors(n_neighbors=NN_size_for_w+1, algorithm='auto').fit(data_array_subs) # Reference database for the NN Oracle is the subsample
        distances_insub, KNN_Sindices_insub = nbrs.kneighbors(data_array_subs)  # Since these query observations are in the reference database, then the
                                                                                # returned 1st NN of each observation is the observation itself (with dist = 0)
        distances_outsub, KNN_Sindices_outsub = nbrs.kneighbors(data_array[out_of_subsample_indexes]) # Since these query observations are NOT in the reference database, then the
                                                                                                      # returned 1st NN of each observation is NOT the observation itself
        nn_dist = np.zeros((data_size, NN_size_for_w))
        nn_dist[subsample_indexes] = distances_insub[:, 1:] # Discard own observation as its 1st NN to keep the real 1st NN as such (keeps k NNs indexed from 1 to k)
        nn_dist[out_of_subsample_indexes] = distances_outsub[:, :-1] # Out of subsample observations are not their own 1st neighbour (keeps k NNs indexed from 0 to k-1)
    MLE_Weibull_scale_param_estimates = np.mean(nn_dist**lid_estimates.reshape([-1, 1]), axis=1)**(1/lid_estimates) # Optimal MLE solution for the scale parameter given LID as the shape parameter
    return MLE_Weibull_scale_param_estimates

def OOB_Likelihood(data_array, *, lid_estimates, subsample_indexes, NN_size_for_w = 5, LE_type = "SNL", dist_assumption = "Tail", rand_gen_seed = None, min_LID_threshold = 0.1, **kwargs):
    '''
    OUT-OF-BAG LIKELIHOOD ESTIMATION (INTERNAL EVALUATION MEASURE) OF CANDIDATE LIDs ESTIMATED FOR AN ENTIRE DATASET BASED ON A SUBSAMPLE (BAG)
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . lid_estimates:      Numpy array of float type containing the candidate LID estimates for every observation in the dataset computed 
                              w.r.t. a subsample of the data (bag) whose observations are indexed by <subsample_indexes>. If there is only 1
                              single candidate estimator to be assessed, then <lid_estimates> can be a 1D numpy array. If there are multiple 
                              such candidate estimators, then <lid_estimates> will be a 2D numpy array of shape (no of estimators, data size).
                              In this case, each row of <lid_estimates> will contain the candidate LID estimates of the corresponding candidate
                              estimator for every observation in the dataset;
        . subsample_indexes:  Row indexes of the subset of data observations that WAS USED as a reference subsample for computation of LIDs;
        . NN_size_for_w:      Number of neighbours used to determine the OOB likelihood of <lid_estimates> based on the theoretical distribution
                              of distances within a small radius w around a query. <NN_size_for_w> needs to be a small integer value because the
                              parametric distribution adopted for likelihood estimation may hold true only within a small region around a query.
                              NB. If a list or numpy array is inputted instead, then the smallest and largest values will be interpreted as a range
                              [min(NN_size_for_w), max(NN_size_for_w)], OOB_LE will be computed independently for every neighbourhood size within
                              this range, and an average of the results will be returned;
        . LE_type:            Type of likelihood function to be used: "LL" stands for the regular Log-Likelihood function (MAXIMISATION function),
                              whereas "SNL" stands for Standardized Negative Log-Likelihood function (MINIMISATION OF ABSOLUTE VALUE function).
                              The former uses the density f(d) directly as log[f(d)], where f(d) is the theoretical density function that has been
                              hypothetised (see <dist_assumption> argument below). The latter (default) standardises the negative of the former
                              (-log[f(d)]) as a z-score, by subtracting its expected value (E{-log[f(d)]}, i.e., Entropy of the assumed density)
                              and dividing the result by its standard deviation (std{-log[f(d)]}, i.e., the square root of the Varentropy of the
                              assumed density). In other words, SNL = (-log[f(d)] - E{-log[f(d)]}) / sqrt[Var{-log[f(d)]}]. For the default
                              "Tail" (asymptotic lower tail) distribution, SNL simplifies to a function of its cdf F(d), as SNL = -log(F(d)) - 1;
        . dist_assumption:    The particular theoretical density function that has been hypothetised: "Tail" stands for the asymptotic lower
                              tail distribution, i.e., f(d) = (LID/w)*(d/w)^{LID-1} (conditional to d<=w), whose CDF is given by F(d)=(d/w)^LID,
                              whereas "Weibull" stands for the Weibull distribution, i.e., f(d) = (LID/scale)*(d/scale)^{LID-1}*exp(-(d/scale)^LID),
                              with cdf given by F(d) = 1 - exp(-(d/scale)^LID). The latter requires that the IN-BAG subsample encoded by argument
                              <subsample_indexes> be no larger than the OOB subsample (which under this distribution needs to be downsized to match
                              the size of the IN-BAG), or in other words, IN-BAG cannot be larger than half of the dataset (i.e., the underlying
                              subsampling rate needs to have been <= 0.5); 
        . rand_gen_seed:      Seed for random generator (If None, clock is used) [IT ONLY MAKES ANY EFFECT WHEN <dist_assumption> = "Weibull"]; 
        . min_LID_threshold:  Any individual LID estimate smaller than <min_LID_threshold> will be replaced with <min_LID_threshold> for the
                              sake of OOB_LE computation. This prevents non-positive LID estimates from producing undefined OOB_LE evaluations
                              due to non-positive log computation attempts. 
      OUTPUT:
        . A float type array containing, for each candidate estimator, the OOB Likelihood of the corresponding LID estimates as an internal
          evaluation measure for the individual ensemble member computed from the given bag. The shape is (no of estimators, 1).
      NOTE:
        . It is important to note that the LID estimate for each data observation (query) is assessed independently using only the distances
          from that observation to its neighbours WITHIN THE OOB subsample (i.e., the OOB is the reference NN oracle during evaluation, as
          opposed to the IN-BAG NN oracle that was supposedly used in the estimation phase). Such OOB NN distances of each query, used here
          for evaluation, HAVE NOT BEEN USED/SEEN during the LID estimation of that query. This means that, at the individual query level,
          this evaluation is independent from the training (apart from the fact that the training is unsupervised), and its averaging across
          the entire dataset (which includes the IN-BAG observations) is a valid, independent evaluation procedure;  
        . This measure is only comparable across different LID estimates computed from the same reference subsample of the same data and
          assessed on the same OOB sample associated with the reference subsample in question, and using the same neighbourhood around
          each query. In other words, fix the observations and compare likelihoods for different parameters (LIDs) of the distribution.
          This means that this measure cannot be used to compare ensembles produced with different subsampling rates or no. of members.
          It also means that the neighbourhood around each query used here cannot be different when comparing different LID estimates,
          even if these different estimates were computed using different neighbourhood sizes for estimators with such a parameter
          (e.g. 2NN and MLE); 
        . This measure does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function;
        . If a list or numpy array is inputted as <NN_size_for_w> instead of an integer, then the smallest and largest values in the list/array
          will be interpreted as a range within which OOB_LE will be computed independently and an average of the results will be returned;
        . Choosing dist_assumption = "Weibull" requires len(subsample_indexes) <= 0.5*data_size, because the IN-BAG subsample encoded
          by <subsample_indexes> cannot be larger than the OOB subsample (which needs to be downsized to match the size of the IN-BAG).
    '''
    no_estimation_types = 1 if lid_estimates.ndim == 1 else lid_estimates.shape[0]
    if not (LE_type == "LL" or LE_type == "SNL"): # LE_type is not properly defined, so OOB is None and nothing else needs to be done
        OOB_LE = np.array([None for x in range(no_estimation_types)]).reshape([-1,1]) # Array of None value(s)
        return OOB_LE # Return array of None and finish
    if type(NN_size_for_w) is not np.ndarray:
        if type(NN_size_for_w) is list: # Cast list into numpy array
            NN_size_for_w = np.array(NN_size_for_w)
        else: # Cast atomic numeric value into numpy array
            NN_size_for_w = np.array([int(NN_size_for_w)])
    max_w = np.max(NN_size_for_w) # Largest neighbourhood size corresponding to the largest value of the radius w
    min_w = np.min(NN_size_for_w) # Smallest neighbourhood size corresponding to the smallest value of the radius w
    data_size = data_array.shape[0] # Number of observations in the dataset
    mask = False*np.ones(data_size, dtype = bool)
    mask[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the IN-BAG, False otherwise (OOB)
    OOB_subsample_indexes = np.array([x for x in range(data_size)])[mask == False] # Indexes of observations NOT contained in the IN-BAG (i.e., OOB)
    if (dist_assumption == "Weibull"):
        if (max_w < len(subsample_indexes) <= len(OOB_subsample_indexes)):
            rand_gen = np.random.RandomState(rand_gen_seed) # Random generator
            OOB_downsampling_indexes = rand_gen.choice(len(OOB_subsample_indexes), size=len(subsample_indexes), replace=False)
            OOB_subsample_indexes = OOB_subsample_indexes[OOB_downsampling_indexes] # Downsampled OOB subsample matching the size of the IN-BAG subsample
            mask = False*np.ones(data_size, dtype = bool)
            mask[OOB_subsample_indexes] = True # Boolean mask where True indicates that the corresponding observation is in the downsampled OOB subsample, False otherwise
            not_OOB_subsample_indexes = np.array([x for x in range(data_size)])[mask == False] # Indexes of observations NOT contained in the downsampled OOB subsample
        else: # NOT possible to downsample the OOB to match the IN-BAG, or to obtain the required number of neighbours from the downsampled OOB
            print("---------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Abort - Weibull OOB LE: IN-BAG Cannot Be Larger than the OOB (Sampling Rate Must Be <= 0.5) and Must Be Larger than the Required Neighbourhood Size")
            print("---------------------------------------------------------------------------------------------------------------------------------------------------")
            # return None, None
            return None
    else: # Tail distribution assumed
        not_OOB_subsample_indexes = subsample_indexes # In this case the OOB is NOT downsampled, so what is not in OOB must necessarily be the IN-BAG (encoded by <subsample_indexes>)  
    data_array_OOB = data_array[OOB_subsample_indexes] # OOB data subsample (possibly downsampled to match the IN-BAG size in case of Weibull dist. assumption)
    nbrs = NearestNeighbors(n_neighbors=max_w+1, algorithm='auto').fit(data_array_OOB) # Reference database for the NN Oracle is the OOB subsample (possibly downsampled)
    distances_OOB, *_ = nbrs.kneighbors(data_array_OOB)  # Since these query observations are in the reference database (OOB, possibly downsampled),
                                                         # then the returned 1st NN of each observation is the observation itself (with dist = 0)
    distances_not_OOB, *_ = nbrs.kneighbors(data_array[not_OOB_subsample_indexes])  # Since these query observations are NOT in the reference database (OOB), then
                                                                                    # the returned 1st NN of each observation is NOT the observation itself.
                                                                                    # NOTE: If/when the OOB is downsampled (under the Weibull dist. assumption), then these queries 
                                                                                    # will no longer be strictly limited to the IN-BAG subsample indexed by <subsample_indexes> 
    nn_dist = np.zeros([data_size, max_w])
    nn_dist[not_OOB_subsample_indexes] = distances_not_OOB[:, :-1] # Subsample observations are not their own 1st neighbour (keeps k NNs indexed from 0 to k-1)
    nn_dist[OOB_subsample_indexes] = distances_OOB[:, 1:] # Discard own observation as its 1st NN to keep the real 1st NN as such (keeps k NNs indexed from 1 to k)
    OOB_LE = np.zeros([no_estimation_types,1])
    OOB_LE_w = np.zeros([no_estimation_types,1])
    for iter_est in range(no_estimation_types):
        lid_est = lid_estimates if lid_estimates.ndim == 1 else lid_estimates[iter_est] # Make sure lid_est is a 1D numpy array
        lid_est = np.maximum(lid_est, min_LID_threshold) # Make sure there aren't non-positive LID estimates
        lid_est = lid_est.reshape([-1, 1]) # Transform into column-vector to make broadcasting and notation simpler
        for kw in range(min_w,max_w+1):
            w = nn_dist[:,kw-1].reshape([-1, 1])
            if (dist_assumption == "Weibull"):
                scale_param = Weibull_Scale_Estimator(data_array, lid_est, subsample_indexes, NN_size_for_w = kw) # Optimal scale par. given the fixed shape par. (candidate LID)
                                                                                                                  # for each observation, both computed w.r.t. the same subsample
                scale_param = scale_param.reshape([-1, 1]) # Transform into column-vector to make broadcasting and notation simpler
            if (LE_type == "LL"): # Regular Log-Likelihood function (MAXIMISATION function)
                if (dist_assumption == "Weibull"):
                    # For each LID estimator: 1/N * 1/k * sum_{i=1}{N} sum_{j=1}{k} log f(d_ij| d_ij <= w_i), where f(d_ij| d_ij <= w_i) = (LID_i/beta_i)*(d_ij/beta_i)^{LID_i - 1}*exp(-(d_ij/beta_i)^{LID_i})
                    OOB_LE_w[iter_est] = (1/data_size)*(1/kw)*np.sum(np.log((lid_est / scale_param)*np.power(nn_dist[:,:kw] / scale_param, lid_est - 1)*np.exp(-np.power(nn_dist[:,:kw] / scale_param, lid_est))))
                else: # Lower Tail Asymptotic Distribution Assumed by Default
                    # For each LID estimator: 1/N * 1/k * sum_{i=1}{N} sum_{j=1}{k} log f(d_ij| d_ij <= w_i), where f(d_ij| d_ij <= w_i) = (LID_i/w_i)*(d_ij/w_i)^{LID_i - 1}
                    OOB_LE_w[iter_est] = (1/data_size)*(1/kw)*np.sum(np.log((lid_est / w)*np.power(nn_dist[:,:kw] / w, lid_est - 1)))
            elif (LE_type == "SNL"): # Standardized Negative Log-Likelihood function (MINIMISATION OF ABSOLUTE VALUE function)
                if (dist_assumption == "Weibull"):
                    Entropy_Weibull = np.euler_gamma*(1 - 1/lid_est) + np.log(scale_param/lid_est) + 1 # E[-log f(d|d<=w)] (colum vector across all data queries)
                    Varentropy_Weibull = (np.pi**2 / 6)*((1 - 1/lid_est)**2) + (2/lid_est) - 1 # Var[-log f(d|d<=w)] (colum vector across all data queries)
                    # Matrix with f(d_ij | d_ij <= w_i) for all queries along rows and their NN distances along columns:
                    Weibull_PDF = (lid_est / scale_param)*np.power(nn_dist[:,:kw] / scale_param, lid_est - 1)*np.exp(-np.power(nn_dist[:,:kw] / scale_param, lid_est))
                    # For each LID estimator: 1/N * 1/k * sum_{i=1}{N} sum_{j=1}{k} (-log f(d_ij| d_ij <= w_i) - E[-log f(d_ij| d_ij <= w_i)]) / sqrt(Var[-log f(d_ij| d_ij <= w_i)])
                    OOB_LE_w[iter_est] = (1/data_size)*(1/kw)*np.sum((-np.log(Weibull_PDF) - Entropy_Weibull) / np.sqrt(Varentropy_Weibull))
                else: # Lower Tail Asymptotic Distribution Assumed by Default
                    # For each LID estimator: 1/N * 1/k * sum_{i=1}{N} sum_{j=1}{k} -log F(d_ij| d_ij <= w_i) - 1, where F(d_ij| d_ij <= w_i) = (d_ij/w_i)^{LID_i}
                    OOB_LE_w[iter_est] = (1/data_size)*(1/kw)*np.sum(-np.log(np.power(nn_dist[:,:kw] / w, lid_est)) - 1)
            OOB_LE[iter_est] += OOB_LE_w[iter_est] / ( max_w - min_w + 1)
    return OOB_LE

def cdf_distance_from_query(d_within_w, w, LID):
    # Required Function Template: cdf(d, *args) -> float
    # Assumption: Theoretical Asymptotic Tail Distribution, i.e., F(d) = (d/w)^LID for small w
    return np.power(d_within_w / w, LID)

def Weibull_cdf_distance_from_query(d_within_w, scale_param, LID):
    # Required Function Template: cdf(d, *args) -> float
    # Assumption: Weibull Distribution, i.e., F(d) = 1 - exp(-(d/scale)^LID)
    return 1 - np.exp(-((d_within_w / scale_param)**LID))

def OOB_Goodness_of_Fit(data_array, *, lid_estimates, subsample_indexes, NN_size_for_w = 7, GoF_type = "CVM", dist_assumption = "Tail", rand_gen_seed = None, min_LID_threshold = 0.1, **kwargs):
    '''
    OUT-OF-BAG GOODNESS OF FIT ESTIMATION (INTERNAL EVALUATION MEASURE) OF CANDIDATE LIDs ESTIMATED FOR AN ENTIRE DATASET BASED ON A SUBSAMPLE (BAG)
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a float type vector;
        . lid_estimates:      Numpy array of float type containing the candidate LID estimates for every observation in the dataset computed 
                              w.r.t. a subsample of the data (bag) whose observations are indexed by <subsample_indexes>. If there is only 1
                              single candidate estimator to be assessed, then <lid_estimates> can be a 1D numpy array. If there are multiple 
                              such candidate estimators, then <lid_estimates> will be a 2D numpy array of shape (no of estimators, data size).
                              In this case, each row of <lid_estimates> will contain the candidate LID estimates of the corresponding candidate
                              estimator for every observation in the dataset;
        . subsample_indexes:  Row indexes of the subset of data observations that WAS USED as a reference subsample for computation of LIDs;
        . NN_size_for_w:      Number of neighbours used to determine the OOB GoF of <lid_estimates> based on the theoretical distribution of
                              distances within a small radius w around a query. <NN_size_for_w> needs to be a small integer value because LID
                              and the parametric distribution used for GoF testing may hold true only within a very small region around a query.
                              On the other hand, it cannot be too small such that there isn't enough power for the test. The default value
                              of 7 is based on the paper: S. Csorgo and J. Faraway "The Exact and Asymptotic Distributions of Cramer-von Mises
                              Statistics", J. of the Royal Stat. Society. Series B (Methodological) , 1996, Vol. 58, No. 1 (1996), pp. 221-234;
        . GoF_type:           Type of GoF test to be used: "CVM" stands for the one-sample Cramér-von Mises test, whereas "KS" stands for the 
                              Kolmogorov-Smirnov test. Both test the sample (evidence) against the CDF of the theoretical density function that
                              has been hypothetised (see <dist_assumption> argument below);
        . dist_assumption:    The particular theoretical density function that has been hypothetised: "Tail" stands for the asymptotic lower
                              tail distribution, i.e., f(d) = (LID/w)*(d/w)^{LID-1} (conditional to d<=w), whose CDF is given by F(d)=(d/w)^LID,
                              whereas "Weibull" stands for the Weibull distribution assumption, with cdf given by F(d) = 1 - exp(-(d/scale)^LID).
                              The latter requires that the IN-BAG subsample encoded by <subsample_indexes> be no larger than the OOB subsample
                              (which under this distribution needs to be downsized to match the size of the IN-BAG), or in other words, IN-BAG
                              cannot be larger than half of the dataset (i.e., the underlying subsampling rate needs to have been <= 0.5); 
        . rand_gen_seed:      Seed for random generator (If None, clock is used) [IT ONLY MAKES ANY EFFECT WHEN <dist_assumption> = "Weibull"]; 
        . min_LID_threshold:  Any individual LID estimate smaller than <min_LID_threshold> will be replaced with <min_LID_threshold> for the
                              sake of OOB_GoF computation. This prevents non-positive LID estimates from producing meaningless test results. 
      OUTPUT:
        . A float type array containing, for each candidate estimator, the OOB GoF of the corresponding LID estimates as an internal
          evaluation measure for the individual ensemble member computed from the given bag. It is an average probability that the
          distance samples around each query come from the theoretical distribution (with the corresponding candidate LIDs as parameter).
          In other words, it is the average p-value (across observations) associated with the null hypothesis that the empirical cdf F(d)
          is the same as the theoretical cdf F0(d), i.e., F = F0. The higher the value the better. The shape is (no of estimators, 1)
      NOTES:
        . It is important to note that the LID estimate for each data observation (query) is assessed independently using only the distances
          from that observation to its neighbours WITHIN THE OOB subsample (i.e., the OOB is the reference NN oracle during evaluation, as
          opposed to the IN-BAG NN oracle that was supposedly used in the estimation phase). Such OOB NN distances of each query, used here
          for evaluation, HAVE NOT BEEN USED/SEEN during the LID estimation of that query. This means that, at the individual query level,
          this evaluation is independent from the training (apart from the fact that the training is unsupervised), and its averaging across
          the entire dataset (which includes the IN-BAG observations) is a valid, independent evaluation procedure;  
        . This measure is only comparable across different LID estimates computed from the same reference subsample of the same data and
          assessed on the same OOB sample associated with the reference subsample in question, and using the same neighbourhood around
          each query. In other words, fix the observations and compare GoFs for different values of the LID parameter of the distribution.
          This means that this measure cannot be used to compare ensembles produced with different subsampling rates or no. of members.
          It also means that the neighbourhood around each query used here cannot be different when comparing different LID estimates,
          even if these different estimates were computed using different neighbourhood sizes for estimators with such a parameter
          (e.g. 2NN and MLE); 
        . This measure does NOT work in the presence of data duplicates, which must be removed (if any) prior to calling this function;
        . Choosing dist_assumption = "Weibull" requires len(subsample_indexes) <= 0.5*data_size, because the IN-BAG subsample encoded
          by <subsample_indexes> cannot be larger than the OOB subsample (which needs to be downsized to match the size of the IN-BAG).
    '''
    data_size = data_array.shape[0] # Number of observations in the dataset
    mask = False*np.ones(data_size, dtype = bool)
    mask[subsample_indexes] = True # Boolean mask where True indicates that the corresponding data observation is part of the IN-BAG, False otherwise (OOB)
    OOB_subsample_indexes = np.array([x for x in range(data_size)])[mask == False] # Indexes of observations NOT contained in the IN-BAG (i.e., OOB)
    if (dist_assumption == "Weibull"):
        if (NN_size_for_w < len(subsample_indexes) <= len(OOB_subsample_indexes)):
            rand_gen = np.random.RandomState(rand_gen_seed) # Random generator
            OOB_downsampling_indexes = rand_gen.choice(len(OOB_subsample_indexes), size=len(subsample_indexes), replace=False)
            OOB_subsample_indexes = OOB_subsample_indexes[OOB_downsampling_indexes] # Downsampled OOB subsample matching the size of the IN-BAG subsample
            mask = False*np.ones(data_size, dtype = bool)
            mask[OOB_subsample_indexes] = True # Boolean mask where True indicates that the corresponding observation is in the downsampled OOB subsample, False otherwise
            not_OOB_subsample_indexes = np.array([x for x in range(data_size)])[mask == False] # Indexes of observations NOT contained in the downsampled OOB subsample
        else: # NOT possible to downsample the OOB to match the IN-BAG, or to obtain the required number of neighbours from the downsampled OOB
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Abort - Weibull OOB GoF: IN-BAG Cannot Be Larger than the OOB (Sampling Rate Must Be <= 0.5) and Must Be Larger than the Required Neighbourhood Size")
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            # return None, None
            return None
    else: # Tail distribution assumed
        not_OOB_subsample_indexes = subsample_indexes # In this case the OOB is NOT downsampled, so what is not in OOB must necessarily be the IN-BAG (encoded by <subsample_indexes>)  
    data_array_OOB = data_array[OOB_subsample_indexes] # OOB data subsample (possibly downsampled to match the IN-BAG size in case of Weibull dist. assumption)
    nbrs = NearestNeighbors(n_neighbors=NN_size_for_w+1, algorithm='auto').fit(data_array_OOB) # Reference database for the NN Oracle is the OOB subsample (possibly downsampled)
    distances_OOB, *_ = nbrs.kneighbors(data_array_OOB)  # Since these query observations are in the reference database (OOB, possibly downsampled),
                                                         # then the returned 1st NN of each observation is the observation itself (with dist = 0)
    distances_not_OOB, *_ = nbrs.kneighbors(data_array[not_OOB_subsample_indexes])  # Since these query observations are NOT in the reference database (OOB), then
                                                                                    # the returned 1st NN of each observation is NOT the observation itself.
                                                                                    # NOTE: If/when the OOB is downsampled (under the Weibull dist. assumption), then these queries 
                                                                                    # will no longer be strictly limited to the IN-BAG subsample indexed by <subsample_indexes> 
    nn_dist = np.zeros([data_size, NN_size_for_w])
    nn_dist[not_OOB_subsample_indexes] = distances_not_OOB[:, :-1] # Subsample observations are not their own 1st neighbour (keeps k NNs indexed from 0 to k-1)
    nn_dist[OOB_subsample_indexes] = distances_OOB[:, 1:] # Discard own observation as its 1st NN to keep the real 1st NN as such (keeps k NNs indexed from 1 to k)
    no_estimation_types = 1 if lid_estimates.ndim == 1 else lid_estimates.shape[0]
    # OOB_GoFs = np.zeros([no_estimation_types,1])
    OOB_GoFp = np.zeros([no_estimation_types,1])
    for iter_est in range(no_estimation_types):
        lid_est = lid_estimates if lid_estimates.ndim == 1 else lid_estimates[iter_est] # Make sure lid_est is a 1D numpy array
        lid_est = np.maximum(lid_est, min_LID_threshold) # Make sure there aren't non-positive LID estimates
        if (dist_assumption == "Weibull"):
            scale_param = Weibull_Scale_Estimator(data_array, lid_est, subsample_indexes, NN_size_for_w = NN_size_for_w) # Optimal scale par. given the fixed shape par. (candidate LID)
                                                                                                                         # for each observation, both computed w.r.t. the same subsample  
        for query_obs in range(data_size):
            w = nn_dist[query_obs,-1]
            candidate_LID = lid_est[query_obs]
            if (GoF_type == "CVM"): # Cramer-Von-Mises
                if (dist_assumption == "Weibull"):
                    scale_par = scale_param[query_obs]
                    CVM = cramervonmises(nn_dist[query_obs], Weibull_cdf_distance_from_query, args=(scale_par,candidate_LID))
                else: # Lower Tail Distribution Assumed by Default
                    CVM = cramervonmises(nn_dist[query_obs], cdf_distance_from_query, args=(w,candidate_LID))
                # OOB_GoFs[iter_est] += CVM.statistic / data_size
                OOB_GoFp[iter_est] += CVM.pvalue / data_size
            elif (GoF_type == "KS"): # Kolmogorov-Smirnov
                if (dist_assumption == "Weibull"):
                    scale_par = scale_param[query_obs]
                    KS = kstest(nn_dist[query_obs], Weibull_cdf_distance_from_query, args=(scale_par,candidate_LID))
                else: # Lower Tail Distribution Assumed by Default
                    KS = kstest(nn_dist[query_obs], cdf_distance_from_query, args=(w,candidate_LID))
                # OOB_GoFs[iter_est] += KS.statistic / data_size
                OOB_GoFp[iter_est] += KS.pvalue / data_size
            else:
                # OOB_GoFs[iter_est] = None
                OOB_GoFp[iter_est] = None
    # return OOB_GoFs, OOB_GoFp
    return OOB_GoFp


# *** SUBSAMPLING-BASED ENSEMBLE (BAGGING) OF LID ESTIMATORS ***:

def lid_Bagging(data_array, *, ensemble_size = 30, subsample_rate = 0.3, rand_gen_seed = None, return_OOB_LE = False, LE_type = "SNL", NN_size_for_w = 7, dist_assumption = "Tail", lid_Estimator = two_NN_LID_Estimator, **kwargs):
    '''
    Perform Subsampling-Based Ensemble (Bagging) of LID Estimates for an Arbitrary Estimator
      INPUTS (user-defined arguments):
        . data_array:      Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a real-valued (float type) vector;
        . ensemble_size:   Number of base members, i.e. different subsamples, used in the bagging ensemble;
        . subsample_rate:  Fraction of dataset observations in each subsample;
        . rand_gen_seed:   Seed for random generator (If None, then clock is used);
        . return_OOB_LE:   Whether (True) or not (False, default) a mean OOB likelihood or alternative goodness of fit measure should be computed/returned.
                           If True, an estimate of the variance of the LID estimator in use, computed across subsamples (bags) for each LID observation
                           and then averaged over all LID observations, is also returned as a complementary assessment measure of the resulting ensemble;
        . LE_type:         This optional argument will only have an effect when <return_OOB_LE = True>. It specifies the type of evaluation measure to be
                           used for OOB LE. The options are: (i) "LL", which corresponds to the regular Log-Likelihood function (a MAXIMISATION function);
                           (ii) "SNL", which corresponds to the Standardized Negative Log-Likelihood function (MINIMISATION OF ABSOLUTE VALUE function).
                           While LL uses the density f(d) = (LID/w)*(d/w)^{LID-1} (conditional to d<=w) directly as log(f(d)), the latter (default) uses
                           the corresponding cdf F(d)=(d/w)^{LID} as -log(F(d)) - 1; (iii) "KS", which corresponds to the p-value associated with the
                           Komolgorov-Smirnov GoF test for the cdf F(d) above averaged across observations and bags (a MAXIMISATION function); "CVM", which
                           is the same as KS except that the Cramer-Von_Mises GoF test is used in lieu of the KS test (also a MAXIMISATION function);
        . NN_size_for_w:   This optional argument will only have an effect when <return_OOB_LE = True>. It specifies the number of neighbours used specifically
                           for evaluation, namely, to determine the OOB LE/GoF of <lid_estimates> based on the assumed theoretical distribution of distances 
                           within a small radius w around a query. <NN_size_for_w> needs to be a small integer value because LID and the parametric distribution
                           used for GoF/LE testing may hold true only within a very small region around a query. On the other hand, it cannot be too small such
                           that there isn't enough representability in the sample. NOTE: This argument should NOT BE CONFUSED with the neighbourhood size that
                           may be required as a hyper-parameter by the particular estimator of choice (e.g. <neighbourhood_size> in <two_NN_LID_Estimator()>
                           or any other NN-based estimator alike), i.e., as a possible argument of <lid_estimator>. The latter can be passed within **kwargs;
        . dist_assumption: This optional argument will only have an effect when <return_OOB_LE = True>. It specifies the particular theoretical density
                           function of distances around a query that has been hypothetised for the sake of evaluation: "Tail" stands for the asymptotic
                           lower tail distribution, i.e., f(d) = (LID/w)*(d/w)^{LID-1} (conditional to d<=w), whose CDF is given by F(d)=(d/w)^LID, whereas
                           "Weibull" stands for the Weibull distribution assumption, with cdf given by F(d) = 1 - exp(-(d/scale)^LID). NOTE: The latter
                           requires that the IN-BAG subsample encoded by <subsample_indexes> be no larger than the OOB subsample (which under this distribution
                           needs to be downsized to match the size of the IN-BAG during evaluation), or in other words, IN-BAG cannot be larger than half of
                           the dataset (i.e., the underlying subsampling rate <subsample_rate> needs to be <= 0.5);
        . lid_Estimator:   Function that returns a numpy array with LID estimates for the observations in <data_array> (default is <two_NN_LID_Estimator>).
                           This function must be able to compute LID estimates for all the observations in <data_array> W.R.T. A SUBSET OF SUCH OBSERVATIONS.
                           It must have input arguments in the form (data_array, subsample_indexes, ...), where <...> are any specific keyword argument(s)
                           used by the particular estimator of choice, whereas <subsample_indexes> are row indexes of a subset of the dataset observations
                           to be used as a reference subsample for computation of LIDs. The LID of each observation is computed w.r.t. the subsample only
                           in such a way that out of subsample observations do NOT affect LIDs of any other observation (within or outside the subsample);
        . **kwargs:        Any specific (keyword) argument(s) used by the particular estimator of choice (<lid_Estimator>) other than the common/standard
                           arguments <data_array> and <subsample_indexes>. For instance, <neighbourhood_size> and/or <perc_deleted> in <two_NN_LID_Estimator()>.
      OUTPUTS:
        . Bagged_LID_estimates: Bagged ensemble of the LID estimates from each subsample (see note below);
        . Mean_OOB_LE (opt.): Mean OOB evaluation as a goodness of fit measure of the ensemble (only if return_OOB_LE == True; else None is returned);
        - Variance_of_Estimator (opt.): An estimate of the variance component of the estimation error (as decomposed via the bias-variance decomposition). It
                                        is computed as the (unbiased) variance of the LID estimates of each individual observation across subsamples (bags),
                                        which is then averaged over all LID observations. It is only returned if return_OOB_LE == True; else None is returned;
        - CV_of_Estimator (opt.): Similar to <Variance_of_Estimator>, but measures the Relative Standard Deviation (RSD, so-called Coefficient of Variance - CV,
                                  which is a standardised measure of dispersion) instead of variance. Specifically, it is computed as the ratio of the Standard
                                  Deviation to the Mean of the LID estimates of each individual observation across subsamples (bags), which is then averaged
                                  over all LID observations. It is only returned if return_OOB_LE == True; else None is returned.
      NOTE:
        . When the LID estimator function provided as input (via the <lid_Estimator> argument) returns multiple types of estimation (e.g. raw and smoothed
          estimates returned by the default estimator, <two_NN_LID_Estimator>), then all of these estimations are independently bagged and returned. These
          multiple types of estimates are expected to be returned concatenated in a single numpy array of shape (no of estimation types, data size).
    '''
    rand_gen = np.random.RandomState(rand_gen_seed) # Random generator
    data_size = data_array.shape[0] # Number of observations in the dataset
    for subs in range(ensemble_size):
        subsample_indexes = rand_gen.choice(data_size, size=np.ceil(subsample_rate*data_size).astype(int), replace=False) # Random index subsampling without replacement
        LID_estimates = lid_Estimator(data_array, subsample_indexes, **kwargs) # Return one or more type(s) of LID estimates (concatenated as a single numpy array) for the dataset w.r.t. the subsample
        if subs == 0: # Variable initialisations
            Sum_LID_estimates = np.zeros(LID_estimates.shape)
            Sum_Squares_LID_estimates = np.zeros(LID_estimates.shape)
            no_estimation_types = 1 if LID_estimates.ndim == 1 else LID_estimates.shape[0]
            Mean_OOB_LE = np.zeros((no_estimation_types,1))
        Sum_LID_estimates += LID_estimates
        Sum_Squares_LID_estimates += (LID_estimates*LID_estimates)
        if return_OOB_LE == True:
            if (LE_type == "SNL" or LE_type == "LL"):
                OOB_LE = OOB_Likelihood(data_array = data_array, lid_estimates = LID_estimates, subsample_indexes = subsample_indexes, NN_size_for_w = NN_size_for_w, dist_assumption = dist_assumption, LE_type = LE_type, rand_gen_seed = rand_gen_seed, **kwargs)
            elif (LE_type == "CVM" or LE_type == "KS"):
                OOB_LE = OOB_Goodness_of_Fit(data_array = data_array, lid_estimates = LID_estimates, subsample_indexes = subsample_indexes, NN_size_for_w = NN_size_for_w, dist_assumption = dist_assumption, GoF_type = LE_type, rand_gen_seed = rand_gen_seed, **kwargs)    
            Mean_OOB_LE += OOB_LE/ensemble_size # Mean OOB LE evaluation across all the multiple subsamples/bags (final result to be completed at the end of the ensemble loop)
    Bagged_LID_estimates = Sum_LID_estimates/ensemble_size # Bagged version of the LID estimates
    LID_var_thru_bags = (Sum_Squares_LID_estimates - (Sum_LID_estimates*Sum_LID_estimates) / ensemble_size) / (ensemble_size-1) # NB. If catastrophic cancellation is noticed: replace this with a more numerically stable method for
                                                                                                                                #     iterative variance computation, such as Welford's algorithm
    LID_RSD_thru_bags = np.sqrt(LID_var_thru_bags) / (Bagged_LID_estimates) # This is the Relative Standard Deviation - RSD (so-called Coefficient of Variation - CV), which is the ratio of the standard deviation of LID (for each
                                                                            # observation, across the multiple subsamples) to the mean. This is a standardised measure of dispersion (useful for datasets with very different LIDs)
    Variance_of_Estimator = np.mean(LID_var_thru_bags, axis = 1) if (no_estimation_types > 1) else np.mean(LID_var_thru_bags) # Expected variance of the estimator, i.e. variance (across bags) averaged over all LID observations
    CV_of_Estimator = np.mean(LID_RSD_thru_bags, axis = 1) if (no_estimation_types > 1) else np.mean(LID_RSD_thru_bags) # Expected Coeff. of Var. (CV) of the estimator, i.e. the Relative Std (RSD across bags) averaged over all LID observations
    if return_OOB_LE == True: # Shapes: (no of estimation types, data size), (no of estimation types, 1), and (no of estimation types, 1)
        return Bagged_LID_estimates, Mean_OOB_LE, Variance_of_Estimator.reshape(-1,1), CV_of_Estimator.reshape(-1,1)
    else:
        return Bagged_LID_estimates, None, None, None


# *** POST-PROCESSED SMOOTHING OF LID ESTIMATES (W.R.T. THE WHOLE DATASET, RATHER THAN ANY SUBSAMPLE) ***:

def lid_Smoothing(data_array, lid_estimates, neighbourhood_size):
    '''
    Perform Smoothing of LID Estimates Taking the LID Estimate OF Each Observation as the Mean of the LIDs of Its Neighbours.
    It Can be Used, e.g., to Perform Smoothing of Bagged Estimates from lid_Bagging()
      INPUTS (user-defined arguments):
        . data_array:         Data set as a 2D numpy array where each row (axis 0) indexes an observation represented as a real-valued (float type) vector;
        . lid_estimates:      1D numpy array with LID estimates for the observations in <data_array>;
        . neighbourhood_size: Number of neighbours used to determine the local region around a query observation, within which the LID of
                              the query is smoothed by taking the mean of the LIDs of the neighbours (neighbourhood_size >= 3).
      OUTPUTS:
        . Smoothed_LID_estimates: 1D numpy array with smoothed LID estimates for the observations in <data_array>.
      NOTE:
        . Different from smoothed estimates that may have been computed w.r.t. a subsample of the data, e.g. from <two_NN_LID_Estimator()>, the smoothing
          here considers neighbours from the whole dataset, rather than a subsample
    '''
    data_size = data_array.shape[0] # Number of observations in the dataset
    nbrs = NearestNeighbors(n_neighbors=neighbourhood_size, algorithm='auto').fit(data_array) # Nearest Neighbours Oracle
    _, KNN_indices, *_ = nbrs.kneighbors(data_array)    # Since query observations are provided as input, and they are all contained in the reference database used by the
                                                        # NN Oracle (actually, in this case, the query set and the reference database are exactly the same), then the 1st NN
                                                        # of each query observation is the observation itself (with dist = 0). Should the input argument had been ommitted,
                                                        # i.e, nbrs.kneighbors(), then the distances and indices of the neighbours of the reference data themselves would
                                                        # be similarly returned; however, in that case, each observation would not appear as its own neighbour in the result
    return np.array([np.mean(lid_estimates[KNN_indices[obs]]) for obs in range(data_size)]) # Smoothed version of the estimates provided as input. Notice that the
                                                                                            # neighbours of each query observation here (whose mean LID is the smoothed
                                                                                            # LID estimate of the query) are the global neighbours (rather than subsample
                                                                                            # neighbours) pre-computed using the whole dataset. Also, notice that the LID
                                                                                            # of the query is included in the mean (since the query appears as its own neighbour)


# *** EXTERNAL EVALUATION OF LID ESTIMATES (W.R.T. A GIVEN GROUND-TRUTH) ***:

def external_Evaluation(lid_estimates, lid_ground_truth):
    '''
    Compute Multiple External Evaluation Measures of LID Estimates w.r.t. a Given Ground-Truth.
      INPUTS (user-defined arguments):
        . lid_estimates:      Numpy array (float type) containing candidate LID estimates for every observation in a dataset. If there is only
                              1 single candidate estimator to be assessed, then <lid_estimates> can be a 1D numpy array. If there are multiple 
                              such candidate estimators, then <lid_estimates> will be a 2D numpy array of shape (no of estimators, data size).
                              In this case, each row of <lid_estimates> will contain the candidate LID estimates of the corresponding candidate
                              estimator for every observation in the dataset;
        . lid_ground_truth:   The ground truth (reference) LIDs against which <lid_estimates> should be compared. It is a 1D numpy array,
                              regardless of the number of candidate estimators contained in <lid_estimates>.                           
      OUTPUTS:
        . MSE:  1D numpy array with the Mean Squared Error between each candidate estimator and the ground truth:
                MSE = 1/n sum_{i=1}^n (lid_estimates_i - lid_ground_truth_i)^2;
                Note that for a constant ground truth, MSE can be decomposed as Var + Bias^2, where <Bias> and <Var> are also returned (see below); 
        . Bias: 1D numpy array with the Mean difference between each candidate estimator and the ground truth, which is the difference between
                the mean value of the estimator and the mean value of the ground truth, i.e.:
                Bias = 1/n sum_{i=1}^n (lid_estimates_i - lid_ground_truth_i) =
                     = 1/n sum_{i=1}^n (lid_estimates_i) - 1/n sum_{i=1}^n (lid_ground_truth_i);
        . Var:  1D numpy array with the variance of the LID estimates: 1/n sum_{i=1}^n (lid_estimates_i - mean_lid_estimates)^2, where
                mean_lid_estimates = 1/n sum_{i=1}^n (lid_estimates_i). Notice that this measure doesn't need the ground truth, but it
                is computed and returned here for the sake of completeness (given the expected Bias-Var decomposition of MSE);
        . Kendall_Tau: 1D numpy array with the Kendall's tau measure of correspondence between the LID ranking induced by each estimator
                and the LID ranking induced by the ground-truth. By default, the scipy.stats function kendalltau() used here implements
                "the 1945 'tau-b' version of Kendall's tau, which can account for ties and which reduces to the 1938 'tau-a' version
                in absence of ties. It ranges within [-1,+1];
        . Spearman: 1D numpy array with the Spearman rank-order correlation (a nonparametric measure of the monotonicity of the relationship
                between two sequences) between each estimator's LIDs and the LIDs according to the ground-truth. It ranges within [-1,+1].
    '''
    no_estimation_types = 1 if lid_estimates.ndim == 1 else lid_estimates.shape[0]
    if no_estimation_types == 1: # There is only one estimator, so all the measures returned are scalar values
        Bias = np.mean(lid_estimates - lid_ground_truth)
        Var = np.power(np.std(lid_estimates),2)
        MSE = np.mean(np.power(lid_estimates - lid_ground_truth,2))
        Kendall_Tau, *_ = kendalltau(lid_estimates, lid_ground_truth)
        Spearman, *_ = spearmanr(lid_estimates, lid_ground_truth)
    else: # There are multiple estimators, so all the measures returned are 1D numpy arrays
        Bias = np.mean(lid_estimates - lid_ground_truth, axis = 1)
        Var = np.power(np.std(lid_estimates, axis = 1),2)
        MSE = np.mean(np.power(lid_estimates - lid_ground_truth,2), axis = 1)
        Kendall_Tau = np.array([kendalltau(lid_estimates[iter], lid_ground_truth)[0] for iter in range(no_estimation_types)])
        Spearman = np.array([spearmanr(lid_estimates[iter], lid_ground_truth)[0] for iter in range(no_estimation_types)])
    return MSE, Bias, Var, Kendall_Tau, Spearman
