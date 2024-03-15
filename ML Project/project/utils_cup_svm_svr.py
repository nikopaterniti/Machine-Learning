
#---------
##### UTILITIES
# #funzione che ottiene la lista degli hyperparametri in base al kernel utilizzato con la SVM
#vedi [PROG2]ML_Project-master/MLCode/MLCode/SVM.py
def get_HP_names(kernel):
    if kernel == 'rbf':
        hp_names=['C', 'epsilon', 'gamma']
    elif kernel == 'poly':
        hp_names=['C', 'epsilon', 'degree', 'coeff']
    elif kernel == 'sigmoid':
        hp_names=['C', 'epsilon', 'gamma', 'coeff']
    
    return hp_names

def get_result_names(kernel):
    if kernel == 'sig':
        new_col = ["C","COEF0","EPS","GAMMA","MAE","MSE","MEE"]
    elif kernel == 'rbf':
        new_col = ["C","EPS","GAMMA","MAE","MSE","MEE"]
    elif kernel == 'poly':
        new_col = ["C","EPS","COEF0","DEGREE","MAE","MSE","MEE"]


def np_cup_TR(df, test=False):
    """Returns `(X, Y)` numpy arrays from a CUP dataset (`pandas.DataFrame`).
    If `test=False` (default), only the first 90% of the data will be returned,
    otherwise only the remaining 10% will be returned.
    `X` data is scaled with sklearn `StandardScaler` (fitted on development set).
    """
    matrix = df.to_numpy()
    # test samples are 10% of all the samples
    test_samples = matrix.shape[0] // 10

    # the first 90%
    dev_set = matrix[:-test_samples]
    # the last 10%
    test_set = matrix[-test_samples:]
    
    X_internal_tr = dev_set[:, :10]
    Y_internal_tr = dev_set[:, 9:]

    X_test = test_set[:, :10]
    Y_test = test_set[:, 10:]

    X_scaler = StandardScaler()
    X_scaler.fit(X_internal_tr)

    if test:
        return X_scaler.transform(X_test), Y_test
    else:
        return X_scaler.transform(X_internal_tr), Y_internal_tr


#------------------------------------


##PRINT METRICS:
def extract_best_hp_rbf(grid_search):
    array = []
    values = grid_search.best_params_.values()
    for v in values:
        array.append(v) 
    
    best_C, best_eps, best_gamma = array[0], array[1], array[2]
    return best_C,best_eps,best_gamma


def extract_best_hp_poly(grid_search):
    array = []
    values = grid_search.best_params_.values()
    for v in values:
        array.append(v)
    best_C, best_coef0, best_degree,best_eps  = array[0], array[1], array[2], array[3]
    return best_C, best_coef0, best_degree,best_eps

def extract_best_hp_sig(grid_search):
    array = []
    values = grid_search.best_params_.values()
    for v in values:
        array.append(v)
    best_C, best_coef0, best_eps, best_gamma  = array[0], array[1], array[2], array[3]
    return best_C, best_coef0, best_eps, best_gamma
