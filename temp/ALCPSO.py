import numpy as np
from function import continueFunction as cF
import copy

for run in range(4):
    func_dim = 30
    func_name = "Rastrigin"
    pop_size = 20
    fe_max = 200000
    challenging_times_max = 2
    leader_lifespan_max = 60
    X_lb = -5.12 * np.ones((pop_size, func_dim))
    X_ub = 5.12 * np.ones((pop_size, func_dim))

    X = (X_ub - X_lb) * np.random.random_sample((pop_size, func_dim)) + X_lb
    V_ub = 0.5 * (X_ub - X_lb)
    V_lb = -1 * V_ub

    V = np.zeros_like(X)
    feval = getattr(cF, func_name)
    y = feval(X)
    fe_num = 0
    fe_num += pop_size

    X_pb, y_pb = copy.deepcopy(X), copy.deepcopy(y)

    opt_y = np.min(y_pb)
    opt_x = X_pb[np.argmin(y_pb), :]

    leader, leader_y = copy.deepcopy(opt_x), copy.deepcopy(opt_y)
    leader_age = 0
    leader_lifespan = 60

    while fe_num < fe_max:
        indicator_glp = False
        indicator_flp = False
        indicator_plp = False
        y_pb_bak = copy.deepcopy(y_pb)
        for pi in range(pop_size):
            # update and limit V
            V[pi, :] = 0.4 * V[pi, :] + 2.0 * np.random.random_sample((func_dim, )) * (
                X_pb[pi, :] - X[pi, :]) + 2.0 * np.random.random_sample((func_dim, )) * (leader - X[pi, :])
            V[pi, :][V[pi, :] < V_lb[pi, :]] = V_lb[pi, :][V[pi, :] < V_lb[pi, :]]
            V[pi, :][V[pi, :] > V_ub[pi, :]] = V_ub[pi, :][V[pi, :] > V_ub[pi, :]]
            # update X
            X[pi, :] = X[pi, :] + V[pi, :]
            X[pi, :][X[pi, :] < X_lb[pi, :]] = X_lb[pi, :][X[pi, :] < X_lb[pi, :]]
            X[pi, :][X[pi, :] > X_ub[pi, :]] = X_ub[pi, :][X[pi, :] > X_ub[pi, :]]

            y[pi] = feval(X[pi, :][np.newaxis, :])
            fe_num += 1
            if y[pi] < leader_y:
                leader_y = copy.deepcopy(y[pi])
                leader = copy.deepcopy(X[pi, :])
                indicator_plp = True
            if y[pi] < y_pb[pi]:
                X_pb[pi, :] = X[pi, :]
                y_pb[pi] = y[pi]            
                if y[pi] < opt_y:
                    opt_y = copy.deepcopy(y[pi])
                    opt_x = copy.deepcopy(X[pi, :])
                    indicator_glp = True
            if np.sum(y_pb) < np.sum(y_pb_bak):
                indicator_flp = True
            
            leader_age = leader_age + 1
            if indicator_glp:
                leader_lifespan = leader_lifespan + 2
            elif indicator_flp:
                leader_lifespan = leader_lifespan + 1
            elif indicator_plp:
                pass
            else:
                leader_lifespan = leader_lifespan - 1
            # print(leader_age, leader_lifespan)
            if leader_age >= leader_lifespan:
                flag_same = True
                challenger = copy.deepcopy(leader)
                for fd_ind in range(func_dim):
                    if np.random.random() < (1/func_dim):
                        challenger[fd_ind] = (X_ub[0, fd_ind] - X_lb[0, fd_ind]) * np.random.random_sample() + X_lb[0, fd_ind]
                        flag_same = False
                if flag_same:
                    challenger[np.random.randint(func_dim)] = (X_ub[0, fd_ind] - X_lb[0, fd_ind]) * np.random.random_sample() + X_lb[0, fd_ind]
                X_bak, V_bak = copy.deepcopy(X), copy.deepcopy(V)
                challenger_y = feval(challenger[np.newaxis, :])
                fe_num += 1
                if challenger_y < opt_y:
                    opt_y = copy.deepcopy(challenger_y)
                    opt_x = copy.deepcopy(challenger)
                flag_improve = False
                for try_ind in range(challenging_times_max):
                    for pi in range(pop_size):
                        V[pi, :] = 0.4 * V[pi, :] + 2.0 * np.random.random_sample((func_dim, )) * (
                        X_pb[pi, :] - X[pi, :]) + 2.0 * np.random.random_sample((func_dim, )) * (challenger - X[pi, :])
                        V[pi, :][V[pi, :] < V_lb[pi, :]] = V_lb[pi, :][V[pi, :] < V_lb[pi, :]]
                        V[pi, :][V[pi, :] > V_ub[pi, :]] = V_ub[pi, :][V[pi, :] > V_ub[pi, :]]
                        # update X
                        X[pi, :] = X[pi, :] + V[pi, :]
                        X[pi, :][X[pi, :] < X_lb[pi, :]] = X_lb[pi, :][X[pi, :] < X_lb[pi, :]]
                        X[pi, :][X[pi, :] > X_ub[pi, :]] = X_ub[pi, :][X[pi, :] > X_ub[pi, :]]

                        y[pi] = feval(X[pi, :][np.newaxis, :])
                        fe_num += 1
                        
                        if y[pi] < y_pb[pi]:
                            y_pb[pi] = copy.deepcopy(y[pi])
                            X_pb[pi, :] = copy.deepcopy(X[pi, :])
                            flag_improve = True
                            if y[pi] < opt_y:
                                opt_y = copy.deepcopy(y[pi])
                                opt_x = copy.deepcopy(X[pi, :])
                            
                        if y[pi] < challenger_y:
                            challenger_y = copy.deepcopy(y[pi])
                            challenger = copy.deepcopy(X[pi, :])
                    if flag_improve:
                        leader = copy.deepcopy(challenger)
                        leader_y = copy.deepcopy(challenger_y)
                        leader_age = 0
                        leader_lifespan = 60
                        break
                if not flag_improve:
                    X = copy.deepcopy(X_bak)
                    V = copy.deepcopy(V_bak)
                    leader_age = leader_lifespan - 1
    print('Run:{0} BestV: {1} \n'.format(run, opt_y))
print(opt_x)


