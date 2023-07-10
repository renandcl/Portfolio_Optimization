import pandas as pd # type: ignore
import numpy as np
import pulp # type: ignore
import copy
import time

import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.optimize import OptimizeResult


def CVaR_optimization_nlp(scenarios: pd.DataFrame, initial_weights: np.ndarray = None, beta = 0.95, expected_return = 0.001) -> Tuple[OptimizeResult, pd.DataFrame]:
    """ CVaR optimization using non linear programming optimizer

        Parameters:
        ----------
        scenarios: pd.DataFrame
            scenarios dataframe
        initial_weights: np.ndarray
            vector of weights
        beta: float, optional
            beta value for confidence level
        expected_return: float, optional
            expected return value

        Returns:
        -------
        OptimizeResult
            optimization result
        pd.DataFrame
            investment decisions

        Examples:
        ---------
        >>> scenarios

    """

    if not initial_weights:
        initial_weights = np.random.dirichlet(np.ones(len(scenarios.columns)))
        
    df = pd.DataFrame()

    def CVaR_obj(p_x):
        df['portfolio_return'] = -scenarios.apply(lambda x: np.dot(x.values,p_x), axis=1)
        VaR = df.portfolio_return.quantile(beta)
        df['CVaR'] = df.portfolio_return.apply(lambda x: np.where(x - VaR > 0, x, 0))
        CVaR = df.CVaR.loc[df.CVaR > VaR].mean()
        return CVaR

    bnds = tuple([(0,1) for _ in range(len(scenarios.columns))])
    cons1 = {'type': 'eq', 'fun': lambda x: -np.sum(x) + 1}
    cons2 = {'type': 'eq', 'fun': lambda x: np.dot(scenarios.mean(),x) - expected_return}

    CVaR_opt = optimize.minimize(CVaR_obj, initial_weights, method='SLSQP', bounds=bnds, constraints=(cons1,cons2), options={'disp': True, 'maxiter': 1000, 'ftol': 1e-8})	

    return CVaR_opt, df


def pulp_CVaR_portfolio_optmization_model(params):
    
    pulp_model_MILP = pulp.LpProblem("Portfolio CVaR optimization MILP", pulp.LpMaximize)

    x = {}
    for j in params['components']:
        x[j] = pulp.LpVariable(f'x_{j}', lowBound=0, cat='Integer')

    y = {}
    for t in params['scenarios']:
        y[t] = pulp.LpVariable(f'y_{t}', lowBound=0, cat='Continuous')

    z = {}
    for j in params['components']:
        z[j] = pulp.LpVariable(f'z_{j}', cat='Binary')

    a = {}
    for j in params['components']:
        a[j] = pulp.LpVariable(f'a_{j}', cat='Binary')

    d = {}
    for j in params['components']:
        d[j] = pulp.LpVariable(f'd_{j}', lowBound=0, cat='Continuous')

    eta = pulp.LpVariable(f'eta', cat='Continuous')

    pulp_model_MILP += eta - (1/params['beta'][0]) * (1/params['n_scenarios'][0]) * pulp.lpSum([y[t] for t in params['scenarios']])

    for t in params['scenarios']:
        pulp_model_MILP += eta + pulp.lpSum([ - params['scenarios_returns'][(t,j)]*params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) <= y[t]

    pulp_model_MILP += pulp.lpSum([params['mean_returns'][j]*params['prices'][j]*x[j] - params['taxes'][j]*params['prices'][j]*d[j] - params['fees'][j]*z[j] for j in params['components']]) >= params['expected_return_percentage'][0]*pulp.lpSum([params['prices'][j]*x[j] for j in params['components']])

    pulp_model_MILP += pulp.lpSum([params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) <= pulp.lpSum([params['prices'][j]*params['initial_lots'][j] for j in params['components']]) + params['additional_capital'][0]

    pulp_model_MILP += pulp.lpSum([params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) >= (pulp.lpSum([params['prices'][j]*params['initial_lots'][j] for j in params['components']]) + params['additional_capital'][0]) * (1 - params['remain_pct'][0])

    for j in params['components']:
        pulp_model_MILP += x[j] * params['prices'][j] >= params['lower_price_boundaries'][j] * a[j]

    for j in params['components']:
        pulp_model_MILP += x[j] * params['prices'][j] <=  a[j] * ( pulp.lpSum([params['prices'][i]*params['initial_lots'][i] for i in params['components']]) + params['additional_capital'][0])

    for j in params['components']:
        pulp_model_MILP += x[j] <= params['upper_lot_boundaries'][j]

    for j in params['components']:
        pulp_model_MILP += d[j] >= (x[j] - params['initial_lots'][j])

    for j in params['components']:
        pulp_model_MILP += d[j] >= -(x[j] - params['initial_lots'][j])

    for j in params['components']:
        pulp_model_MILP += d[j] <= params['max_lot_rebalance'][j]*z[j]
  
    pulp_model_MILP += pulp.lpSum([a[j] for j in params['components']]) <= params['total_components'][0]

    return pulp_model_MILP, x, y, d, z, a, eta


def pulp_CVaR_portfolio_optmization_relaxed_model(params):
    
    # # PulP model for portfolio optimization
    pulp_model_relaxed = pulp.LpProblem("Portfolio CVaR optimization Relaxed", pulp.LpMaximize)

    # # define set for component names, j
    # set components;

    # # define set for component names, t
    # set scenarios;

    # # define parameter for number of scenarios, T
    # param n_scenarios;

    # # define parameter with historical data, mu[t,j]
    # param scenarios_returns {scenarios, components};

    # # define parameter with last component price, q[j]
    # param prices {components};

    # # define income tax per component, c[j]
    # param taxes {components};

    # # define brokerage tax per component, f[j]
    # param fees {components};

    # # define parameter for each component return, mu[j]
    # param mean_returns {components};

    # # define parameter to initial lot for components, kappa_0[j]
    # param initial_lots {components};

    # # define parameter min price for components, L[j]
    # param lower_price_boundaries {components};

    # # define parameter max lot size for components, U[j]
    # param upper_lot_boundaries {components};

    # # define parameter with the lot maximum rebalance, gamma[j]
    # param max_lot_rebalance {components};

    # # define parameter for expected percentage return, mu_0
    # param expected_return_percentage;

    # # define parameter for total additional capital, B
    # param additional_capital;

    # # define parameter for beta confidence level, beta
    # param beta;

    # # define parameter for total components, m
    # param total_components;

    # # define percentage of remain capital
    # param remain_pct;

    # # define variable as integer for number of lots of each component on portfolio, kappa[j]
    # var lot_qte_decisions {components} integer >= 0;  

    x = {}
    for j in params['components']:
        # x[j] = pulp.LpVariable(f'x_{j}', lowBound=0, cat='Integer')
        x[j] = pulp.LpVariable(f'x_{j}', lowBound=0, cat='Continuous')

    # # define variable of net portfolio return on t [max(0,eta - eta_t)], y[t]
    # var conditional_delta_VaR_CVaR{scenarios} >= 0;

    y = {}
    for t in params['scenarios']:
        y[t] = pulp.LpVariable(f'y_{t}', lowBound=0, cat='Continuous')

    # define variable as binary for decisions of component rebalance on portfolio, z[j]
    # var binary_rebalance_decisions {component in components} binary; 

    z = {}
    for j in params['components']:
        # z[j] = pulp.LpVariable(f'z_{j}', cat='Binary')
        z[j] = pulp.LpVariable(f'z_{j}', lowBound=0, upBound=1, cat='Continuous')

    # define variable as binary for component allocation in the portfolio, a[j]
    # var binary_activation {component in components} binary;

    a = {}
    for j in params['components']:
        # a[j] = pulp.LpVariable(f'a_{j}', cat='Binary')
        a[j] = pulp.LpVariable(f'a_{j}', lowBound=0, upBound=1, cat='Continuous')

    # # define variable for lot rebalance, delta[j]
    # var rebalance {components} >= 0;

    d = {}
    for j in params['components']:
        d[j] = pulp.LpVariable(f'd_{j}', lowBound=0, cat='Continuous')

    # # # define variable for remain capital, rc
    # # var remain_capital >=0;

    # rc = pulp.LpVariable(f'remain', lowBound=0, cat='Continuous')

    # # define variable for expected return on beta in optimal portfolio, eta
    # var value_on_beta;

    eta = pulp.LpVariable(f'eta', cat='Continuous')

    # # define objective function
    # maximize F_obj:
    #     value_on_beta - (1/beta) * (1/n_scenarios) * sum{scenario in scenarios} (conditional_delta_VaR_CVaR[scenario]);

    pulp_model_relaxed += eta - (1/params['beta'][0]) * (1/params['n_scenarios'][0]) * pulp.lpSum([y[t] for t in params['scenarios']])

    # # define constraint to portfolio scenarios net return max(0,eta - eta_t) <= 0
    # subject to portfolio_net_returns_constraint {scenario in scenarios}:
    #     value_on_beta - sum{component in components} (scenarios_returns[scenario, component] * prices[component] * lot_qte_decisions[component] + taxes[component] * prices[component] * rebalance[component] + fees[component] * binary_rebalance_decisions[component]) <= conditional_delta_VaR_CVaR[scenario];

    for t in params['scenarios']:
        pulp_model_relaxed += eta + pulp.lpSum([ - params['scenarios_returns'][(t,j)]*params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) <= y[t]

    # # define constraint to portfolio expected return
    # subject to expected_return_constraint:
    #     sum{component in components} (mean_returns[component] * prices[component] * lot_qte_decisions[component] - taxes[component] * prices[component] * rebalance[component] - fees[component] * binary_rebalance_decisions[component]) <= expected_return_percentage * sum{component in components} (prices[component] * lot_qte_decisions[component]); 

    pulp_model_relaxed += pulp.lpSum([params['mean_returns'][j]*params['prices'][j]*x[j] - params['taxes'][j]*params['prices'][j]*d[j] - params['fees'][j]*z[j] for j in params['components']]) >= params['expected_return_percentage'][0]*pulp.lpSum([params['prices'][j]*x[j] for j in params['components']])

    # # define constraint of total capital upper bound
    # subject to total_capital_constraint_upper_bound:
        # sum{component in components} (prices[component] * lot_qte_decisions[component]) <= sum{component in components} (prices[component] * initial_lots[component]) + additional_capital;

    pulp_model_relaxed += pulp.lpSum([params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) <= pulp.lpSum([params['prices'][j]*params['initial_lots'][j] for j in params['components']]) + params['additional_capital'][0]

    # # define constraint of total capital lower bound
    # subject to total_capital_constraint_lower_bound:
        # sum{component in components} (prices[component] * lot_qte_decisions[component]) >= sum{component in components} (prices[component] * initial_lots[component]) + additional_capital * (1 - remain_pct);

    pulp_model_relaxed += pulp.lpSum([params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) >= (pulp.lpSum([params['prices'][j]*params['initial_lots'][j] for j in params['components']]) + params['additional_capital'][0]) * (1 - params['remain_pct'][0])

    # # define constraint to lower bound price of lot
    # subject to price_component_lb {component in components}:
    #     lot_qte_decisions[component] * prices[component] >= lower_price_boundaries [component] * binary_activation[component];

    for j in params['components']:
        pulp_model_relaxed += x[j] * params['prices'][j] >= params['lower_price_boundaries'][j] * a[j]

    # # define constraint big M to activate binary
    # subject to binary_activation_constraint {component in components}:
    #     lot_qte_decisions[component] * prices[component] <= binary_activation[component] * (sum{j in components} (prices[j] * initial_lots[j]) + additional_capital);

    for j in params['components']:
        pulp_model_relaxed += x[j] * params['prices'][j] <=  a[j] * ( pulp.lpSum([params['prices'][i]*params['initial_lots'][i] for i in params['components']]) + params['additional_capital'][0] )


    # # define constraint to upper bound of lot size
    # subject to lot_component_ub {component in components}:
    #     lot_qte_decisions[component] <= upper_lot_boundaries [component];

    for j in params['components']:
        pulp_model_relaxed += x[j] <= params['upper_lot_boundaries'][j]

    # # define constraint to positive lot rebalance
    # subject to lot_rebalance_positive_constraint {component in components}:
    #     rebalance[component] >= (lot_qte_decisions[component] - initial_lots[component]);

    for j in params['components']:
        pulp_model_relaxed += d[j] >= (x[j] - params['initial_lots'][j])

    # # define constraint to non negative lot rebalance
    # subject to lot_rebalance_non_negative_constraint {component in components}:
    #     rebalance[component] >= -(lot_qte_decisions[component] - initial_lots[component]);

    for j in params['components']:
        pulp_model_relaxed += d[j] >= -(x[j] - params['initial_lots'][j])

    # # define constraint to maximum lot rebalance
    # subject to lot_rebalance_constraint {component in components}:
    #     rebalance[component] <= max_lot_rebalance[component] * binary_rebalance_decisions[component];

    for j in params['components']:
        pulp_model_relaxed += d[j] <= params['max_lot_rebalance'][j]*z[j]
        
    # # define constraint to quantity of components
    # subject to quantity_component_constraint:
    #     sum{component in components} (binary_activation[component]) <= total_components;

    pulp_model_relaxed += pulp.lpSum([a[j] for j in params['components']]) <= params['total_components'][0]

    return pulp_model_relaxed, x, y, d, z, a, eta


def test_pulp_portfolio_net_returns_constraint(params, df_log_returns, x_pulp, y_pulp, d_pulp, z_pulp, a_pulp, eta_pulp):
    # Tests Constraints
    pulp_portfolio_net_returns_constraint = (eta_pulp - (df_log_returns*pd.Series(params['prices'])*x_pulp).sum(axis=1) + (pd.Series(params['taxes'])*pd.Series(params['prices'])*d_pulp).sum() + (pd.Series(params['fees'])*z_pulp).sum() <= y_pulp).all()
    pulp_expected_return_constraint = ((pd.Series(params['mean_returns'])*pd.Series(params['prices'])*x_pulp).sum() - (pd.Series(params['taxes'])*pd.Series(params['prices'])*d_pulp).sum() - (pd.Series(params['fees'])*z_pulp).sum() ).mean() >= params['expected_return_percentage'][0] * (pd.Series(params['prices'])*x_pulp).sum()
    pulp_total_capital_ub = (pd.Series(params['prices'])*x_pulp).sum() - (pd.Series(params['taxes'])*pd.Series(params['prices'])*d_pulp).sum() - (pd.Series(params['fees'])*z_pulp).sum() <= (pd.Series(params['prices'])*pd.Series(params['initial_lots'])).sum() + params['additional_capital'][0]
    pulp_total_capital_lb = (pd.Series(params['prices'])*x_pulp).sum() - (pd.Series(params['taxes'])*pd.Series(params['prices'])*d_pulp).sum() - (pd.Series(params['fees'])*z_pulp).sum() >= (pd.Series(params['prices'])*pd.Series(params['initial_lots'])).sum() + params['additional_capital'][0] * (1-params['remain_pct'][0])
    pulp_price_component_lb = (pd.Series(params['prices'])*x_pulp >= pd.Series(params['lower_price_boundaries'])*a_pulp).all()
    pulp_binary_activation_constraint = (x_pulp * pd.Series(params['prices']) <= a_pulp * ((pd.Series(params['initial_lots']) * pd.Series(params['prices'])).sum() + params['additional_capital'][0])).all()
    pulp_lot_component_ub = (x_pulp <= pd.Series(params['upper_lot_boundaries'])).all()
    pulp_lot_rebalance_positive_constraint = (d_pulp >= (x_pulp - pd.Series(params['initial_lots']))).all() # type: ignore
    pulp_lot_rebalance_non_negative_constraint = (d_pulp >= - (x_pulp - pd.Series(params['initial_lots']))).all() # type: ignore
    pulp_lot_rebalance_constraint = (d_pulp <= pd.Series(params['max_lot_rebalance'])*z_pulp).all()
    pulp_quantity_component_constraint = (a_pulp.sum() <= params['total_components'][0])

    print("Portfolio net returns constraint accepted:", pulp_portfolio_net_returns_constraint)
    print("Portfolio expected return constraint accepted:", pulp_expected_return_constraint)
    print("Total capital constraint upper bound accepted:", pulp_total_capital_ub)
    print("Total capital constraint lower bound accepted:", pulp_total_capital_lb)
    print("Price component constraint lower bound accepted:", pulp_price_component_lb)
    print("Binary activation constraint accepted:", pulp_binary_activation_constraint)
    print("Lot component constraint upper bound accepted:", pulp_lot_component_ub)
    print("Lot rebalance constraint positive accepted:", pulp_lot_rebalance_positive_constraint)
    print("Lot rebalance constraint non negative accepted:", pulp_lot_rebalance_non_negative_constraint)
    print("Lot rebalance constraint accepted:", pulp_lot_rebalance_constraint)
    print("Quantity component constraint accepted:", pulp_quantity_component_constraint)


def plot_portfolio(params,pulp_portfolio_returns):
    # histogram of portfolio returns
    ax = sns.histplot(pulp_portfolio_returns, bins=50, kde=True, color='gray', stat='density') # type: ignore

    # vertical line on CVaR
    ax.axvline(pulp_portfolio_returns.quantile(params['beta'][0]), color='black', linestyle='--')
    pulp_cvar_b = pulp_portfolio_returns.loc[pulp_portfolio_returns<pulp_portfolio_returns.quantile(params['beta'][0])].mean()
    ax.axvline(pulp_cvar_b, color='grey', linestyle='--')
    plt.show()


def init_params_config(df_log_returns,df_params,n_assets,n_scenarios,coins,invested_money,mu_0,B,beta,N,pho):
    # filter number of assets
    np.random.seed(42)
    assets = np.random.choice(df_log_returns.columns, n_assets, replace=False)

    # check if any of the coins is in the list of assets and add if not and remove another one not in the list
    for coin in coins:
        if coin not in assets:
            assets = np.append(assets, coin)
            assets = np.delete(assets, 0) if assets[0] not in coins else np.delete(assets, 1)

    # sort assets
    assets = np.sort(assets)

    # filter data
    df_log_returns = df_log_returns.loc[df_log_returns.index >= n_scenarios, assets] # type: ignore
    df_params = df_params.loc[assets]

    # Initial lots invested from coins
    np.random.seed(42)
    df_params.loc[df_params.index.isin(coins), 'kappa'] = pd.Series(data = (np.random.dirichlet(np.ones(len(coins)))*(invested_money)/df_params.q[coins]).astype(int), index = df_params.loc[coins].index)

    # lower price boundaries
    lower_price_boundaries = pd.Series(data = 500, index = df_params.index)
    params = {
        'components': df_log_returns.columns.to_list(),
        'scenarios': df_log_returns.index.to_list(),
        'n_scenarios': [len(df_log_returns.index.to_list())],
        'scenarios_returns': df_log_returns.stack().to_dict(),
        'prices': df_params.q.to_dict(),
        'taxes': df_params.c.to_dict(),
        'fees': df_params.f.to_dict(),
        'mean_returns': df_log_returns.mean().to_dict(),
        'initial_lots': df_params.kappa.to_dict(),
        'upper_lot_boundaries': df_params.U.to_dict(),
        'lower_price_boundaries': lower_price_boundaries.to_dict(),
        'max_lot_rebalance': df_params.gamma.to_dict(),
        'expected_return_percentage': [mu_0],
        'additional_capital': [B],
        'beta': [beta],
        'total_components': [N],
        'remain_pct': [pho]
        }
    
    return params, df_log_returns, df_params


def pulp_CVaR_portfolio_optmization_model_solve(params):

    pulp_model, _, _, _, _, _, _ = pulp_CVaR_portfolio_optmization_model(params)

    # solve
    pulp_model.solve()

    solve_status = pulp.LpStatus[pulp_model.status]
    solve_time = pulp_model.solutionTime
    obj_fun = pulp.value(pulp_model.objective)

    return solve_status, solve_time, obj_fun


def pulp_CVaR_portfolio_optmization_model_solve_full(params,df_log_returns):
    pulp_model, x, y, d, z, a, eta = pulp_CVaR_portfolio_optmization_model(params)

    # solve
    pulp_model.solve()

    # print results
    print(pulp_model.solutionTime)
    print(pulp.LpStatus[pulp_model.status])
    print(pulp.value(pulp_model.objective))

    # Series 
    x_pulp = pd.Series([x[j].value() for j in params['components']], index=params['components'])
    y_pulp = pd.Series([y[t].value() for t in params['scenarios']], index=params['scenarios'])
    d_pulp = pd.Series([d[j].value() for j in params['components']], index=params['components'])
    z_pulp = pd.Series([z[j].value() for j in params['components']], index=params['components'])
    a_pulp = pd.Series([a[j].value() for j in params['components']], index=params['components'])
    x_prices_pulp = pd.Series([x[j].value()*params['prices'][j] for j in params['components']], index=params['components'])
    eta_pulp = eta.value()

    # create dataframe
    pulp_res_comp = pd.DataFrame({'x':x_pulp, 'd':d_pulp, 'z':z_pulp, 'a':a_pulp, 'x_prices':x_prices_pulp})

    pulp_res_comp.loc[z_pulp == 1]
    pulp_portfolio_returns = (df_log_returns*pd.Series(params['prices'])*x_pulp).sum(axis=1) - (pd.Series(params['taxes'])*pd.Series(params['prices'])*d_pulp).sum() - (pd.Series(params['fees'])*z_pulp).sum()

    test_pulp_portfolio_net_returns_constraint(params, df_log_returns, x_pulp, y_pulp, d_pulp, z_pulp, a_pulp, eta_pulp)

    return pulp_res_comp, pulp_portfolio_returns


def pulp_CVaR_portfolio_optmization_relaxed_model_solve_full(params,df_log_returns):
    pulp_relaxed_model, x_r, y_r, d_r, z_r, a_r, eta_r = pulp_CVaR_portfolio_optmization_relaxed_model(params)

    # solve
    pulp_relaxed_model.solve()

    # print results
    print(pulp_relaxed_model.solutionTime)
    print(pulp.LpStatus[pulp_relaxed_model.status])
    print(pulp.value(pulp_relaxed_model.objective))

    # x to series 
    x_pulp_r = pd.Series([x_r[j].value() for j in params['components']], index=params['components'])
    y_pulp_r = pd.Series([y_r[t].value() for t in params['scenarios']], index=params['scenarios'])
    d_pulp_r = pd.Series([d_r[j].value() for j in params['components']], index=params['components'])
    z_pulp_r = pd.Series([z_r[j].value() for j in params['components']], index=params['components'])
    a_pulp_r = pd.Series([a_r[j].value() for j in params['components']], index=params['components'])
    x_prices_pulp_r = pd.Series([x_r[j].value()*params['prices'][j] for j in params['components']], index=params['components'])
    eta_pulp_r = eta_r.value()

    # create dataframe
    pulp_res_comp_r = pd.DataFrame({'x':x_pulp_r, 'd':d_pulp_r, 'z':z_pulp_r, 'a':a_pulp_r, 'x_prices':x_prices_pulp_r})

    pulp_portfolio_returns_r = (df_log_returns*pd.Series(params['prices'])*x_pulp_r).sum(axis=1) - (pd.Series(params['taxes'])*pd.Series(params['prices'])*d_pulp_r).sum() - (pd.Series(params['fees'])*z_pulp_r).sum()

    test_pulp_portfolio_net_returns_constraint(params, df_log_returns, x_pulp_r, y_pulp_r, d_pulp_r, z_pulp_r, a_pulp_r, eta_pulp_r)

    return pulp_res_comp_r, pulp_portfolio_returns_r


def params_heuristic(df_log_returns,df_params,bucket,params):
    params_h = {
    'components': df_log_returns[bucket].columns.to_list(),
    'scenarios': df_log_returns[bucket].index.to_list(),
    'n_scenarios': [len(df_log_returns[bucket].index.to_list())],
    'scenarios_returns': df_log_returns[bucket].stack().to_dict(),
    'prices': df_params.q[bucket].to_dict(),
    'taxes': df_params.c[bucket].to_dict(),
    'fees': df_params.f[bucket].to_dict(),
    'mean_returns': df_log_returns[bucket].mean().to_dict(),
    'initial_lots': df_params.kappa[bucket].to_dict(),
    'upper_lot_boundaries': df_params.U[bucket].to_dict(),
    'lower_price_boundaries': pd.Series(params['lower_price_boundaries'])[bucket].to_dict(),
    'max_lot_rebalance': df_params.gamma[bucket].to_dict(),
    'expected_return_percentage': params['expected_return_percentage'],
    'additional_capital': params['additional_capital'],
    'beta': params['beta'],
    'total_components': params['total_components'],
    'remain_pct': params['remain_pct'],
    'min_bucket_components': [1]
    }
    return params_h


def pulp_CVaR_portfolio_optmization_model_heuristic(params):
    
    pulp_model_MILP = pulp.LpProblem("Portfolio CVaR optimization MILP", pulp.LpMaximize)

    x = {}
    for j in params['components']:
        x[j] = pulp.LpVariable(f'x_{j}', lowBound=0, cat='Integer')

    y = {}
    for t in params['scenarios']:
        y[t] = pulp.LpVariable(f'y_{t}', lowBound=0, cat='Continuous')

    z = {}
    for j in params['components']:
        z[j] = pulp.LpVariable(f'z_{j}', cat='Binary')

    a = {}
    for j in params['components']:
        a[j] = pulp.LpVariable(f'a_{j}', cat='Binary')

    d = {}
    for j in params['components']:
        d[j] = pulp.LpVariable(f'd_{j}', lowBound=0, cat='Continuous')

    eta = pulp.LpVariable(f'eta', cat='Continuous')

    pulp_model_MILP += eta - (1/params['beta'][0]) * (1/params['n_scenarios'][0]) * pulp.lpSum([y[t] for t in params['scenarios']])

    for t in params['scenarios']:
        pulp_model_MILP += eta + pulp.lpSum([ - params['scenarios_returns'][(t,j)]*params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) <= y[t]

    pulp_model_MILP += pulp.lpSum([params['mean_returns'][j]*params['prices'][j]*x[j] - params['taxes'][j]*params['prices'][j]*d[j] - params['fees'][j]*z[j] for j in params['components']]) >= params['expected_return_percentage'][0]*pulp.lpSum([params['prices'][j]*x[j] for j in params['components']])

    pulp_model_MILP += pulp.lpSum([params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) <= pulp.lpSum([params['prices'][j]*params['initial_lots'][j] for j in params['components']]) + params['additional_capital'][0]

    pulp_model_MILP += pulp.lpSum([params['prices'][j]*x[j] + params['taxes'][j]*params['prices'][j]*d[j] + params['fees'][j]*z[j] for j in params['components']]) >= (pulp.lpSum([params['prices'][j]*params['initial_lots'][j] for j in params['components']]) + params['additional_capital'][0]) * (1 - params['remain_pct'][0])

    for j in params['components']:
        pulp_model_MILP += x[j] * params['prices'][j] >= params['lower_price_boundaries'][j] * a[j]

    for j in params['components']:
        pulp_model_MILP += x[j] * params['prices'][j] <=  a[j] * ( pulp.lpSum([params['prices'][i]*params['initial_lots'][i] for i in params['components']]) + params['additional_capital'][0])

    for j in params['components']:
        pulp_model_MILP += x[j] <= params['upper_lot_boundaries'][j]

    for j in params['components']:
        pulp_model_MILP += d[j] >= (x[j] - params['initial_lots'][j])

    for j in params['components']:
        pulp_model_MILP += d[j] >= -(x[j] - params['initial_lots'][j])

    for j in params['components']:
        pulp_model_MILP += d[j] <= params['max_lot_rebalance'][j]*z[j]
        
    pulp_model_MILP += pulp.lpSum([a[j] for j in params['components']]) <= params['total_components'][0]

    pulp_model_MILP += pulp.lpSum([a[j]*params['bucket_components'][j] for j in params['components']]) >= params['min_bucket_components'][0]
    
    return pulp_model_MILP, x, y, d, z, a, eta


def pulp_CVaR_portfolio_optimization_heuristic_model_solve(params,df_log_returns,df_params,bucket_size):

    pulp_relaxed_model, _, _, _, z_r, a_r, _ = pulp_CVaR_portfolio_optmization_relaxed_model(params)
    
    start_time = time.time()

    pulp_relaxed_model.solve()

    # Data to series 
    z_pulp_r = pd.Series([z_r[j].value() for j in params['components']], index=params['components'])
    a_pulp_r = pd.Series([a_r[j].value() for j in params['components']], index=params['components'])

    # Identify the kernel and organize the remaining assets into ordered lists.
    h_filter = z_pulp_r.loc[(z_pulp_r > 0) | (a_pulp_r > 0)].index.to_list() # type: ignore
    unselected = list(set(params['components']) - set(h_filter))
    # sort unselected by sortino ratio, e Sortino, dado pelo retorno do portfolio dividido pelo desvio padr ̃ao dos retornos negativos no ativo
    sortino_ratios_sorted = (df_log_returns.mean()[unselected]/(df_log_returns[df_log_returns < 0]).std()[unselected]).sort_values(ascending=False).index.to_list()
    # sortino_ratios_sorted = unselected
    sorted_unselected_buckets = [sortino_ratios_sorted[i:i + params['total_components'][0]] for i in range(0, len(sortino_ratios_sorted), bucket_size)]

    # Concatenate heuristic_filter, unselected into buckets, as heuristic_filter as first bucket and unselected as the rest
    buckets = [h_filter] + sorted_unselected_buckets

    h_results = {}
    result = copy.deepcopy(h_filter)

    for i, b in zip(range(len(buckets)),buckets): # type: ignore
        if i == 0:
            bucket = copy.deepcopy(b)
        else:
            bucket = copy.deepcopy(result + b)

        params_h = params_heuristic(df_log_returns,df_params,bucket,params)

        pulp_model_h, x_h, y_h, d_h, z_h, a_h, eta_h = pulp_CVaR_portfolio_optmization_model(params_h)

        # solve
        pulp_model_h.solve()
            
        if pulp_model_h.status == 1:
            # Data to series
            x_pulp_h = pd.Series([x_h[j].value() for j in params_h['components']], index=params_h['components'])
            y_pulp_h = pd.Series([y_h[t].value() for t in params_h['scenarios']], index=params_h['scenarios'])
            d_pulp_h = pd.Series([d_h[j].value() for j in params_h['components']], index=params_h['components'])
            z_pulp_h = pd.Series([z_h[j].value() for j in params_h['components']], index=params_h['components'])
            a_pulp_h = pd.Series([a_h[j].value() for j in params_h['components']], index=params_h['components'])
            x_prices_pulp_h = pd.Series([x_h[j].value()*params_h['prices'][j] for j in params_h['components']], index=params_h['components'])
            eta_h = eta_h.value()
            
            result = z_pulp_h[(a_pulp_h > 0) | (z_pulp_h > 0)].index.to_list() # type: ignore
            
            # create dataframe
            pulp_res_comp_h = pd.DataFrame({'x':x_pulp_h[result], 'd':d_pulp_h[result], 'z':z_pulp_h[result], 'a':a_pulp_h[result], 'x_prices':x_prices_pulp_h[result]})

            pulp_portfolio_returns_h = (df_log_returns*pd.Series(params_h['prices'])*x_pulp_h).sum(axis=1) - (pd.Series(params_h['taxes'])*pd.Series(params_h['prices'])*d_pulp_h).sum() - (pd.Series(params_h['fees'])*z_pulp_h).sum()


            h_results[i] = {
                'opt_status': pulp.LpStatus[pulp_model_h.status],
                'opt_objective': pulp.value(pulp_model_h.objective),
                'solve_time': pulp_model_h.solutionTime,
                'bucket_result': result,
                'bucket': bucket,
                'pulp_res_comp': pulp_res_comp_h,
                'y': y_pulp_h,
                'eta': eta_h,
                'pulp_portfolio_returns': pulp_portfolio_returns_h            
                }
        else:
            h_results[i]= {
                'opt_status': pulp.LpStatus[pulp_model_h.status],
                'opt_objective': None,
                'bucket_result': None,
                'bucket': bucket,
                'pulp_res_comp': None,
                'y': None,
                'eta': None,
                'pulp_portfolio_returns': None
                }
            result = buckets[0]

    best_i = max(h_results, key=lambda x: h_results[x]['opt_objective'])

    end_time = time.time()

    solve_time = end_time - start_time
    obj_fun = h_results[best_i]['opt_objective']
    solve_status = h_results[best_i]['opt_status']

    return solve_status, solve_time, obj_fun


def pulp_CVaR_portfolio_optimization_heuristic_model_solve_cardinal(params,df_log_returns,df_params,bucket_size):

    pulp_relaxed_model, _, _, _, z_r, a_r, _ = pulp_CVaR_portfolio_optmization_relaxed_model(params)
    
    start_time = time.time()

    pulp_relaxed_model.solve()

    # Data to series 
    z_pulp_r = pd.Series([z_r[j].value() for j in params['components']], index=params['components'])
    a_pulp_r = pd.Series([a_r[j].value() for j in params['components']], index=params['components'])

    # Identify the kernel and organize the remaining assets into ordered lists.
    h_filter = z_pulp_r.loc[(z_pulp_r > 0) | (a_pulp_r > 0)].index.to_list() # type: ignore
    unselected = list(set(params['components']) - set(h_filter))
    # sort unselected by sortino ratio, e Sortino, dado pelo retorno do portfolio dividido pelo desvio padr ̃ao dos retornos negativos no ativo
    sortino_ratios_sorted = (df_log_returns.mean()[unselected]/(df_log_returns[df_log_returns < 0]).std()[unselected]).sort_values(ascending=False).index.to_list()
    # sortino_ratios_sorted = unselected
    sorted_unselected_buckets = [sortino_ratios_sorted[i:i + params['total_components'][0]] for i in range(0, len(sortino_ratios_sorted), bucket_size)]

    # Concatenate heuristic_filter, unselected into buckets, as heuristic_filter as first bucket and unselected as the rest
    buckets = [h_filter] + sorted_unselected_buckets

    h_results = {}
    result = copy.deepcopy(h_filter)

    for i, b in zip(range(len(buckets)),buckets): # type: ignore
        if i == 0:
            bucket = copy.deepcopy(b)
        else:
            bucket = copy.deepcopy(result + b)

        params_h = params_heuristic(df_log_returns,df_params,bucket,params)
        
        # vector of components where b is 1 on bucket components
        params_h['bucket_components'] = pd.Series(index = bucket, data=np.where(np.isin(params_h['components'], b), 1, 0)).to_dict()

        pulp_model_h, x_h, y_h, d_h, z_h, a_h, eta_h = pulp_CVaR_portfolio_optmization_model_heuristic(params_h)

        # solve
        pulp_model_h.solve()
        
        if i == 0 and pulp_model_h.status != 1:
            best_result = -np.inf 
        else:
            best_result = pulp.value(pulp_model_h.objective)
            
            
        if pulp_model_h.status == 1:
            # Data to series
            x_pulp_h = pd.Series([x_h[j].value() for j in params_h['components']], index=params_h['components'])
            y_pulp_h = pd.Series([y_h[t].value() for t in params_h['scenarios']], index=params_h['scenarios'])
            d_pulp_h = pd.Series([d_h[j].value() for j in params_h['components']], index=params_h['components'])
            z_pulp_h = pd.Series([z_h[j].value() for j in params_h['components']], index=params_h['components'])
            a_pulp_h = pd.Series([a_h[j].value() for j in params_h['components']], index=params_h['components'])
            x_prices_pulp_h = pd.Series([x_h[j].value()*params_h['prices'][j] for j in params_h['components']], index=params_h['components'])
            eta_h = eta_h.value()
            
            result = z_pulp_h[(a_pulp_h > 0) | (z_pulp_h > 0)].index.to_list() # type: ignore
            
            # create dataframe
            pulp_res_comp_h = pd.DataFrame({'x':x_pulp_h[result], 'd':d_pulp_h[result], 'z':z_pulp_h[result], 'a':a_pulp_h[result], 'x_prices':x_prices_pulp_h[result]})

            pulp_portfolio_returns_h = (df_log_returns*pd.Series(params_h['prices'])*x_pulp_h).sum(axis=1) - (pd.Series(params_h['taxes'])*pd.Series(params_h['prices'])*d_pulp_h).sum() - (pd.Series(params_h['fees'])*z_pulp_h).sum()


            h_results[i] = {
                'opt_status': pulp.LpStatus[pulp_model_h.status],
                'opt_objective': pulp.value(pulp_model_h.objective),
                'solve_time': pulp_model_h.solutionTime,
                'bucket_result': result,
                'bucket': bucket,
                'pulp_res_comp': pulp_res_comp_h,
                'y': y_pulp_h,
                'eta': eta_h,
                'pulp_portfolio_returns': pulp_portfolio_returns_h            
                }
        else:
            h_results[i]= {
                'opt_status': pulp.LpStatus[pulp_model_h.status],
                'opt_objective': None,
                'bucket_result': None,
                'bucket': bucket,
                'pulp_res_comp': None,
                'y': None,
                'eta': None,
                'pulp_portfolio_returns': None
                }
            result = buckets[0]
            
        if pulp_model_h.status == 1 and best_result < pulp.value(pulp_model_h.objective):
            best_result = pulp.value(pulp_model_h.objective)
        else:
            break
            
        

    best_i = max(h_results, key=lambda x: h_results[x]['opt_objective'])

    end_time = time.time()

    solve_time = end_time - start_time
    obj_fun = h_results[best_i]['opt_objective']
    solve_status = h_results[best_i]['opt_status']

    return solve_status, solve_time, obj_fun


def main():
    # import data
    df_log_returns_init = pd.read_csv('df_log_retornos.csv', index_col=0)
    df_params_init = pd.read_csv('df_params.csv', index_col=0)

    # reindex data
    df_log_returns_init.reset_index(inplace=True, drop=True)
    # define parameters
    n_scenarios = len(df_log_returns_init) - 120 
    C_0 = 9000
    mu_0 = 0.001
    B = 1000
    pho = 0.001
    l = 20 # number of assets to add to the kernel at each iteration
    coins = ['BTCBRL', 'BNBBRL','ETHBRL']
    beta = [.01, .05]
    N = [5, 10, 20]
    n_assets = [300,600,900]
    # define empty dataframes to store results with columns n_assets, beta, N, LP_FO, LP_T, He_FO, He_T, Gap
    df_results = pd.DataFrame(columns=['n_assets', 'beta', 'N', 'LP_FO', 'LP_T', 'He_FO', 'He_T', 'Hec_FO', 'Hec_T', 'Gap', 'Gapc'])
                    
    for n_asset in n_assets:
        for b in beta:
            for n in N:
                params, df_log_returns, df_params = init_params_config(df_log_returns_init,df_params_init,n_asset,n_scenarios,coins,C_0,mu_0,B,b,n,pho)
                solve_status_lp, solve_time_lp, obj_fun_lp = pulp_CVaR_portfolio_optmization_model_solve(params)
                solve_status_h, solve_time_h, obj_fun_h = pulp_CVaR_portfolio_optimization_heuristic_model_solve(params,df_log_returns,df_params,l)
                solve_status_hc, solve_time_hc, obj_fun_hc = pulp_CVaR_portfolio_optimization_heuristic_model_solve_cardinal(params,df_log_returns,df_params,l)
                gap = (obj_fun_h - obj_fun_lp)/obj_fun_lp * 100
                gapc = (obj_fun_hc - obj_fun_lp)/obj_fun_lp * 100
                df_results = df_results.append({
                    'n_assets': n_asset, 
                    'beta': b, 
                    'N': n, 
                    'LP_FO': obj_fun_lp, 
                    'LP_T': solve_time_lp, 
                    'He_FO': obj_fun_h, 
                    'He_T': solve_time_h, 
                    'Hec_FO': obj_fun_hc,
                    'Hec_T': solve_time_hc,
                    'Gap': gap,
                    'Gapc': gapc
                    }, ignore_index=True) # type: ignore

                df_results.to_csv('df_results.csv', float_format='%.2f')


# if __name__ == '__main__':
    # main()