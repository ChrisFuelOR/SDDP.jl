#  Copyright 2017-21, Oscar Dowson.                                     #src
#  This Source Code Form is subject to the terms of the Mozilla Public  #src
#  License, v. 2.0. If a copy of the MPL was not distributed with this  #src
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.             #src

# # Air conditioning

# Taken from [Anthony Papavasiliou's notes on SDDP](https://perso.uclouvain.be/anthony.papavasiliou/public_html/SDDP.pdf)

# Consider the following problem
# * Produce air conditioners for 3 months
# * 200 units/month at 100 \\\$/unit
# * Overtime costs 300 \\\$/unit
# * Known demand of 100 units for period 1
# * Equally likely demand, 100 or 300 units, for periods 2, 3
# * Storage cost is 50 \\\$/unit
# * All demand must be met

# The known optimal solution is \\\$62,500

using SDDP, GLPK, Test, Infiltrator

function air_conditioning_model(integrality_handler, iteration_limit, solver)

    model = SDDP.LinearPolicyGraph(
        stages = 3,
        lower_bound = 0.0,
        optimizer = GLPK.Optimizer,
        integrality_handler = integrality_handler,
    ) do sp, stage
        @variable(sp, 0 <= stored_production <= 100, Int, SDDP.State, initial_value = 0, epislon = binaryPrecision)
        @variable(sp, 0 <= production <= 200, Int)
        @variable(sp, overtime >= 0, Int)
        @variable(sp, demand)
        DEMAND = [[100.0], [100.0, 300.0], [100.0, 300.0]]
        SDDP.parameterize(ω -> JuMP.fix(demand, ω), sp, DEMAND[stage])
        @constraint(
            sp,
            stored_production.out == stored_production.in + production + overtime - demand
        )
        @stageobjective(sp, 100 * production + 300 * overtime + 50 * stored_production.out)
    end

    model.ext[:solver] = solver

    SDDP.train(model, iteration_limit = iteration_limit, log_frequency = 1)
    @test SDDP.calculate_bound(model) ≈ 62_500.0
    return
end

#for integrality_handler in [SDDP.SDDiP(), SDDP.ContinuousRelaxation()]
#    air_conditioning_model(integrality_handler)
#end

# Parameter configuration
################################################################################
iteration_limit = 100
iteration_limit_lag = 100
lag_atol = 1e-8
lag_rtol = 1e-8
sol_method = :kelley
status_regime = :rigorous
bound_regime = :value
init_regime = :zeros
cut_type = :L
solver = GLPK.Optimizer
lag_solver = GLPK.Optimizer
bundle_alpha = 0.5
bundle_factor = 1.0
level_factor = 0.2
binaryPrecision = 0.1

bundleParams = SDDP.BundleParams(bundle_alpha, bundle_factor, level_factor)
algoParams = SDDP.AlgoParams(sol_method, status_regime, bound_regime, init_regime, cut_type, lag_solver, bundleParams, binaryPrecision)
################################################################################

for integrality_handler in [SDDP.SDDiP_con(algoParams=algoParams, rtol=lag_rtol, atol=lag_atol, iteration_limit=iteration_limit_lag)]
    air_conditioning_model(integrality_handler, iteration_limit, solver)
end
