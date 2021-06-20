using SDDP, GLPK, Test, Infiltrator, GAMS, Gurobi

struct Generator
    comm_ini::Int
    gen_ini::Float64
    pmax::Float64
    pmin::Float64
    fuel_cost::Float64
    om_cost::Float64
    su_cost::Float64
    sd_cost::Float64
    ramp_up::Float64
    ramp_dw::Float64
end

function unit_commitment_model(integrality_handler)

    generators = [
        Generator(0, 0.0, 1.18, 0.32, 48.9, 0.0, 182.35, 18.0, 0.42, 0.33),
        Generator(1, 1.06, 1.19, 0.37, 52.1, 0.0, 177.68, 17.0, 0.31, 0.36),
        Generator(0, 0.0, 1.05, 0.48, 42.8, 0.0, 171.69, 17.0, 0.21, 0.22),
        #Generator(0, 0.0, 1.13, 0.48, 54.0, 0.0, 171.60, 17.0, 0.28, 0.27),
        #Generator(0, 0.0, 1.02, 0.47, 49.4, 0.0, 168.04, 17.0, 0.22, 0.275),
        #Generator(1, 0.72, 1.9, 0.5, 64.1, 0.0, 289.59, 28.0, 0.52, 0.62),
        #Generator(0, 0.0, 2.08, 0.62, 60.3, 0.0, 286.89, 28.0, 0.67, 0.5),
        #Generator(1, 0.55, 2.11, 0.55, 66.1, 0.0, 329.89, 33.0, 0.64, 0.69),
        #Generator(1, 2.2, 2.82, 0.85, 61.6, 0.0, 486.81, 49.0, 0.9, 0.79),
        #Generator(0, 0.0, 3.23, 0.84, 54.9, 0.0, 503.34, 50.0, 1.01, 1.00),
    ]

    num_of_generators = size(generators,1)
    num_of_stages = 3

    demand_penalty = 5e2
    demand = [3.06 2.91 2.71 2.7 2.73 2.91 3.38 4.01 4.6 4.78 4.81 4.84 4.89 4.44 4.57 4.6 4.58 4.47 4.32 4.36 4.5 4.27 3.93 3.61 3.43 3.02 2.9 2.54 2.73 3.01 3.45 3.89 4.5 4.76 4.9 5.04]

    model = SDDP.LinearPolicyGraph(
        stages = num_of_stages,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
        integrality_handler = integrality_handler,
    ) do subproblem, stage

        # state variables
        @variable(
            subproblem,
            0.0 <= commit[i = 1:num_of_generators] <= 1.0,
            SDDP.State,
            Bin,
            initial_value = generators[i].comm_ini
        )

        @variable(
            subproblem,
            0.0 <= gen[i = 1:num_of_generators] <= generators[i].pmax,
            SDDP.State,
            initial_value = generators[i].gen_ini
        )

        # start-up variables
        @variable(subproblem, up[i=1:num_of_generators], Bin)
        @variable(subproblem, down[i=1:num_of_generators], Bin)

        # demand slack
        @variable(subproblem, demand_slack >= 0.0)
        @variable(subproblem, neg_demand_slack >= 0.0)

        # cost variables
        @variable(subproblem, startup_costs[i=1:num_of_generators] >= 0.0)
        @variable(subproblem, shutdown_costs[i=1:num_of_generators] >= 0.0)
        @variable(subproblem, fuel_costs[i=1:num_of_generators] >= 0.0)
        @variable(subproblem, om_costs[i=1:num_of_generators] >= 0.0)

        # generation bounds
        @constraint(subproblem, genmin[i=1:num_of_generators], gen[i].out >= commit[i].out * generators[i].pmin)
        @constraint(subproblem, genmax[i=1:num_of_generators], gen[i].out <= commit[i].out * generators[i].pmax)

        # ramping
        # we do not need a case distinction as we defined initial_values
        @constraint(subproblem, rampup[i=1:num_of_generators], gen[i].out - gen[i].in <= generators[i].ramp_up * commit[i].in + generators[i].pmin * (1-commit[i].in))
        @constraint(subproblem, rampdown[i=1:num_of_generators], gen[i].in - gen[i].out <= generators[i].ramp_dw * commit[i].out + generators[i].pmin * (1-commit[i].out))

        # start-up and shut-down
        # we do not need a case distinction as we defined initial_values
        @constraint(subproblem, startup[i=1:num_of_generators], up[i] >= commit[i].out - commit[i].in)
        @constraint(subproblem, shutdown[i=1:num_of_generators], down[i] >= commit[i].in - commit[i].out)

        # load balance
        @constraint(subproblem, load, sum(gen[i].out for i in 1:num_of_generators) + demand_slack - neg_demand_slack == demand[stage] )

        # costs
        @constraint(subproblem, startupcost[i=1:num_of_generators], num_of_stages/24 * generators[i].su_cost * up[i] == startup_costs[i])
        @constraint(subproblem, shutdowncost[i=1:num_of_generators], generators[i].sd_cost * down[i] == shutdown_costs[i])
        @constraint(subproblem, fuelcost[i=1:num_of_generators], generators[i].fuel_cost * gen[i].out == fuel_costs[i])
        @constraint(subproblem, omcost[i=1:num_of_generators], generators[i].om_cost * gen[i].out == om_costs[i])

        # define stage objective
        su_costs = subproblem[:startup_costs]
        sd_costs = subproblem[:shutdown_costs]
        f_costs = subproblem[:fuel_costs]
        om_costs = subproblem[:om_costs]
        demand_slack = subproblem[:demand_slack]
        neg_demand_slack = subproblem[:neg_demand_slack]
        @stageobjective(subproblem,
                        sum(su_costs[i] + sd_costs[i] + f_costs[i] + om_costs[i] for i in 1:num_of_generators)
                        + demand_slack * demand_penalty + neg_demand_slack * demand_penalty)

    end

    model.ext[:solver] = solver

    SDDP.train(
        model,
        iteration_limit = 5,
        time_limit = 1000,
        log_frequency = 1,
        print_level = 2)
    #@test SDDP.calculate_bound(model) ≈ 62_500.0
    return
end

for integrality_handler in [SDDP.SDDiP(iteration_limit = 1000)]
    unit_commitment_model(integrality_handler)
end