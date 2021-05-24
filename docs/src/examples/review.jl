# Copyright 2021, christian fuellner.
# Example for SDDP Review Paper

using SDDP, GLPK, Test

function example()
    model = SDDP.PolicyGraph(
        SDDP.LinearGraph(3),
        bellman_function = SDDP.BellmanFunction(lower_bound = -10),
        optimizer = GLPK.Optimizer,
    ) do sp, t
        @variable(sp, 6 >= x[1:2] >= 0, SDDP.State, initial_value = 0.0)
        @stageobjective(sp, x[1].out + x[2].out)
        @variable(sp, ξ)

        if t == 1
            @constraint(sp, x[2].out == 0)
            JuMP.fix(ξ, 0)

        elseif t == 2
            @constraint(sp, x[2].out == 0)
            @constraint(sp, x[1].out >= ξ - x[1].in)
            SDDP.parameterize(sp, [4,5,6]) do ω
                JuMP.fix(ξ, ω)
            end

        elseif t == 3
            @constraint(sp, x[1].out - x[2].out == ξ - x[1].in)
            SDDP.parameterize(sp, [1,2,4]) do ω
                JuMP.fix(ξ, ω)
            end

        end

    end

    det = SDDP.deterministic_equivalent(model, GLPK.Optimizer)
    JuMP.optimize!(det)
    #@test JuMP.objective_value(det) == -2

    SDDP.train(model, iteration_limit = 10)
    #@test SDDP.calculate_bound(model) == -2
end

example()
