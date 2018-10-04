#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

#==
    An implementation of the QuickStart example from FAST
    https://github.com/leopoldcambier/FAST/tree/daea3d80a5ebb2c52f78670e34db56d53ca2e778/demo

==#

using Kokako, GLPK, Test

function fast_quickstart()
    model = Kokako.PolicyGraph(Kokako.LinearGraph(2),
                bellman_function = Kokako.AverageCut(lower_bound=-5),
                optimizer = with_optimizer(GLPK.Optimizer)
                        ) do sp, t
        @variable(sp, x >= 0, Kokako.State, root_value = 0.0)
        if t == 1
            @stageobjective(sp, x.out)
        else
            @variable(sp, s >= 0)
            @constraint(sp, s <= x.in)
            Kokako.parameterize(sp, [2, 3]) do ω
                JuMP.set_upper_bound(s, ω)
            end
            @stageobjective(sp, -2s)
        end
    end
    Kokako.train(model, iteration_limit = 3, print_level = 0)
    @test Kokako.calculate_bound(model) == -2
end

fast_quickstart()
