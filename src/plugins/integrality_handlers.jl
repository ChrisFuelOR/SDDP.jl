#  Copyright 2017-21, Oscar Dowson, Lea Kapelevich.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# ========================= General methods ================================== #

"""
    enforce_integrality(
        binaries::Vector{Tuple{JuMP.VariableRef, Float64, Float64}},
        integers::Vector{VariableRef})

Set all variables in `binaries` to `SingleVariable-in-ZeroOne()`, and all
variables in `integers` to `SingleVariable-in-Integer()`.

See also [`relax_integrality`](@ref).
"""
function enforce_integrality(
    binaries::Vector{Tuple{JuMP.VariableRef,Float64,Float64}},
    integers::Vector{VariableRef},
)
    JuMP.set_integer.(integers)
    for (x, x_lb, x_ub) in binaries
        if isnan(x_lb)
            JuMP.delete_lower_bound(x)
        else
            JuMP.set_lower_bound(x, x_lb)
        end
        if isnan(x_ub)
            JuMP.delete_upper_bound(x)
        else
            JuMP.set_upper_bound(x, x_ub)
        end
        JuMP.set_binary(x)
    end
    return
end

get_integrality_handler(subproblem::JuMP.Model) = get_node(subproblem).integrality_handler

# ========================= Continuous relaxation ============================ #

"""
    ContinuousRelaxation()

The continuous relaxation integrality handler. Duals are obtained in the
backward pass by solving a continuous relaxation of each subproblem.
Integrality constraints are retained in policy simulation.
"""
struct ContinuousRelaxation <: AbstractIntegralityHandler end

function setup_state(
    subproblem::JuMP.Model,
    state::State,
    state_info::StateInfo,
    name::String,
    ::ContinuousRelaxation,
)
    node = get_node(subproblem)
    sym_name = Symbol(name)
    @assert !haskey(node.states, sym_name)  # JuMP prevents duplicate names.
    node.states[sym_name] = state
    graph = get_policy_graph(subproblem)
    graph.initial_root_state[sym_name] = state_info.initial_value
    return
end

# Requires node.subproblem to have been solved with DualStatus == FeasiblePoint
function get_dual_variables(node::Node, ::ContinuousRelaxation)
    # Note: due to JuMP's dual convention, we need to flip the sign for
    # maximization problems.
    dual_values = Dict{Symbol,Float64}()
    if JuMP.dual_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
        write_subproblem_to_file(node, "subproblem.mof.json", throw_error = true)
    end
    dual_sign = JuMP.objective_sense(node.subproblem) == MOI.MIN_SENSE ? 1.0 : -1.0
    for (name, state) in node.states
        ref = JuMP.FixRef(state.in)
        dual_values[name] = dual_sign * JuMP.dual(ref)
    end
    return dual_values
end

function relax_integrality(model::PolicyGraph, ::ContinuousRelaxation)
    binaries = Tuple{JuMP.VariableRef,Float64,Float64}[]
    integers = JuMP.VariableRef[]
    for (key, node) in model.nodes
        bins, ints = _relax_integrality(node.subproblem)
        append!(binaries, bins)
        append!(integers, ints)
    end
    return binaries, integers
end

# Relax all binary and integer constraints in `model`. Returns two vectors:
# the first containing a list of binary variables and previous bounds,
# and the second containing a list of integer variables.
function _relax_integrality(m::JuMP.Model)
    # Note: if integrality restriction is added via @constraint then this loop doesn't catch it.
    binaries = Tuple{JuMP.VariableRef,Float64,Float64}[]
    integers = JuMP.VariableRef[]
    # Run through all variables on model and unset integrality
    for x in JuMP.all_variables(m)
        if JuMP.is_binary(x)
            x_lb, x_ub = NaN, NaN
            JuMP.unset_binary(x)
            # Store upper and lower bounds
            if JuMP.has_lower_bound(x)
                x_lb = JuMP.lower_bound(x)
                JuMP.set_lower_bound(x, max(x_lb, 0.0))
            else
                JuMP.set_lower_bound(x, 0.0)
            end
            if JuMP.has_upper_bound(x)
                x_ub = JuMP.upper_bound(x)
                JuMP.set_upper_bound(x, min(x_ub, 1.0))
            else
                JuMP.set_upper_bound(x, 1.0)
            end
            push!(binaries, (x, x_lb, x_ub))
        elseif JuMP.is_integer(x)
            JuMP.unset_integer(x)
            push!(integers, x)
        end
    end
    return binaries, integers
end

# =========================== SDDiP ========================================== #

"""
    SDDiP(; iteration_limit::Int = 100, atol::Float64, rtol::Float64)

The SDDiP integrality handler introduced by Zou, J., Ahmed, S. & Sun, X.A.
Math. Program. (2019) 175: 461. Stochastic dual dynamic integer programming.
https://doi.org/10.1007/s10107-018-1249-5.

Calculates duals by solving the Lagrangian dual for each subproblem. Kelley's
method is used to compute Lagrange multipliers. `iteration_limit` controls the
maximum number of iterations, and `atol` and `rtol` are the absolute and
relative tolerances used in the termination criteria.

All state variables are assumed to take nonnegative values only.
"""

mutable struct SDDiP <: AbstractIntegralityHandler
    iteration_limit::Int
    optimizer::Any
    subgradients::Vector{Float64}
    old_rhs::Vector{Float64}
    best_mult::Vector{Float64}
    slacks::Vector{GenericAffExpr{Float64,VariableRef}}
    atol::Float64
    rtol::Float64
    algoParams::AlgoParams

    function SDDiP(; iteration_limit::Int = 100, atol::Float64 = 1e-8, rtol::Float64 = 1e-8, algoParams::AlgoParams)
        integrality_handler = new()
        integrality_handler.iteration_limit = iteration_limit
        integrality_handler.atol = atol
        integrality_handler.rtol = rtol
        integrality_handler.algoParams = algoParams
        return integrality_handler
    end
end

function update_integrality_handler!(
    integrality_handler::SDDiP,
    optimizer::Any,
    num_states::Int,
)
    integrality_handler.optimizer = optimizer
    integrality_handler.subgradients = Vector{Float64}(undef, num_states)
    integrality_handler.old_rhs = similar(integrality_handler.subgradients)
    integrality_handler.best_mult = similar(integrality_handler.subgradients)
    integrality_handler.slacks =
        Vector{GenericAffExpr{Float64,VariableRef}}(undef, num_states)
    return integrality_handler
end

function setup_state(
    subproblem::JuMP.Model,
    state::State,
    state_info::StateInfo,
    name::String,
    ::SDDiP,
)
    if state_info.out.binary
        # Only in this case we treat `state` as a real state variable
        setup_state(subproblem, state, state_info, name, ContinuousRelaxation())
    else
        if !isfinite(state_info.out.upper_bound)
            error("When using SDDiP, state variables require an upper bound.")
        end

        if state_info.out.integer
            # Initial value must be integral
            initial_value = binexpand(
                Int(state_info.initial_value),
                floor(Int, state_info.out.upper_bound),
            )
            num_vars = length(initial_value)

            binary_vars = JuMP.@variable(
                subproblem,
                [i in 1:num_vars],
                base_name = "_bin_" * name,
                SDDP.State,
                Bin,
                initial_value = initial_value[i]
            )

            JuMP.@constraint(
                subproblem,
                state.in == bincontract([binary_vars[i].in for i = 1:num_vars])
            )
            JuMP.@constraint(
                subproblem,
                state.out == bincontract([binary_vars[i].out for i = 1:num_vars])
            )
        else
            epsilon =
                (
                    haskey(state_info.kwargs, :epsilon) ? state_info.kwargs[:epsilon] : 0.1
                )::Float64
            initial_value = binexpand(
                float(state_info.initial_value),
                float(state_info.out.upper_bound),
                epsilon,
            )
            num_vars = length(initial_value)

            binary_vars = JuMP.@variable(
                subproblem,
                [i in 1:num_vars],
                base_name = "_bin_" * name,
                SDDP.State,
                Bin,
                initial_value = initial_value[i]
            )

            JuMP.@constraint(
                subproblem,
                state.in == bincontract([binary_vars[i].in for i = 1:num_vars], epsilon)
            )
            JuMP.@constraint(
                subproblem,
                state.out == bincontract([binary_vars[i].out for i = 1:num_vars], epsilon)
            )
        end
    end
    return
end

function get_dual_variables(node::Node, integrality_handler::SDDiP)
    dual_values = Dict{Symbol,Float64}()
    dual_vars = zeros(length(node.states))
    solver_obj = JuMP.objective_value(node.subproblem)
    try
        kelley_obj = _kelley(node, dual_vars, integrality_handler)::Float64
        @assert isapprox(solver_obj, kelley_obj, atol = 1e-8, rtol = 1e-8)
    catch e
        write_subproblem_to_file(node, "subproblem.mof.json", throw_error = false)
        rethrow(e)
    end
    for (i, name) in enumerate(keys(node.states))
        dual_values[name] = -dual_vars[i]
    end
    return dual_values
end

relax_integrality(::PolicyGraph, ::SDDiP) =
    Tuple{JuMP.VariableRef,Float64,Float64}[], JuMP.VariableRef[]

function _solve_primal!(
    subgradients::Vector{Float64},
    node::Node,
    dual_vars::Vector{Float64},
    slacks,
)
    model = node.subproblem
    old_obj = JuMP.objective_function(model)
    # Set the Lagrangian the objective in the primal model
    fact = (JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? 1 : -1)
    new_obj = old_obj + fact * LinearAlgebra.dot(dual_vars, slacks)
    JuMP.set_objective_function(model, new_obj)
    JuMP.optimize!(model)
    lagrangian_obj = JuMP.objective_value(model)

    # Reset old objective, update subgradients using slack values
    JuMP.set_objective_function(model, old_obj)
    subgradients .= fact .* JuMP.value.(slacks)
    return lagrangian_obj
end

function _kelley(node::Node, dual_vars::Vector{Float64}, integrality_handler::SDDiP)
    atol = integrality_handler.atol
    rtol = integrality_handler.rtol
    model = node.subproblem
    # Assume the model has been solved. Solving the MIP is usually very quick
    # relative to solving for the Lagrangian duals, so we cheat and use the
    # solved model's objective as our bound while searching for the optimal duals
    @assert JuMP.termination_status(model) == MOI.OPTIMAL
    obj = JuMP.objective_value(model)

    for (i, (name, state)) in enumerate(node.states)
        integrality_handler.old_rhs[i] = JuMP.fix_value(state.in)
        integrality_handler.slacks[i] = state.in - integrality_handler.old_rhs[i]
        JuMP.unfix(state.in)
        JuMP.set_lower_bound(state.in, 0)
        JuMP.set_upper_bound(state.in, 1)
    end

    # Subgradient at current solution
    subgradients = integrality_handler.subgradients
    # Best multipliers found so far
    best_mult = integrality_handler.best_mult
    # Dual problem has the opposite sense to the primal
    dualsense = (
        JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? JuMP.MOI.MAX_SENSE :
            JuMP.MOI.MIN_SENSE
    )

    # Approximation of Lagrangian dual as a function of the multipliers
    approx_model = JuMP.Model(integrality_handler.optimizer)

    # Objective estimate and Lagrangian duals
    @variables approx_model begin
        θ
        x[1:length(dual_vars)]
    end
    JuMP.@objective(approx_model, dualsense, θ)

    if dualsense == MOI.MIN_SENSE
        JuMP.set_lower_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (Inf, Inf, -Inf)
    else
        JuMP.set_upper_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (-Inf, -Inf, Inf)
    end

    iter = 0
    while iter < integrality_handler.iteration_limit
        iter += 1
        # Evaluate the real function and a subgradient
        f_actual = _solve_primal!(subgradients, node, dual_vars, integrality_handler.slacks)

        # Update the model and update best function value so far
        if dualsense == MOI.MIN_SENSE
            JuMP.@constraint(
                approx_model,
                θ >= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_actual <= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        else
            JuMP.@constraint(
                approx_model,
                θ <= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_actual >= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        end
        # Get a bound from the approximate model
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL
        f_approx = JuMP.objective_value(approx_model)

        # More reliable than checking whether subgradient is zero
        if isapprox(best_actual, f_approx, atol = atol, rtol = rtol)
            dual_vars .= best_mult
            if dualsense == JuMP.MOI.MIN_SENSE
                dual_vars .*= -1
            end
            for (i, (name, state)) in enumerate(node.states)
                JuMP.fix(state.in, integrality_handler.old_rhs[i], force = true)
            end

            return best_actual
        end
        # Next iterate
        dual_vars .= value.(x)
    end
    error("Could not solve for Lagrangian duals. Iteration limit exceeded.")
end

# ==================== General stuff for extension =========================== #

"""
This struct has been added by chrisfuelor to store parameters required to
configure the level bundle method if used to solve the Lagrangian duals
"""
mutable struct BundleParams
    bundle_alpha::Float64
    bundle_factor::Float64
    level_factor::Float64
end

"""
This struct has been added by chrisfuelor to store parameters required to
configure the solution of the Lagrangian dual problems in SDDiP.

sol_method:
--------------------------------------------------------------------------------
- defines which solution method is used for the Lagrangian duals
- can be either :kelley or :bundle_level
- in the first case, Kelley's cutting-plane method is used, in the second case
a level bundle method

status_regime:
--------------------------------------------------------------------------------
- defines some regime for the desired solution state of the Lagrangian duals
- can be either: :rigorous or :lax
- :rigorous -> cuts will be only used if the Lagrangian duals are solved as
attempted, otherwise the algorithm terminates with an error
- :lax -> even if Lagrangian duals are not solved as attempted (e.g. due to
iteration limit reached, subgradients zero but LB!=UB, bounds stalling), the
current dual values are used to define a valid cut for the value function,
hoping that this cut will suffice to cut away the current incumbent

bound_regime:
--------------------------------------------------------------------------------
- defines if bounds are used for the Lagrangian dual
- can be :both, :value, :duals, :none
- :none -> no specific bounds are used for the Lagrangian dual (some weak
default bound is used to prevent unbounded subproblems)
- :value -> as in original SDDiP, the optimal value of the Lagrangian dual
is bounded by the primal solution
- :duals -> the feasible set of the dual multipliers is bounded
- :both -> combination of :value and :duals

init_regime:
--------------------------------------------------------------------------------
- defines how the dual variables are initialized
- can be :zeros, :LP, :cplex_fixed
- :zeros -> dual multipliers are initialized as zero vector
- :LP -> duals of LP relaxation are used for initialization
- :cplex_fixed -> CPLEX (in constrast to Gurobi) also provides duals if an MILP
is solved (they are determined by using an LP with fixed integer variables);
use these

cut_type:
--------------------------------------------------------------------------------
- defines which kind of cut is constructed
- can be :L, :SB or :B
- :L -> Lagrangian cuts are determined by solving Lagrangian dual
(note that for state_regime :original, the duality gap may not be closed)
- :SB -> Strengthened Benders cuts are determined
- :B -> original Benders cuts are determined by solving the LP relaxation

bundle_parameters:
--------------------------------------------------------------------------------
- parameters specific to the level bundle method

solver:
--------------------------------------------------------------------------------
- defines which solver should be used to solve the subproblems
- note that in integrality_handler already an optimizer is defined;
however, if this one is GAMS, then a specific solver has to be defined
"""

mutable struct AlgoParams
    sol_method::Symbol
    status_regime::Symbol
    bound_regime::Symbol
    init_regime::Symbol
    cut_type::Symbol
    solver::Any
    bundle_parameters::Union{Nothing, BundleParams}
    binaryPrecision::Union{Nothing, Float64}
end

# =========================== SDDiP_bin ====================================== #

"""
    SDDiP_bin(; iteration_limit::Int = 100, atol::Float64, rtol::Float64)

The SDDiP integrality handler introduced by Zou, J., Ahmed, S. & Sun, X.A.
Math. Program. (2019) 175: 461. Stochastic dual dynamic integer programming.
https://doi.org/10.1007/s10107-018-1249-5.

Calculates duals by solving the Lagrangian dual for each subproblem. Kelley's
method is used to compute Lagrange multipliers. `iteration_limit` controls the
maximum number of iterations, and `atol` and `rtol` are the absolute and
relative tolerances used in the termination criteria.

Additionally, more specific algoParams are introduced with which the settings
for the Lagrangian duals can be configured.

In SDDiP_bin, state variables are approximated by some binary expansion.

All state variables are assumed to take nonnegative values only.
"""

mutable struct SDDiP_bin <: AbstractIntegralityHandler
    iteration_limit::Int
    optimizer::Any
    subgradients::Vector{Float64}
    old_rhs::Vector{Float64}
    best_mult::Vector{Float64}
    slacks::Vector{GenericAffExpr{Float64,VariableRef}}
    atol::Float64
    rtol::Float64
    algoParams::Union{Nothing,AlgoParams}

    function SDDiP_bin(; iteration_limit::Int = 100, atol::Float64 = 1e-8, rtol::Float64 = 1e-8, algoParams::AlgoParams = nothing)
        integrality_handler = new()
        integrality_handler.iteration_limit = iteration_limit
        integrality_handler.atol = atol
        integrality_handler.rtol = rtol
        integrality_handler.algoParams = algoParams
        return integrality_handler
    end
end

function update_integrality_handler!(
    integrality_handler::SDDiP_bin,
    optimizer::Any,
    num_states::Int,
)
    integrality_handler.optimizer = optimizer
    integrality_handler.subgradients = Vector{Float64}(undef, num_states)
    integrality_handler.old_rhs = similar(integrality_handler.subgradients)
    integrality_handler.best_mult = similar(integrality_handler.subgradients)
    integrality_handler.slacks =
        Vector{GenericAffExpr{Float64,VariableRef}}(undef, num_states)
    return integrality_handler
end

function setup_state(
    subproblem::JuMP.Model,
    state::State,
    state_info::StateInfo,
    name::String,
    ::SDDiP_bin,
)
    if state_info.out.binary
        # Only in this case we treat `state` as a real state variable
        setup_state(subproblem, state, state_info, name, ContinuousRelaxation())
    else
        if !isfinite(state_info.out.upper_bound)
            error("When using SDDiP, state variables require an upper bound.")
        end

        if state_info.out.integer
            # Initial value must be integral
            initial_value = binexpand(
                Int(state_info.initial_value),
                floor(Int, state_info.out.upper_bound),
            )
            num_vars = length(initial_value)

            binary_vars = JuMP.@variable(
                subproblem,
                [i in 1:num_vars],
                base_name = "_bin_" * name,
                SDDP.State,
                Bin,
                initial_value = initial_value[i]
            )

            JuMP.@constraint(
                subproblem,
                state.in == bincontract([binary_vars[i].in for i = 1:num_vars])
            )
            JuMP.@constraint(
                subproblem,
                state.out == bincontract([binary_vars[i].out for i = 1:num_vars])
            )
        else
            epsilon =
                (
                    haskey(state_info.kwargs, :epsilon) ? state_info.kwargs[:epsilon] : 0.1
                )::Float64
            initial_value = binexpand(
                float(state_info.initial_value),
                float(state_info.out.upper_bound),
                epsilon,
            )
            num_vars = length(initial_value)

            binary_vars = JuMP.@variable(
                subproblem,
                [i in 1:num_vars],
                base_name = "_bin_" * name,
                SDDP.State,
                Bin,
                initial_value = initial_value[i]
            )

            JuMP.@constraint(
                subproblem,
                state.in == bincontract([binary_vars[i].in for i = 1:num_vars], epsilon)
            )
            JuMP.@constraint(
                subproblem,
                state.out == bincontract([binary_vars[i].out for i = 1:num_vars], epsilon)
            )
        end
    end
    return
end

relax_integrality(::PolicyGraph, ::SDDiP_bin) =
    Tuple{JuMP.VariableRef,Float64,Float64}[], JuMP.VariableRef[]


"""
Relax copy constraints
"""
function relax(node::Node, ::SDDiP_bin)

    for (i, (name, state)) in enumerate(node.states)
        integrality_handler.old_rhs[i] = JuMP.fix_value(state.in)
        integrality_handler.slacks[i] = state.in - integrality_handler.old_rhs[i]
        JuMP.unfix(state.in)

        JuMP.set_lower_bound(state.in, 0)
        JuMP.set_upper_bound(state.in, 1)
    end
end


# =========================== SDDiP_con ====================================== #

"""
    SDDiP_con(; iteration_limit::Int = 100, atol::Float64, rtol::Float64)

The SDDiP integrality handler introduced by Zou, J., Ahmed, S. & Sun, X.A.
Math. Program. (2019) 175: 461. Stochastic dual dynamic integer programming.
https://doi.org/10.1007/s10107-018-1249-5.

Calculates duals by solving the Lagrangian dual for each subproblem. Kelley's
method is used to compute Lagrange multipliers. `iteration_limit` controls the
maximum number of iterations, and `atol` and `rtol` are the absolute and
relative tolerances used in the termination criteria.

Additionally, more specific algoParams are introduced with which the settings
for the Lagrangian duals can be configured.

In SDDiP_con, state variables are not approximated, but kept as they are.

All state variables are assumed to take nonnegative values only.
"""

mutable struct SDDiP_con <: AbstractIntegralityHandler
    iteration_limit::Int
    optimizer::Any
    subgradients::Vector{Float64}
    old_rhs::Vector{Float64}
    best_mult::Vector{Float64}
    slacks::Vector{GenericAffExpr{Float64,VariableRef}}
    atol::Float64
    rtol::Float64
    algoParams::Union{Nothing,AlgoParams}

    function SDDiP_con(; iteration_limit::Int = 100, atol::Float64 = 1e-8, rtol::Float64 = 1e-8, algoParams::AlgoParams = nothing)
        integrality_handler = new()
        integrality_handler.iteration_limit = iteration_limit
        integrality_handler.atol = atol
        integrality_handler.rtol = rtol
        integrality_handler.algoParams = algoParams
        return integrality_handler
    end
end

function update_integrality_handler!(
    integrality_handler::SDDiP_con,
    optimizer::Any,
    num_states::Int,
)
    integrality_handler.optimizer = optimizer
    integrality_handler.subgradients = Vector{Float64}(undef, num_states)
    integrality_handler.old_rhs = similar(integrality_handler.subgradients)
    integrality_handler.best_mult = similar(integrality_handler.subgradients)
    integrality_handler.slacks =
        Vector{GenericAffExpr{Float64,VariableRef}}(undef, num_states)
    return integrality_handler
end

function setup_state(
    subproblem::JuMP.Model,
    state::State,
    state_info::StateInfo,
    name::String,
    ::SDDiP_con,
)
    node = get_node(subproblem)
    sym_name = Symbol(name)
    @assert !haskey(node.states, sym_name)  # JuMP prevents duplicate names.
    node.states[sym_name] = state
    graph = get_policy_graph(subproblem)
    graph.initial_root_state[sym_name] = state_info.initial_value
    return
end

function relax_integrality(model::PolicyGraph, ::SDDiP_con)
    binaries = Tuple{JuMP.VariableRef,Float64,Float64}[]
    integers = JuMP.VariableRef[]
    for (key, node) in model.nodes
        bins, ints = _relax_integrality(node.subproblem)
        append!(binaries, bins)
        append!(integers, ints)
    end
    return binaries, integers
end

"""
Relax copy constraints
"""
function relax(node::Node, ::SDDiP_con)

    for (i, (name, state)) in enumerate(node.states)
        integrality_handler.old_rhs[i] = JuMP.fix_value(state.in)
        integrality_handler.slacks[i] = state.in - integrality_handler.old_rhs[i]
        JuMP.unfix(state.in)

        JuMP.set_lower_bound(state.in, node.ext[:lower_bounds][name])
        JuMP.set_upper_bound(state.in, node.ext[:upper_bounds][name])
    end
end

# =========================== Solution technique============================== #

"""
Calling the Lagrangian dual solution method and determining dual variables
required to construct a cut
"""
function get_dual_variables(
    node::SDDP.Node,
    integrality_handler::Union{SDDiP_bin,SDDiP_con}
    )

    # storages for Lagrangian specific results
    lag_obj = 0
    lag_iterations = 0
    lag_status = :none

    # storages for return of dual values
    dual_values = Dict{Symbol,Float64}()

    number_of_states = length(node.states)

    # SOLVE PRIMAL PROBLEM TO OBTAIN BOUND FOR OPTIMAL VALUE
    ############################################################################
    solver_obj = JuMP.objective_value(node.subproblem)
    @assert JuMP.termination_status(node.subproblem) == MOI.OPTIMAL

    # DIFFERENT APPROACHES BASED ON CUT TYPE
    ############################################################################
    @assert integrality_handler.algoParams.cut_type in [:B, :SB, :L]

    if integality_handler.algoParams.cut_type == :B
        ########################################################################
        # Create Benders cut by solving LP relaxation

        TimerOutputs.@timeit NCNBD_TIMER "dual_initialization" begin
            dual_vars = initialize_duals(node, :LP)
        end
        lag_obj = JuMP.objective_value(node.subproblem)
        lag_status = :B

    elseif integrality_handler.algoParams.cut_type == :SB
        ########################################################################
        #Create strengthened Benders cut

        # Initialize dual variables by solving LP dual
        TimerOutputs.@timeit NCNBD_TIMER "dual_initialization" begin
            dual_vars = initialize_duals(node, :LP)
        end

        # solve lagrangian relaxed problem for these dual values
        lag_obj = _getStrengtheningInformation(node, dual_vars, integrality_handler)
        lag_iterations = 1
        lag_status = :SB

    elseif integrality_handler.algoParams.cut_type == :L
        ########################################################################
        TimerOutputs.@timeit NCNBD_TIMER "dual_initialization" begin
            dual_vars = initialize_duals(node, integrality_handler.algoParams.init_regime)
        end

        try
            # SOLVE LAGRANGIAN DUAL
            ####################################################################
            if integrality_handler.algoParams.sol_method == :kelley
                results = _kelley(node, solver_obj, dual_vars, integrality_handler)
                lag_obj = results.lag_obj
                lag_iterations = results.iterations
                lag_status = results.lag_status
            elseif integrality_handler.algoParams.sol_method == :bundle_level
                results = _bundle_level(node, solver_obj, dual_vars, integrality_handler)
                lag_obj = results.lag_obj
                lag_iterations = results.iterations
                lag_status = results.lag_status
            end

            # OPTIMAL VALUE CHECKS
            ####################################################################
            if integrality_handler.algoParams.status_regime == :rigorous
                if lag_status == :conv
                    error("Lagrangian dual converged to value < solver_obj.")
                elseif lag_status == :sub
                    error("Lagrangian dual had subgradients zero without LB=UB.")
                elseif lag_status == :iter
                    error("Solving Lagrangian dual exceeded iteration limit.")
                end
            elseif integrality_handler.algoParams.status_regime == :lax
                # all cuts will be used as they are valid even though not necessarily tight
            end

        catch e
            SDDP.write_subproblem_to_file(node, "subproblem.mof.json", throw_error = false)
            rethrow(e)
        end

    # SET DUAL VARIABLES AND STATES CORRECTLY FOR RETURN
    ############################################################################
    for (i, name) in enumerate(keys(node.states))
        dual_values[name] = -dual_vars[i]
    end

    # reset solver
    if node.optimizer == "GAMS"
        solver = node.subproblem.ext[:sddp_policy_graph].ext[:solver]
        if solver == "CPLEX"
            set_optimizer(node.subproblem, optimizer_with_attributes(node.optimizer, "Solver"=>solver, "optcr"=>0.0, "numericalemphasis"=>0))
        elseif solver == "Gurobi"
            set_optimizer(node.subproblem, optimizer_with_attributes(node.optimizer, "Solver"=>solver, "optcr"=>0.0, "numericalemphasis"=>0))
        else
            set_optimizer(node.subproblem, optimizer_with_attributes(node.optimizer, "Solver"=>solver, "optcr"=>0.0)
        end
    elseif
        set_optimizer(node.subproblem, optimizer_with_attributes(node.optimizer, "optcr"=>0.0)
    end

    return (
        dual_values=dual_values,
        intercept=lag_obj,
        iterations=lag_iterations,
        lag_status=lag_status,
    )
end


"""
Kelley's method to solve Lagrangian dual
"""
function _kelley(
    node::SDDP.Node,
    obj::Float64,
    dual_vars::Vector{Float64},
    integrality_handler::Union{SDDP.SDDiP_bin, SDDP.SDDiP_con},
    )

    # INITIALIZATION
    ############################################################################
    atol = integrality_handler.atol
    rtol = integrality_handler.rtol
    algoParams = integrality_handler.algoParams
    model = node.subproblem

    # Assume the model has been solved. Solving the MIP is usually very quick
    # relative to solving for the Lagrangian duals, so we cheat and use the
    # solved model's objective as our bound while searching for the optimal duals

    # relax the copy constraints based on integrality_handler type
    relax(node, integrality_handler)

    # LOGGING OF LAGRANGIAN DUAL
    ############################################################################
    #lag_log_file_handle = open("C:/Users/cg4102/Documents/julia_logs/Lagrange.log", "a")
    #print_helper(print_lagrange_header, lag_log_file_handle)

    # SET-UP APPROXIMATION MODEL
    ############################################################################
    # Subgradient at current solution
    subgradients = integrality_handler.subgradients
    # Best multipliers found so far
    best_mult = integrality_handler.best_mult
    # Dual problem has the opposite sense to the primal
    dualsense = (
        JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? JuMP.MOI.MAX_SENSE :
            JuMP.MOI.MIN_SENSE
    )

    # Approximation of Lagrangian dual as a function of the multipliers
    approx_model = JuMP.Model(integrality_handler.optimizer)

    if integrality_handler.optimizer == "GAMS"
        if algoParams.solver == "CPLEX"
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
        elseif algoParams.solver == "Gurobi"
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "NumericFocus"=>1))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
        else
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0)
        end
    elseif
        set_optimizer(approx_model, optimizer_with_attributes(integrality_handler.optimizer, "optcr"=>0.0)
        set_optimizer(model, optimizer_with_attributes(integrality_handler.optimizer, "optcr"=>0.0)
    end

    # Objective estimate and Lagrangian duals
    @variables approx_model begin
        θ
        x[1:length(dual_vars)]
    end
    JuMP.@objective(approx_model, dualsense, θ)

    # BOUND OPTIMAL VALUE IF INTENDED
    ############################################################################
    if dualsense == MOI.MIN_SENSE
        if algoParams.bound_regime == :value || algoParams_bound_regime == :both
            JuMP.set_lower_bound(θ, obj)
        else
            JuMP.set_lower_bound(θ, 0.0)
        end
        (best_actual, f_actual, f_approx) = (Inf, Inf, -Inf)
    else
        if algoParams.bound_regime == :value || algoParams.bound_regime == :both
            JuMP.set_upper_bound(θ, obj)
        else
            JuMP.set_upper_bound(θ, 9e10)
        end
        (best_actual, f_actual, f_approx) = (-Inf, -Inf, Inf)
    end

    # BOUND DUAL VARIABLES IF INTENDED
    ############################################################################
    if algoParams.bound_regime == :duals || algoParams.bound_regime == :both
        # TODO: Set dual_bound appropriately
        dual_bound = 0
        for i in 1:length(dual_vars)
            JuMP.set_lower_bound(x[i], -dual_bound)
            JuMP.set_upper_bound(x[i], dual_bound)
        end
    end

    # CUTTING-PLANE METHOD
    ############################################################################
    iter = 0
    lag_status = :none

    while iter < integrality_handler.iteration_limit
        iter += 1

        # SOLVE LAGRANGIAN RELAXATION FOR GIVEN DUAL_VARS
        ########################################################################
        # Evaluate the real function and a subgradient
        f_actual = _solve_Lagrangian_relaxation!(subgradients, node, dual_vars, integrality_handler.slacks, :yes)

        # ADD CUTTING PLANE
        ########################################################################
        # Update the model and update best function value so far
        if dualsense == MOI.MIN_SENSE
            JuMP.@constraint(
                approx_model,
                θ >= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_actual <= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        else
            JuMP.@constraint(
                approx_model,
                θ <= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_actual >= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        end

        # SOLVE APPROXIMATION MODEL
        ########################################################################
        # Get a bound from the approximate model
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL
        f_approx = JuMP.objective_value(approx_model)

        #print("UB: ", f_approx, ", LB: ", f_actual)
        #println()

        # CONVERGENCE CHECKS AND UPDATE
        ########################################################################
        # convergence achieved
        if isapprox(best_actual, f_approx, atol = atol, rtol = rtol)
            # convergence to obj -> tight cut
            if isapprox(best_actual, obj, atol = atol, rtol = rtol)
                lag_status = :aopt
            # convergence to a smaller value than obj -> valid cut
            # maybe possible due to numerical issues
            else
                lag_status = :conv
            end

        # zero subgradients despite no convergence -> valid cut
        # maybe possible due to numerical issues
        elseif all(subgradients.==0)
            lag_status = :sub

        # lb exceeds ub: no convergence
        elseif best_actual > f_approx + atol/10.0
            error("Could not solve for Lagrangian duals. LB > UB.")
        end

        # return
        if lag_status == :sub || lag_status == :aopt || lag_status == :conv
            dual_vars .= best_mult
            if dualsense == JuMP.MOI.MIN_SENSE
                dual_vars .*= -1
            end

            for (i, (name, state)) in enumerate(node.states)
                #prepare_state_fixing!(node, state_comp)
                JuMP.fix(state.in, integrality_handler.old_rhs[i], force = true)
            end

            return (lag_obj = best_actual, iterations = iter, lag_status = lag_status)
        end

        # PREPARE NEXT ITERATION
        ########################################################################
        # Next iterate
        dual_vars .= value.(x)

        # Logging
        print_helper(print_lag_iteration, lag_log_file_handle, iter, f_approx, best_actual, f_actual)

    end

    lag_status = :iter
    #error("Could not solve for Lagrangian duals. Iteration limit exceeded.")
    return (lag_obj = best_actual, iterations = iter, lag_status = lag_status)

end


"""
Solving the Lagrangian relaxation problem
"""
function _solve_Lagrangian_relaxation!(
    subgradients::Vector{Float64},
    node::SDDP.Node,
    dual_vars::Vector{Float64},
    slacks,
    update_subgradients::Symbol,
)
    model = node.subproblem
    old_obj = JuMP.objective_function(model)
    # Set the Lagrangian relaxation of the objective in the primal model
    fact = (JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? 1 : -1)
    new_obj = old_obj + fact * LinearAlgebra.dot(dual_vars, slacks)
    JuMP.set_objective_function(model, new_obj)
    JuMP.optimize!(model)
    lagrangian_obj = JuMP.objective_value(model)

    if update_subgradients == :yes
        subgradients .= fact .* JuMP.value.(slacks)
    end

    # Reset old objective, update subgradients using slack values
    JuMP.set_objective_function(model, old_obj)

    return lagrangian_obj
end


"""
Initializing duals.
"""
function initialize_duals(
    node::SDDP.Node,
    subproblem::JuMP.Model,
    dual_regime::Symbol,
)

    # Get number of states and create zero vector for duals
    number_of_states = length(node.states)
    dual_vars_initial = zeros(number_of_states)

    # DUAL REGIME I: USE ZEROS
    ############################################################################
    if dual_regime == :zeros
        # Do nothing, since zeros are already defined

    # DUAL REGIME II: USE LP RELAXATION
    ############################################################################
    elseif dual_regime == :LP
        # Create LP Relaxation
        undo_relax = JuMP.relax_integrality(subproblem);

        # Solve LP Relaxation
        JuMP.optimize!(subproblem)

        # Get dual values (reduced costs) for states as initial solution
        for (i, name) in enumerate(keys(node.states))
           reference_to_constr = FixRef(name)
           dual_vars_initial[i] = JuMP.getdual(reference_to_constr)
        end

        # Undo relaxation
        undo_relax()

    # DUAL REGIME III: USE FIXED MIP MODEL (DUALS ONLY PROVIDED BY CPLEX)
    ############################################################################
    elseif dual_regime == :cplex_fixed
        # Define cplex solver
        set_optimizer(subproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>"CPLEX", "optcr"=>0.0))

        # Solve original primal model in binary space
        JuMP.optimize!(subproblem)

        # Get dual values (reduced costs) for binary states as initial solution
        for (i, name) in enumerate(keys(node.states))
           reference_to_constr = FixRef(name)
           dual_vars_initial[i] = JuMP.getdual(reference_to_constr)
        end

        # Undo relaxation
        undo_relax()

    return dual_vars_initial
end


"""
Level bundle method to solve the Lagrangian duals.
"""
function _bundle_level(
    node::SDDP.Node,
    obj::Float64,
    dual_vars::Vector{Float64},
    integrality_handler::Union{SDDP.SDDiP_bin, SDDP.SDDiP_con},
    )

    # Assume the model has been solved. Solving the MIP is usually very quick
    # relative to solving for the Lagrangian duals, so we cheat and use the
    # solved model's objective as our bound while searching for the optimal duals

    # INITIALIZATION
    ############################################################################
    atol = integrality_handler.atol # corresponds to deltabar
    rtol = integrality_handler.rtol # corresponds to deltabar
    algoParams = integrality_handler.algoParams
    model = node.subproblem

    # initialize bundle parameters
    @assert !isnothing(algoParams.bundle_parameters)
    level_factor = algoParams.bundle_parameters.level_factor

    # relax the copy constraints based on integrality_handler type
    relax(node, integrality_handler)

    # LOGGING OF LAGRANGIAN DUAL
    ############################################################################
    #lag_log_file_handle = open("C:/Users/cg4102/Documents/julia_logs/Lagrange.log", "a")
    #print_helper(print_lagrange_header, lag_log_file_handle)

    # SET-UP APPROXIMATION MODEL
    ############################################################################
    # Subgradient at current solution
    subgradients = integrality_handler.subgradients
    # Best multipliers found so far
    best_mult = integrality_handler.best_mult
    # Dual problem has the opposite sense to the primal
    dualsense = (
        JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? JuMP.MOI.MAX_SENSE :
            JuMP.MOI.MIN_SENSE
    )

    # Approximation of Lagrangian dual as a function of the multipliers
    approx_model = JuMP.Model(integrality_handler.optimizer)

    if integrality_handler.optimizer == "GAMS"
        if algoParams.solver == "CPLEX"
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
        elseif algoParams.solver == "Gurobi"
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "NumericFocus"=>1))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
        else
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0)
        end
    elseif
        set_optimizer(approx_model, optimizer_with_attributes(integrality_handler.optimizer, "optcr"=>0.0)
        set_optimizer(model, optimizer_with_attributes(integrality_handler.optimizer, "optcr"=>0.0)
    end

    # Define Lagrangian dual multipliers
    @variables approx_model begin
        θ
        x[1:length(dual_vars)]
    end

    # BOUND OPTIMAL VALUE IF INTENDED
    ############################################################################
    if dualsense == MOI.MIN_SENSE
        if algoParams.bound_regime == :value || algoParams_bound_regime == :both
            JuMP.set_lower_bound(θ, obj)
        else
            JuMP.set_lower_bound(θ, 0.0)
        end
        (best_actual, f_actual, f_approx) = (Inf, Inf, -Inf)
    else
        if algoParams.bound_regime == :value || algoParams.bound_regime == :both
            JuMP.set_upper_bound(θ, obj)
        else
            JuMP.set_upper_bound(θ, 9e10)
        end
        (best_actual, f_actual, f_approx) = (-Inf, -Inf, Inf)
    end

    # BOUND DUAL VARIABLES IF INTENDED
    ############################################################################
    if algoParams.bound_regime == :duals || algoParams.bound_regime == :both
        # TODO: Set dual_bound appropriately
        dual_bound = 0
        for i in 1:length(dual_vars)
            JuMP.set_lower_bound(x[i], -dual_bound)
            JuMP.set_upper_bound(x[i], dual_bound)
        end
    end

    # CUTTING-PLANE METHOD
    ############################################################################
    iter = 0
    lag_status = :none
    while iter < integrality_handler.iteration_limit
        iter += 1

        # SOLVE LAGRANGIAN RELAXATION FOR GIVEN DUAL_VARS
        ########################################################################
        # Evaluate the real function and determine a subgradient
        f_actual = _solve_Lagrangian_relaxation!(subgradients, node, dual_vars, integrality_handler.slacks, :yes)

        # ADD CUTTING PLANE TO APPROX_MODEL
        ########################################################################
        # Update the model and update best function value so far
        if dualsense == MOI.MIN_SENSE
            JuMP.@constraint(
                approx_model,
                θ >= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
                # Reset upper bound to inf?
            )
            if f_actual <= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        else
            JuMP.@constraint(
                approx_model,
                θ <= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
                # Reset lower boumd to -inf?
            )
            if f_actual >= best_actual
                # bestmult is not simply getvalue.(x), since approx_model may just haven gotten lucky
                # same for best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        end

        # SOLVE APPROXIMATION MODEL
        ########################################################################
        # Define objective for approx_model
        JuMP.@objective(approx_model, dualsense, θ)

        # Get an upper bound from the approximate model
        # (we could actually also use obj here)
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL
        f_approx = JuMP.objective_value(approx_model)

        # Construct the gap (not directly used for termination, though)
        #gap = abs(best_actual - f_approx)
        gap = abs(best_actual - obj)

        # print("UB: ", f_approx, ", LB: ", f_actual, best_actual)
        # println()

        # CONVERGENCE CHECKS AND UPDATE
        ########################################################################
        # convergence achieved
        if isapprox(best_actual, f_approx, atol = atol, rtol = rtol)
            # convergence to obj -> tight cut
            if isapprox(best_actual, obj, atol = atol, rtol = rtol)
                lag_status = :aopt
            # convergence to a smaller value than obj -> valid cut
            # maybe possible due to numerical issues
            else
                lag_status = :conv
            end

        # zero subgradients despite no convergence -> valid cut
        # maybe possible due to numerical issues
        elseif all(subgradients.== 0)
            lag_status = :sub

        # lb exceeds ub: no convergence
        elseif best_actual > f_approx + atol/10.0
            error("Could not solve for Lagrangian duals. LB > UB.")
        end

        # return
        if lag_status == :sub || lag_status == :aopt || lag_status == :conv
            dual_vars .= best_mult
            if dualsense == JuMP.MOI.MIN_SENSE
                dual_vars .*= -1
            end

            for (i, (name, state)) in enumerate(node.states)
                JuMP.fix(state.in, integrality_handler.old_rhs[i], force = true)
            end

            #if appliedSolvers.MILP == "CPLEX"
            #    set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "numericalemphasis"=>0))
            #elseif appliedSolvers.MILP == "Gurobi"
            #    set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "NumericFocus"=>1))
            #else
            #    set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0))
            #end

            return (lag_obj = best_actual, iterations = iter, lag_status = lag_status)
        end

        # FORM A NEW LEVEL
        ########################################################################
        if dualsense == :Min
            level = f_approx + gap * level_factor
            #+ atol/10.0 for numerical issues?
            JuMP.setupperbound(θ, level)
        else
            level = f_approx - gap * level_factor
            #- atol/10.0 for numerical issues?
            JuMP.setlowerbound(θ, level)
        end

        # DETERMINE NEXT ITERATE USING PROXIMAL PROBLEM
        ########################################################################
        # Objective function of approx model has to be adapted to new center
        JuMP.@objective(approx_model, Min, sum((dual_vars[i] - x[i])^2 for i=1:length(dual_vars)))
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL

        # Next iterate
        dual_vars .= value.(x)

        # Logging
        # print_helper(print_lag_iteration, lag_log_file_handle, iter, f_approx, best_actual, f_actual)

    end

    lag_status = :iter
    #error("Could not solve for Lagrangian duals. Iteration limit exceeded.")
    return (lag_obj = best_actual, iterations = iter, lag_status = lag_status)

end


"""
Solve lagrangian relaxation to obtain intercept for strengthened Benders cuts
"""
function _getStrengtheningInformation(
    node::SDDP.Node,
    dual_vars::Vector{Float64},
    integrality_handler::Union{SDDP.SDDiP_bin, SDDP.SDDiP_con},
    )

    # INITIALIZATION
    ############################################################################
    algoParams = integrality_handler.algoParams
    model = node.subproblem

    # relax the copy constraints based on integrality_handler type
    relax(node, integrality_handler)

    # LOGGING OF LAGRANGIAN DUAL
    ############################################################################
    #lag_log_file_handle = open("C:/Users/cg4102/Documents/julia_logs/Lagrange.log", "a")
    #print_helper(print_lagrange_header, lag_log_file_handle)

    if integrality_handler.optimizer == "GAMS"
        if algoParams.solver == "CPLEX"
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
        elseif algoParams.solver == "Gurobi"
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "NumericFocus"=>1))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0, "numericalemphasis"=>0))
        else
            set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0))
            set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>algoParams.solver, "optcr"=>0.0)
        end
    elseif
        set_optimizer(approx_model, optimizer_with_attributes(integrality_handler.optimizer, "optcr"=>0.0)
        set_optimizer(model, optimizer_with_attributes(integrality_handler.optimizer, "optcr"=>0.0)
    end

    # SOLVE LAGRANGIAN RELAXATION FOR GIVEN DUAL_VARS
    ########################################################################
    # Evaluate the real function and a subgradient
    best_actual = _solve_Lagrangian_relaxation!(subgradients, node, dual_vars, integrality_handler.slacks, :no)

    # Logging
    # print_helper(print_lag_iteration, lag_log_file_handle, iter, f_approx, best_actual, f_actual)

    return best_actual
end
