module VIVEnv

using ImageCore
using WaterLily, StaticArrays,LinearAlgebra, ParametricBodies
using PyPlot
using PyCall
np = pyimport("numpy")
@pyimport types
using ColorSchemes, ImageIO, ImageCore
using  FileIO, ImageIO, Colors
using Statistics
import Images
using FileIO, ImageIO
using ImageCore: channelview


export init_env, reset!, step!, get_state
abstract type AbstractEnv end

mutable struct VIVSimEnv{T} <: AbstractEnv
    sim::Simulation
    infos::Vector{Any}
    D::Int
    t::T
    vx::T         # 当前控制速度（施加给圆柱体的 x 方向速度）
    vy::T
    px::T         # 当前 x 位移
    py::T
    px_ref::Base.RefValue{T}  # 圆柱当前速度-x
    py_ref::Base.RefValue{T}  # 圆柱当前速度-y
end

_maybe_call(x) = x
function _maybe_call(value::PyObject)
    return pyisinstance(value, types.FunctionType) ? value() : value
end

function make_sim(px_ref::Base.RefValue{<:Real}, py_ref::Base.RefValue{<:Real}; D=16, Re=150, size=[8,6], center = [3,3])
    radius, U = Float32(D/2), 1.0
    center = SA[Float32(center[1]), Float32(center[2])]*Float32(D)

    map(x, t) = x

    # 圆柱体位置函数（由 p_ref 控制）
    body = AutoBody(
        (x, t) -> √sum(abs2, x .- center) - radius,
        map
    )

    map_func = (x, t) -> begin
        ξ = x .- center
        SA[ξ[1], abs(ξ[2])]
    end

    foil(s, t) = D * SA[(1-s)^2, 0.6f0*(0.2969f0*(1-s) - 0.126f0*(1-s)^2 -
                                       0.3516f0*(1-s)^4 + 0.2843f0*(1-s)^6 -
                                       0.1036f0*(1-s)^8)]

    body_foil = HashedBody(foil, (0, 1); map=map_func)

    Simulation((size[1]*D, size[2]*D), (U, 0), D; ν=U * D / Re, body=body, T=Float32)
end


function get_Power_coeff(sim, D)
    net_thrust = WaterLily.pressure_force(sim.flow, sim.body) + WaterLily.viscous_force(sim.flow, sim.body)
    F = net_thrust[1]
    return F / (1 * D * sim.U^3)
end

function init_env(; t = 0.0f0, px = 0.0f0, py = 0.0f0, vx=0.0f0, vy = 0.0f0, D=16)
    px_ref = Ref(px)
    py_ref = Ref(py)

    infos = Any[]
    sim = make_sim(px_ref, py_ref)

    VIVSimEnv(sim, infos, D, t, vx, vy, px, py, px_ref, py_ref)
end


function reset!(env::VIVSimEnv, statics::Dict{Any, Any}, variables::Dict{Any, Any})
    env.t = 0.0f0

    env.D = Int(statics["L_unit"])
    size = Vector{Int}(statics["size"])
    center = Vector{Float64}(statics["location"])

    pos = Vector{Float64}(_maybe_call(variables["position"]))
    vel = Vector{Float64}(_maybe_call(variables["velocity"]))

    env.px = pos[1]
    env.py = pos[2]
    env.vx = vel[1] 
    env.vy = vel[2]

    env.py_ref[] = env.py
    env.px_ref[] = env.px

    # env.D = Int(statics["L_unit"])
    # radius = Float32(env.D)/2
    # size = Vector{Int}(statics["size"])
    # center = Vector{Float64}(statics["location"])

    # println("center = $(center)")

    # pos = Vector{Float64}(_maybe_call(variables["position"]))
    # vel = Vector{Float64}(_maybe_call(variables["velocity"]))
    
    # env.px = pos[1]
    # env.py = pos[2]
    # env.vx = vel[1]
    # env.vy = vel[2]
    # env.px_ref[] = env.px
    # env.py_ref[] = env.py

    env.infos = Any[]
    env.sim = make_sim(env.px_ref, env.py_ref; D=env.D, size = size, center)
    # println("pos = $(pos)")
    println("Resetting environment with py = $(env.py), D = $(env.D)")
    return get_state(env)
end

function get_state(env::VIVSimEnv)
    return Float32[env.t, env.py, env.vy]
end


function step!(env::VIVSimEnv, F::Real; render=false)
    done = false
    img = nothing
    img_np = nothing

    # mass of the circle
    rho = 1.0                               # density of fluid
    radius = env.D/2
    mₐ = π * radius^2 * rho                 # displacement mass
    m = 2 * mₐ                              # structural mass

    # elastic coefficient
    fn = 1/4/env.D
    kx = 4 * π^2 * m * fn^2
    ky = kx

    # force = -WaterLily.total_force(env.sim)
#     println("force in x direction = $(force[1]), force in y direction = $(force[2])")

    Δt = env.sim.flow.Δt[end]

    # # x方向（VIV响应）
    # ax = (force[1] - kx * env.px) / m
    # env.px += Δt * (env.vx + Δt * ax / 2)
    # env.vx += Δt * ax
    # env.px_ref[] = env.px

    # # y方向（VIV响应）
    # ay = (force[2] - ky * env.py) / m
    # env.py += Δt * (env.vy + Δt * ay / 2)
    # env.vy += Δt * ay
    # env.py_ref[] = env.py

    # === Warm-up: t ∈ [0, 2]
    while env.t < 10.0

        env.t += Δt
        # measure body
        measure!(env.sim, env.t)
        mom_step!(env.sim.flow,env.sim.pois)

    end

    env.t += Δt
    measure!(env.sim, env.t)
    mom_step!(env.sim.flow,env.sim.pois)

    # info = Dict("F" => Float64(F), "fluid_force_y" => Float64(force[2]), "y_dis" => Float64(env.py))

    # push!(env.infos,info)
#     println("step $t: length(cps) = $(length(env.infos)), last cp = $(info)")
    reward = - abs(env.py)/env.D

    # if abs(env.py)/env.D >= 1
    #     done = true
    #     reward = -100
    #     println("Break")
    # end
    if env.t >= 250.0
        done = true
        println("Done_True")
    elseif reward <= -3.0
        done = true
        println("Break_Off")
        reward -= 10.0
    end

    if render
        # 计算流场的涡度 σ（仍旧保存到 σ）
        @inside env.sim.flow.σ[I] = WaterLily.curl(3, I, env.sim.flow.u) * env.sim.L / env.sim.U 
        @inside env.sim.flow.σ[I] = ifelse(abs(env.sim.flow.σ[I]) < 1e-3, 0f0, env.sim.flow.σ[I])

        # 保存副本 σ_array 用于可视化
        σ_array = env.sim.flow.σ[inside(env.sim.flow.p)] |> Array

        # 临时数组 sdf 计算圆柱轮廓
        sdf = similar(env.sim.flow.σ)
        WaterLily.measure_sdf!(sdf, env.sim.body, WaterLily.time(env.sim))
        sdf_array = sdf[inside(env.sim.flow.p)] |> Array

        isdir("images") || mkdir("images")

        # 绘图
        fig = figure()
        imshow(σ_array'; origin="lower", cmap="bwr", vmin=-5, vmax=5)
        colorbar()

        # 轮廓线表示圆柱（SDF = 0 处是边界）
        contour(sdf_array'; origin="lower", levels=[0.0], colors="black", linewidths=1)

        tight_layout()
        filename = joinpath("images", "test_" * string(round(Int, env.t * 100)) * ".png")
        savefig(filename)

        img = σ_array'
        img_np = np.array(img)

        close(fig)
    end
    return get_state(env), reward, done, img_np, nothing
end

end
