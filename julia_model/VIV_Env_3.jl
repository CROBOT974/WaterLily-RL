module VIVEnv

using ImageCore
using WaterLily, StaticArrays, LinearAlgebra
using PyPlot
using PyCall
np = pyimport("numpy")
@pyimport types
using ColorSchemes, ImageIO, ImageCore
using FileIO, ImageIO, Colors
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
    step::Int
    vx::T         # 当前控制速度（施加给圆柱体的 x 方向速度）
    vy::T
    px::T         # 当前 x 位移
    py::T
    px_ref::Base.RefValue{T}  # 圆柱当前位移-x
    py_ref::Base.RefValue{T}  # 圆柱当前位移-y
    vx_ref::Base.RefValue{T}  # 圆柱当前速度-x
    vy_ref::Base.RefValue{T}  # 圆柱当前速度-y
    t_ref::Base.RefValue{T}
end

_maybe_call(x) = x
function _maybe_call(value::PyObject)
    return pyisinstance(value, types.FunctionType) ? value() : value
end
T = Float32

function make_sim(px_ref::Base.RefValue{<:Real}, py_ref::Base.RefValue{<:Real}, vx_ref::Base.RefValue{<:Real}, vy_ref::Base.RefValue{<:Real}, t_ref::Base.RefValue{T}; D=16, Re=150, size=[8, 6], center=[3, 3])
    radius, U = Float32(D / 2), 1.0
    center = SA[Float32(center[1]), Float32(center[2])] * Float32(D)

    posx(t) = px_ref[] + (t - t_ref[]) * vx_ref[]
    posy(t) = py_ref[] + (t - t_ref[]) * vy_ref[]

    map(x, t) = begin
        # println("t0  = $(t_ref[] )")
        x - SA[posx(t), posy(t)]
    end


    # 圆柱体位置函数（由 p_ref 控制）
    body = AutoBody(
        (x, t) -> √sum(abs2, x .- center) - radius,
        map
    )
    Simulation((size[1] * D, size[2] * D), (U, 0), D; ν=U * D / Re, body=body, T=eltype(px_ref[]))
end


function get_Power_coeff(sim, D)
    net_thrust = WaterLily.pressure_force(sim.flow, sim.body) + WaterLily.viscous_force(sim.flow, sim.body)
    F = net_thrust[1]
    return F / (1 * D * sim.U^3)
end

function init_env(; step=0, px=0.0f0, py=0.0f0, vx=0.0f0, vy=0.0f0, D=16)
    t = 0f0
    px_ref = Ref(px)
    py_ref = Ref(py)
    vx_ref = Ref(vx)
    vy_ref = Ref(vy)
    t_ref = Ref(t)

    infos = Any[]
    sim = make_sim(px_ref, py_ref, vx_ref, vy_ref, t_ref)

    VIVSimEnv(sim, infos, D, step, vx, vy, px, py, px_ref, py_ref, vx_ref, vy_ref, t_ref)
end


function reset!(env::VIVSimEnv, statics::Dict{Any,Any}, variables::Dict{Any,Any})
    env.step = 0

    env.D = Int(statics["L_unit"])
    radius = Float32(env.D) / 2
    size = Vector{Int}(statics["size"])
    center = Vector{Float64}(statics["location"])

    println("center = $(center)")

    pos = Vector{Float64}(_maybe_call(variables["position"]))
    vel = Vector{Float64}(_maybe_call(variables["velocity"]))

    env.px = pos[1]
    env.py = pos[2]
    env.vx = vel[1]
    env.vy = vel[2]
    env.px_ref[] = env.px
    env.py_ref[] = env.py
    env.vx_ref[] = env.vx
    env.vy_ref[] = env.vy
    env.t_ref[] = 0f0

    env.infos = Any[]
    env.sim = make_sim(env.px_ref, env.py_ref, env.vx_ref, env.vy_ref, env.t_ref; D=env.D, size=size, center)
    println("pos = $(pos)")
    println("Resetting environment with py = $(env.py), D = $(env.D)")
    return get_state(env)
end

function get_state(env::VIVSimEnv)
    return [env.step, env.py, env.vy]
end


function step!(env::VIVSimEnv, F::Real; render=false)
    done = false
    img = nothing
    img_np = nothing

    # mass of the circle
    rho = 1.0f0                               # density of fluid
    radius = T(env.D / 2)
    mₐ = T(π * radius^2 * rho)                 # displacement mass
    m = T(2 * mₐ)                             # structural mass

    # elastic coefficient
    fn = T(1 / 4 / env.D)
    kx = T(4 * π^2 * m * fn^2)
    ky = kx


    #     println("force in x direction = $(force[1]), force in y direction = $(force[2])")

    Δt = env.sim.flow.Δt[end]

    # # x方向（VIV响应）
    # env.px_ref[] = env.px + Δt * env.vx

    # # y方向（VIV响应）
    # env.py_ref[] = env.py + Δt * env.vy

    tphys = sum(env.sim.flow.Δt[1:end-1])
    env.t_ref[] = tphys
    tphys += Δt



    # === Warm-up: t ∈ [0, 2]
    if env.step < 0.0

        env.step += 1
        # measure body
        measure!(env.sim, tphys)
        mom_step!(env.sim.flow, env.sim.pois)
        reward = 0

    else

        force = -WaterLily.pressure_force(env.sim)
        println("force in x direction = $(force[1]), force in y direction = $(force[2])")

        env.step += 1
        measure!(env.sim, tphys)
        mom_step!(env.sim.flow, env.sim.pois)

        # x方向（VIV响应）
        ax = (force[1] - kx * env.px) / m
        env.px += Δt * (env.vx + Δt * ax / 2.0f0)
        env.vx += Δt * ax
        # y方向（VIV响应
        ay = (force[2] + F - ky * env.py) / m
        env.py += Δt * (env.vy + Δt * ay / 2.0f0)
        env.vy += Δt * ay

        env.px_ref[] = env.px
        env.py_ref[] = env.py

        env.vx_ref[] = env.vx
        env.vy_ref[] = env.vy

        info = Dict("F" => Float64(F),
            "fluid_force_y" => Float64(force[2]),
            "fluid_force_x" => Float64(force[1]),
            "ax" => Float64(ax),
            "ay" => Float64(ay),
            "y_dis" => Float64(env.py),
            "x_dis" => Float64(env.px),
            "vx" => Float64(env.vx))

        push!(env.infos, info)
        #     println("step $t: length(cps) = $(length(env.infos)), last cp = $(info)")
        reward = -abs(env.py) / env.D

        # if abs(env.py)/env.D >= 1
        #     done = true
        #     reward = -100
        #     println("Break")
        # end
        if env.step ≥ 1600
            done = true
            println("Done_True")
        elseif any(>(3), (abs(env.py) / env.D, abs(env.px) / env.D))
            done = true
            println("Break_Off")
            reward -= 10
        end

        if render
            # 计算流场的涡度 σ（仍旧保存到 σ）
            @inside env.sim.flow.σ[I] = WaterLily.curl(3, I, env.sim.flow.u) * env.sim.L / env.sim.U
            @inside env.sim.flow.σ[I] = ifelse(abs(env.sim.flow.σ[I]) < 1e-3, 0.0, env.sim.flow.σ[I])

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

            title("tphys = $(round(tphys, digits=2)) ax = $(round(ax, digits=2)) ay = $(round(ay, digits=2))")

            tight_layout()
            filename = joinpath("images", "test_" * string(round(Int, env.step)) * "step.png")
            savefig(filename)

            img = σ_array'
            img_np = np.array(img)

            close(fig)
        end
    end
    return get_state(env), reward, done, img_np, env.infos
end

end


