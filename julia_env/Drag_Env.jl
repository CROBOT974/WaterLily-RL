module DragEnv

using ImageCore
using WaterLily, StaticArrays,LinearAlgebra
using PyPlot
using PyCall
np = pyimport("numpy")
@pyimport types
using ColorSchemes, ImageIO, ImageCore
using  FileIO, ImageIO, Colors
export init_env, reset!, step!, get_state
using Statistics
import Images 
using FileIO, ImageIO  
using ImageCore: channelview 
mutable struct DragSimEnv{T}
    sim::Simulation
    D::Int
    t::T
    ξ::T
    ξ_ref::Base.RefValue{T}
end


# Rotation matrix
rot(θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
_maybe_call(x) = x
function _maybe_call(value::PyObject)
    return pyisinstance(value, types.FunctionType) ? value() : value
end

function make_sim(ξ_ref::Base.RefValue{<:Real}; D=48, Re=500, d_D=0.15f0, g_D=0.05f0, size=[6,2], center = [2,0])
    C, R, U = SA[center[1], center[2]]*D, D÷2, 1.0
    big = AutoBody((x, t) -> √(sum(abs2, x - C)) - R)
    r = d_D * R
    c = C + (R + r + g_D * D) * SA[1 / 2, √3 / 2]
    small = AutoBody(
        (x, t) -> √sum(abs2, x) - r,                    # ← 定义几何形状
        (x, t) -> rot(ξ_ref[] * U * t / r) * (x - c)    # ← 动态读值
    )
    Simulation((size[1]*D, size[2]*D), (U, 0), D; ν=U * D / Re, body=big + small, T=eltype(ξ_ref[]))
end


function get_Power_coeff(sim, D)
    net_thrust = WaterLily.pressure_force(sim.flow, sim.body) + WaterLily.viscous_force(sim.flow, sim.body)
    F = net_thrust[1]
    return F / (1 * D * sim.U^3)
end

function init_env(; ξ=3.0f0, D=48)

    ξ_ref = Ref(ξ)
    sim = make_sim(ξ_ref; D=D)
    DragSimEnv(sim, D, 0.0f0, ξ, ξ_ref)
end


function reset!(env::DragSimEnv, statics::Dict{Any, Any}, variables::Dict{Any, Any})
    env.t = 0.0f0

    env.D = Int(statics["L_unit"])
    size = Vector{Int}(statics["size"])
    d_D = Float64(statics["L_ratio"])
    g_D = Float64(statics["L_gap"])
    center = Vector{Int}(statics["location"])

    env.ξ = Float64(_maybe_call(variables["ksi"]))

    env.ξ_ref[] = env.ξ                     #
    env.sim = make_sim(env.ξ_ref; D=env.D, d_D= d_D, g_D = g_D, size = size, center = center)  #
    
    println("Resetting environment with ξ = $(env.ξ), D = $(env.D)")
    return get_state(env)
end



function get_state(env::DragSimEnv)
    cp = get_Power_coeff(env.sim, env.D)
    return [env.t, cp]
end


function step!(env::DragSimEnv, ξ::Real; Δt=0.05, render=false,
               w1=10.0,w2=0.1)

    Δξ = ξ - env.ξ   # ←  env.ξ 
    env.ξ = Float32(ξ)
    env.ξ_ref[] = env.ξ
    
    done = false
    cps = Float64[]
    img = nothing
    img_np = nothing
    # === Warm-up: t ∈ [0, 2]
    while env.t < 2.0

        env.t += Δt
        sim_step!(env.sim, env.t; remeasure=true)
    end

    # === Cp accumulation: t ∈ [1, 3]
    env.t += Δt

    sim_step!(env.sim, env.t; remeasure=true)

    cp = get_Power_coeff(env.sim, env.D)
    push!(cps,cp)
    reward = - (w1 * abs(cp) + w2 * Δξ^2)

#     println("reward = $(reward)")

    if env.t >= 8.0
        done = true
        println("Done_True")
      
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

        title("cp=$(round(cp, digits=2))")

        tight_layout()
        filename = joinpath("images", "test_" * string(round(Int, env.t * 100)) * ".png")
        savefig(filename)

        img = σ_array'
        img_np = np.array(img)

        close(fig)
    end
    return get_state(env), reward, done, img_np, cps
end

end