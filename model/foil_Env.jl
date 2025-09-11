module FOILEnv

using ImageCore
using ImageMagick
using WaterLily, StaticArrays, LinearAlgebra, ParametricBodies
using PyPlot
using PyCall
np = pyimport("numpy")
@pyimport types
using ColorSchemes, ImageIO, ImageCore
using  FileIO, ImageIO, Colors
export init_env, reset!, step!, get_state
using Statistics
import Images
using ImageCore: channelview 
mutable struct FOILSimEnv{T}
    sim::Simulation
    D::Int
    t::T
    vhx::T
    vhy::T
    hx::T
    hy::T
    θ::T
    dθ::T
    ddθ::T
    hy_ref::Base.RefValue{T}
    θ_ref::Base.RefValue{T}  # 翼形受力角度
end

_maybe_call(x) = x
function _maybe_call(value::PyObject)
    return pyisinstance(value, types.FunctionType) ? value() : value
end

# Rotation matrix
rot(θ) = SA[cos(θ + π) -sin(θ + π); sin(θ + π) cos(θ + π)]

function make_sim(hy_ref::Base.RefValue{<:Real}, θ_ref::Base.RefValue{<:Real}; D=32, Re=1000, size = [8, 8], r_nose = [1, 4], r_pivot = [0.25, 0])
    nose, pivot = SA[r_nose[1], r_nose[2]]*D, SA[r_pivot[1], r_pivot[2]]*D
    radius, U = D/2, 1.0
    hx = 0

    map_func = (x, t) -> begin
        ξ = rot(θ_ref[]) * (x - nose - SA[hx, hy_ref[]] - pivot) + pivot
        SA[ξ[1], abs(ξ[2])]
    end

    foil(s, t) = D * SA[(1-s)^2, 0.6f0*(0.2969f0*(1-s) - 0.126f0*(1-s)^2 -
                                       0.3516f0*(1-s)^4 + 0.2843f0*(1-s)^6 -
                                       0.1036f0*(1-s)^8)]

    body = HashedBody(foil, (0, 1); map=map_func)
    Simulation((size[1] * D, size[2] * D), (U, 0), D; ν=U * D / Re, body=body, T=eltype(hy_ref[]))
end


function pressure_force(sim)
    sim.flow.f .= 0
    @WaterLily.loop sim.flow.f[I,:] .= sim.flow.p[I] * WaterLily.nds(sim.body, loc(0,I,Float64), WaterLily.time(sim)) over I ∈ inside(sim.flow.p)
    sum(Float64, sim.flow.f, dims=ntuple(i->i, ndims(sim.flow.u)-1))[:] |> Array
end

function init_env(; t = 0.0f0, hx=0.0f0, hy = 0.0f0, vhx = 0.0f0, vhy = 0.0f0, θ = 0.0f0, dθ = 0.0f0, ddθ = 0.0f0, D=32)
    
    hy_ref = Ref(hy)
    θ_ref = Ref(θ)

    sim = make_sim(hy_ref, θ_ref)
    FOILSimEnv(sim, D, t, vhx, vhy, hx, hy, θ, dθ, ddθ, hy_ref, θ_ref)
end


function reset!(env::FOILSimEnv, statics::Dict{Any, Any}, variables::Dict{Any, Any})
    env.t = 0.0f0

    env.D = Int(statics["L_unit"])
    size = Vector{Int}(statics["size"])
    r_nose = Vector{Float64}(statics["nose"])
    r_pivot = Vector{Float64}(statics["rot_center"])

    pos = Vector{Float64}(_maybe_call(variables["position"]))
    vel = Vector{Float64}(_maybe_call(variables["velocity"]))
    theta = Float64(_maybe_call(variables["theta"]))
    d_theta = Float64(_maybe_call(variables["rot_vel"]))
    dd_theta = Float64(_maybe_call(variables["rot_acc"]))
    
    env.hx = pos[1]
    env.hy = pos[2]
    env.vhx = vel[1] 
    env.vhy = vel[2]
    env.θ = theta
    env.dθ = d_theta
    env.ddθ = dd_theta

    env.hy_ref[] = env.hy
    env.θ_ref[] = env.θ
    env.sim = make_sim(env.hy_ref, env.θ_ref; D=env.D, size = size, r_nose = r_nose, r_pivot = r_pivot)
    return get_state(env)
end

function get_state(env::FOILSimEnv)
    return Float32[env.hx, env.hy, env.vhx, env.vhy, env.θ]
end

function step!(env::FOILSimEnv, ay::Real; Δt=env.sim.flow.Δt[end], render=false)
    done = false
    img = nothing
    img_np = nothing
    cps = nothing

    L = 32
    m = 0.68*0.12*L^2
    mₐ = π/4*L^2
    I = m^2/12
    I₀ = 5*I
    c_θ = 5.0f0
    k = SA[0.0f0,0.0f0]
    c = SA[0.0f0,0.0f0]

    pivot = SA[0.25f0 * env.D, 0.0f0]

    F = pressure_force(env.sim)
    println("force in y direction = $(F[2])")
    M = WaterLily.pressure_moment(pivot, env.sim.flow, env.sim.body)[1]
    println("Moment = $(M)")

    # Δt = env.sim.flow.Δt[end]

    if env.θ >= 0
        β = @. (M - c_θ * env.dθ + 0 * I₀ * env.ddθ) / (I + I₀)
    else
        β = @. (-M - c_θ * env.dθ + 0 * I₀ * env.ddθ) / (I + I₀)
    end
    env.dθ += β * Δt
    env.θ += (env.dθ + Δt * β / 2.) * Δt
    env.θ = mod(env.θ + π, 2π) - π  # 将θ限制在[-π, π]范围内
    env.ddθ = β
    env.θ_ref[] = env.θ

    # ahy = ((-F[2] - c[2] * env.vhy - k[2] * env.hy)/m )
    # env.vhy += ahy * Δt
    # env.hy  += env.vhy * Δt

    env.hy_ref[] = env.hy

    # === Warm-up: t ∈ [0, 5]
#     while env.t < 2.0
#
#         env.t += Δt
#         # measure body
#         measure!(env.sim, env.t)
#         mom_step!(env.sim.flow,env.sim.pois)
#
#     end

    env.t += Δt
    measure!(env.sim, env.t)
    mom_step!(env.sim.flow,env.sim.pois)

    reward = - abs(env.θ)
#     println("reward = $(env.hy)")

    if env.t >= 150.0
        done = true
        println("Done_True")
      
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
    return get_state(env), reward, done, img_np, cps
end

end