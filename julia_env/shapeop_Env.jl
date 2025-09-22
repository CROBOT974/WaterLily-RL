module FOILEnv
using ImageCore
using ImageMagick
using WaterLily, StaticArrays, LinearAlgebra, ParametricBodies
using PyPlot
using PyCall
np = pyimport("numpy")
@pyimport types
using ColorSchemes, ImageIO, ImageCore
using FileIO, ImageIO, Colors
export init_env, reset!, step!, get_state
using Statistics
import Images
using ImageCore: channelview
mutable struct FOILSimEnv{T}
    sim::Simulation
    infos::Vector{Any}
    D::Int
    t::T
    shape::Vector{T}
    shape_pre::Vector{T}
    sha1_ref::Base.RefValue{T}
    sha2_ref::Base.RefValue{T}
    sha3_ref::Base.RefValue{T}
    sha4_ref::Base.RefValue{T}
    trigger::Bool
    reward::T
end

_maybe_call(x) = x
function _maybe_call(value::PyObject)
    return pyisinstance(value, types.FunctionType) ? value() : value
end

function foil_curve(sha1, sha2, sha3, sha4, L)
    upper = ParametricBody(BSplineCurve(SA{Float32}[
            4L 3L 2.5L 2L 1.5L 1L 1L;
            2L 2.3L 2.3L 2.4L 2.5L 2.1L 2L
        ], degree=5))

    lower = ParametricBody(BSplineCurve(SA{Float32}[
            1L 1L 1.5L 2L 2.5L 3L 4L;
            2L 1.9L (2-sha1)*L (2-sha2)*L (2-sha3)*L (2-sha4)*L 2L
        ], degree=5))

    return upper ∩ lower
end


# Rotation matrix
rot(θ) = SA[cos(θ + π) -sin(θ + π); sin(θ + π) cos(θ + π)]

function make_sim(sha1_ref::Base.RefValue{<:Real}, sha2_ref::Base.RefValue{<:Real}, sha3_ref::Base.RefValue{<:Real}, sha4_ref::Base.RefValue{<:Real}; D=32, Re=1000, size=[6, 4])
    U = 1.0
    L = Float32(D)

    # function foil_curve(sha1, sha2, sha3, L)
    #     upper = ParametricBody(BSplineCurve(SA{Float32}[
    #         3L 2L 1.5L 1L 1L;
    #         2L 2.3L 2.3L (2 + sha1)*L 2L
    #     ], degree=4))

    #     lower = ParametricBody(BSplineCurve(SA{Float32}[
    #         1L 1L 1.5L 2L 3L;
    #         2L (2 - sha1)*L (2 - sha2)*L (2 - sha3)*L 2L
    #     ], degree=4))

    #     return upper ∩ lower
    # end
    foil = foil_curve(sha1_ref[], sha2_ref[], sha3_ref[], sha4_ref[], L)

    Simulation((size[1] * D, size[2] * D), (U, 0), D; ν=U * D / Re, body=foil, mem=Array, T=Float32)
end


function pressure_force(sim)
    sim.flow.f .= 0
    WaterLily.@loop sim.flow.f[I, :] .= sim.flow.p[I] * WaterLily.nds(sim.body, loc(0, I, Float64), WaterLily.time(sim)) over I ∈ inside(sim.flow.p)
    sum(Float64, sim.flow.f, dims=ntuple(i -> i, ndims(sim.flow.u) - 1))[:] |> Array
end

function init_env(; t=0.0f0, shape=[0.0f0, 0.0f0, 0.0f0, 0.0f0], reward=0f0, trigger=false, D=32)

    sha1_ref = Ref(shape[1])
    sha2_ref = Ref(shape[2])
    sha3_ref = Ref(shape[3])
    sha4_ref = Ref(shape[4])

    shape_pre = shape

    infos = Any[]

    sim = make_sim(sha1_ref, sha2_ref, sha3_ref, sha4_ref)
    FOILSimEnv(sim, infos, D, t, shape, shape_pre, sha1_ref, sha2_ref, sha3_ref, sha4_ref, trigger, reward)
end


function reset!(env::FOILSimEnv, statics::Dict{Any,Any}, variables::Dict{Any,Any})
    env.t = 0.0f0
    trigger = false

    env.D = Int(statics["L_unit"])
    size = Vector{Int}(statics["size"])

    env.shape = Vector{Float32}(variables["shape"])


    env.infos = Any[]
    env.sha1_ref[] = env.shape[1]
    env.sha2_ref[] = env.shape[2]
    env.sha3_ref[] = env.shape[3]
    env.sha4_ref[] = env.shape[4]
    env.shape_pre = env.shape
    env.reward = 0f0

    env.sim = make_sim(env.sha1_ref, env.sha2_ref, env.sha3_ref, env.sha4_ref; D=env.D, size=size)
    return get_state(env)
end

function get_state(env::FOILSimEnv)
    point1 = 2 - env.shape[1]
    point2 = 2 - env.shape[2]
    point3 = 2 - env.shape[3]
    point4 = 2 - env.shape[4]
    return Float32[point1, point2, point3, point4]
end

function step!(env::FOILSimEnv, shape::Vector{Float64}; Δt=0.7, render=false)
    done = false
    img = nothing
    img_np = nothing
    reward = nothing
    L = env.D

    if !env.trigger

        env.trigger = true
    else
        env.trigger = false
        env.shape = shape
        env.sha1_ref[] = env.shape[1]
        env.sha2_ref[] = env.shape[2]
        env.sha3_ref[] = env.shape[3]
        env.sha4_ref[] = env.shape[4]
        env.sim.body = foil_curve(env.sha1_ref[], env.sha2_ref[], env.sha3_ref[], env.sha4_ref[], L)
        norm = LinearAlgebra.norm(env.shape_pre - env.shape)

        # === Warm-up: t ∈ [0, 5]
        while env.t < 2.0

            env.t += Δt

            # measure body
            measure!(env.sim, env.t)
            mom_step!(env.sim.flow, env.sim.pois)

        end

        while env.t < 450.0
            force = WaterLily.total_force(env.sim)
            env.t += Δt
            measure!(env.sim, env.t)
            mom_step!(env.sim.flow, env.sim.pois)
            env.reward += force[2] / force[1]

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

                title("下 = $(2)L, $(1.9)L, $((2 - env.shape[1]))L, $((2 - env.shape[2]))L, $((2 - env.shape[3]))L, $(2)L. Norm = $(norm)")

                tight_layout()
                filename = joinpath("images", "test_" * string(round(Int, env.t * 100)) * ".png")
                savefig(filename)

                img = σ_array'
                img_np = np.array(img)

                close(fig)
            end
        end

        reward = env.reward / floor((env.t - 2) / Δt)
        # reward = - norm
        cp = reward
        #     println("reward = $(env.hy)")

        if env.t >= 450.0
            done = true
            println("Done_True")
        end
        
        push!(env.infos, cp)
    end

    return get_state(env), reward, done, img_np, env.infos
end

end