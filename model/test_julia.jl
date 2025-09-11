# run_viv_standalone.jl
using PyPlot
using CSV, DataFrames

# 载入你的模块
include("VIV_Env_2.jl")
using .VIVEnv

# —— 场景参数（和你 Gym 侧一致即可）——
diameter = 16
statics = Dict{Any,Any}(
    "L_unit" => diameter,
    "action_scale" => 100,
    "size" => [10, 8],
    "location" => [3.0, 4.0],
)

variables = Dict{Any,Any}(
    "position" => [0.0, diameter/6],
    "velocity" => [0.0, 0.0],
)

# —— 初始化与 reset! ——
env = VIVEnv.init_env()
state0 = VIVEnv.reset!(env, statics, variables)
println("初始状态: ", state0)

# —— 运行设置 ——
Nsteps = 2000             # 调用 step! 的次数（包括你代码里的 warmup 分支）
render_every = 50         # 每隔多少步保存一张涡量图（渲染很耗时，训练时建议关）

# —— 历史记录 ——（时间、位移、速度、受力）
t_hist  = Float32[]       # 物理时间 tphys = sum(Δt)
py_hist = Float32[]       # y 位移（单位：与 env.get_state 一致）
vy_hist = Float32[]       # y 速度
fx_hist = Float32[]
fy_hist = Float32[]

for k in 1:Nsteps
    # 这里把 F 当作 0（无外部控制），render 仅按需开启
    state, reward, done, img, infos = VIVEnv.step!(env, 0.0; render = true)

    # 记录物理时间与状态
    push!(t_hist, sum(env.sim.flow.Δt))
    push!(py_hist, state[2])
    push!(vy_hist, state[3])

    # 记录流体力（可能头几步 env.infos 为空，做个保护）
    if !isempty(env.infos)
        last = env.infos[end]
        push!(fx_hist, last["fluid_force_x"])
        push!(fy_hist, last["fluid_force_y"])
    else
        push!(fx_hist, NaN)
        push!(fy_hist, NaN)
    end

    if done
        println("提前结束：在第 $k 步触发 done")
        break
    end
end
