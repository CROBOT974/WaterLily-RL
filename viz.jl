using Plots; gr()

function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) = Plots.contourf(clamp.(ω|>Array,-limit,limit)',dpi=300,
               color=palette(:RdBu_11), clims=(-limit, limit), linewidth=0,
               aspect_ratio=:equal, legend=false, border=:none)

function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end

function flood(f::Array;shift=(0.,0.),cfill=:RdBu_11,clims=(),levels=10,kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],f',
        linewidth=0, levels=levels, color=cfill, clims = clims, 
        aspect_ratio=:equal; kv...)
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    contour!(sim.flow.σ[R]'|>Array;levels,lines)
end

function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                  remeasure=false,plotbody=false,kv...)
    t₀ = round(sim_time(sim))
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
        sim_step!(sim,tᵢ;remeasure)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
        flood(sim.flow.σ[R]|>Array; kv...)
        plotbody && body_plot!(sim)
        verbose && println("tU/L=",round(tᵢ,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end