using WaterLily, StaticArrays, ParametricBodies, GLMakie, CUDA

# 简单的多边形面积公式
function polygon_area(x, y)
    n = length(x)
    return 0.5 * abs(sum(x[i]*y[mod1(i+1,n)] - x[mod1(i+1,n)]*y[i] for i in 1:n))
end

# 从ParametricBody取点
function sample_points(body::ParametricBody; N=200)
    ts = range(0, 1; length=N)
    pts = [body.curve(t) for t in ts]  # 每个点是一个 SVector{2,T}
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    return xs, ys
end

# 面积
function foil_area(upper::ParametricBody, lower::ParametricBody; N=400)
    xu, yu = sample_points(upper; N=N)
    xl, yl = sample_points(lower; N=N)

    # 上表面按顺序，下表面反向，闭合成多边形
    xs = vcat(xu, xl)
    ys = vcat(yu, yl)

    return polygon_area(xs, ys)
end

function circle_and_foil(L=2^6;Re=1000,U=1,mem=CUDA.CuArray,T=Float32)
    # A moving circle using AutoBody
    circle = AutoBody((x,t)->√sum(abs2,x)-L÷2, (x,t)->x-SA[L+U*t,3L÷2-10])

    # A foil defined by the intersection of the upper and lower surface
    upper = ParametricBody(BSplineCurve(SA{T}[5L 4L 3.5L 3L 3L; 2L 2.2L 2.2L 2.3L 2L],degree=4))
    lower = ParametricBody(BSplineCurve(SA{T}[3L 3L 3.5L 4L 5L; 2L 1.7L 1.8L 2.1L 2L],degree=4))
    foil = upper ∩ lower

    A = foil_area(upper, lower)
    println("几何面积 = ", A)

    # Simulate the flow past the two general bodies
    # Simulation((10L,4L), (U,0), L; ν=U*L/Re, body=foil+circle, mem, T)
    Simulation((10L,4L), (U,0), L; ν=U*L/Re, body=foil, mem, T)
end

# make a simulation and run it
viz!(circle_and_foil(),duration=9,clims=(-0.1,0.1))