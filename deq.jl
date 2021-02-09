using LinearAlgebra, ForwardDiff
using NPZ
using Plots
using Optim, LineSearches
using Random

Random.seed!(1)

"""
	Newton
"""
function newton(res::Function, x;
		tol_r = 1.0e-8, tol_d = 1.0e-6)
	y = copy(x)
	Δy = copy(x)

    r = res(y)

	println("res: $(norm(r))")
    iter = 0

    while norm(r, 2) > tol_r && iter < 50
        ∇r = ForwardDiff.jacobian(res, y)

		try
        	Δy = -1.0 * ∇r \ r
		catch
			@warn "implicit-function failure"
			return y
		end

        α = 1.0

		iter_ls = 0
        while α > 1.0e-8 && iter_ls < 100
            ŷ = y + α * Δy
            r̂ = res(ŷ)

            if norm(r̂) < norm(r)
                y = ŷ
                r = r̂
                break
            else
                α *= 0.5
				iter_ls += 1
            end

			iter_ls == 10 && (@warn "line search failed")
        end

		println("res: $(norm(r))")

        iter += 1
    end

	iter == 10 && (@warn "Newton failure")

    return y
end

"""
	LM
"""
function levenberg_marquardt(res::Function, x;
		reg = 1.0e-8, tol_r = 1.0e-8, tol_d = 1.0e-6)

	y = copy(x)
	Δy = copy(x)

	merit(z) = res(z)' * res(z)

	α = 1.0
	iter = 0

	while iter < 100
		me = merit(y)
		r = res(y)
		∇r = ForwardDiff.jacobian(res, y)

		_H = ∇r' * ∇r
		Is = Diagonal(diag(_H))
		H = (_H + reg * Is)

		pd_iter = 0
		while !isposdef(Hermitian(Array(H)))
			reg *= 2.0
			H = (_H + reg * Is)
			pd_iter += 1

			if pd_iter > 100 || reg > 1.0e12
				@error "regularization failure"
			end
		end

		try
			Δy = -1.0 * H \ (∇r' * r)
		catch
			@warn "implicit-function theorem failure"
			return y
		end

		ls_iter = 0
		while merit(y + α * Δy) > me + 1.0e-4 * (∇r' * r)' * (α * Δy)
			α *= 0.5
			reg = reg
			ls_iter += 1

			if ls_iter > 100 || reg > 1.0e12
				@error "line search failure"
			end
		end

		y .+= α * Δy
		α = min(1.2 * α, 1.0)
		reg = 0.5 * reg

		println("res: $(norm(r))")


		iter += 1

		norm(α * Δy, Inf) < tol_d && (return y)
		norm(r, Inf) < tol_r && (return y)
	end

	@warn "Gauss Newton failure"
	return y
end

# image dimensions
w = h = 8
n = w * h

# latent-space dimension
m = 20

# provided parameters
A = npzread(joinpath(pwd(), "parameters/A.npy"))
b = npzread(joinpath(pwd(), "parameters/b.npy"))
c = npzread(joinpath(pwd(), "parameters/c.npy"))
W1 = npzread(joinpath(pwd(), "parameters/w1.npy"))
W2 = npzread(joinpath(pwd(), "parameters/w2.npy"))
θ0 = vcat(vec(W1), vec(W2), vec(b))
num_params = length(θ0)

y = reshape(npzread(joinpath(pwd(), "parameters/y_gt.npy")), w, h)
heatmap(y, color=:grays, aspect_ratio=1)

# # image dimensions
# w = h = 3
# n = w * h
#
# # latent-space dimension
# m = 3
#
# # provided parameters
# # A = npzread(joinpath(pwd(), "parameters/A.npy"))
# A = randn(n, n)
# b = randn(n) #npzread(joinpath(pwd(), "parameters/b.npy"))
# # c = npzread(joinpath(pwd(), "parameters/c.npy"))
# c = rand(n)
# W1 = randn(n, n) #npzread(joinpath(pwd(), "parameters/w1.npy"))
# W2 = randn(n, m) #npzread(joinpath(pwd(), "parameters/w2.npy"))
# θ0 = vcat(vec(W1), vec(W2), vec(b))
# num_params = length(θ0)
#
# y = rand(w, h) #randreshape(npzread(joinpath(pwd(), "parameters/y_gt.npy")), w, h)
# heatmap(y, color=:grays, aspect_ratio=1)

# model
function f(θ, x, z)
    # unpack parameters
    W1 = reshape(θ[1:n * n], n, n)
    W2 = reshape(θ[n * n .+ (1:n * m)], n, m)
    b = θ[n * n + n * m .+ (1:n)]

    # evaluate network
    tanh.(W1 * x + W2 * z + b)
    # W1 * x + W2 * z + b

end
res(θ, x, z) = x - f(θ, x, z)

# deq metric
g(x, z, y) = 1.0 / n * (O(x) - y)' * (O(x) - y) + 1.0e-3 * z' * z

# output
O(x) = tanh.(A * x + c)

# parameter metric
f(θ) = θ' * θ

# random initialization
x0 = 0.1 * randn(n)
z0 = 0.1 * randn(m)

# find fixed point for fixed θ
res(w) = res(θ0, w[1:n], w[n .+ (1:m)])
w_sol = newton(res, [x0; z0])
w_sol = levenberg_marquardt(res, [x0; z0])

x_sol = w_sol[1:n]
z_sol = w_sol[n .+ (1:m)]
@show norm(res(w_sol))

# solve using Lagrangian
num_var_lagrangian = n + m + n
L(x, z, λ, y) = g(x, z, y) + λ' * res(θ0, x, z)
L(w) = L(w[1:n], w[n .+ (1:m)], w[n + m .+ (1:n)], vec(y))

λ0 = zeros(n)
L(x0, z0, λ0, vec(y))
L([x0; z0; λ0])

∇L(w) = ForwardDiff.gradient(L, w)
w_sol = newton(∇L, [x0; z0; λ0])
w_sol = levenberg_marquardt(∇L, [x0; z0; λ0])

@show norm(∇L(w_sol))
x_sol = w_sol[1:n]
z_sol = w_sol[n .+ (1:m)]
λ_sol = w_sol[n + m .+ (1:n)]

heatmap(reshape(x_sol, w, h), color=:grays, aspect_ratio=1)

# solve using augmented Lagrangian
function solve_al(x0, z0)
	num_var_al = n + m
	λ0 = zeros(n)
	ρ0 = 1.0
	L(x, z, λ, ρ, y) = g(x, z, y) + λ' * res(θ0, x, z) + 0.5 * ρ * res(θ0, x, z)' * res(θ0, x, z)

	for i = 1:5
		L(w) = L(w[1:n], w[n .+ (1:m)], λ0, ρ0, vec(y))
		∇L(w) = ForwardDiff.gradient(L, w)
		w_sol = newton(∇L, [x0; z0])

		@show norm(∇L(w_sol))

		x0 = w_sol[1:n]
		z0 = w_sol[n .+ (1:m)]

		λ0 = λ0 + ρ0 * res(θ0, x0, z0)
		ρ0 = 10.0 * ρ0
	end

	return x0, z0
end

x_sol, z_sol = solve_al(x0, z0)

heatmap(reshape(x_sol, w, h), color=:grays, aspect_ratio=1)

# # meta Lagrangian
# num_params
# num_var_lagrangian = n + m + n
# L(θ, x, z, λ, y) = g(x, z, y) + λ' * res(θ, x, z)
# L(w) = L(w[1:num_params],
# 	w[num_params .+ (1:n)],
# 	w[num_params + n .+ (1:m)],
# 	w[num_params + n + m .+ (1:n)],
# 	vec(y))
#
# λ0 = zeros(n)
# L(θ0, x0, z0, λ0, vec(y))
# L([θ0; x0; z0; λ0])
#
# ∇L(w) = ForwardDiff.gradient(L, w)[(num_params + 1):end]
#
# num_var_meta = num_params + num_var_lagrangian + num_var_lagrangian
#
# M(θ, x, z, λ, ν, y) = f(θ) + ν' * ∇L([θ; x; z; λ])
# M(w) = M(w[1:num_params],
#          w[num_params .+ (1:n)],
#          w[num_params + n .+ (1:m)],
#          w[num_params + n + m .+ (1:n)],
#          w[num_params + n + m + n .+ (1:num_var_lagrangian)],
#          vec(y))
# ∇M(w) = ForwardDiff.gradient(M, w)
# ∇M!(a, w) = ForwardDiff.gradient!(a, M, w)
#
# w0 = [θ0; x_sol; z_sol; λ_sol; zeros(num_var_lagrangian)]
# M(w0)
# ∇M(w0)
# ∇M!(rand(num_var_meta), w0)
#
# # w_sol = newton(∇M, [θ0; x_sol; z_sol; λ_sol; zeros(num_var_lagrangian)])
#
# sol = optimize(M, ∇M!, w0,
# 	GradientDescent(alphaguess = 1.0e-5, linesearch = LineSearches.Static()),
# 	Optim.Options(show_trace=true, iterations = 1000))
#
# heatmap(reshape(sol.minimizer[num_params .+ (1:n)], w, h), color=:grays, aspect_ratio=1)
# heatmap(y, color=:grays, aspect_ratio=1)
