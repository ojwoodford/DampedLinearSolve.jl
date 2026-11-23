# Mixed-precision iterative refinement of the dense Jacobian system's QR solve, for an element type T narrower than
# Float64 (SP-D11). Float64 systems are not refined: the solve selects the refinement by dispatch on T, and a Float64
# system holds no refinement storage.
#
# The method is Björck's refinement of the augmented system, the residual carried as an unknown (Björck, "Iterative
# refinement of linear least squares solutions I", BIT 7, 1967; Higham, Accuracy and Stability of Numerical
# Algorithms, 2nd ed., §20.5). The damped, conditioned problem is min ‖A y − c‖, with A = [J S; √(λD) S] ((m + n) × n;
# undamped, A = J S) and c = [r; 0], and x = S y. Its augmented system is
#     ρ + A y = c,   Aᵀρ = 0,
# whose solution is the least-squares y and its residual ρ = c − A y. The T factorization A = Q R (a QR of J S, and
# for a damped solve a second QR of the stacked [R₁; √(λD) S], so Q is the product of the two stages' reflectors)
# solves it to the accuracy of T; each step of refinement forms the augmented system's residual in Float64, from the
# stored J (exact in Float64), the scale and the damping, not from the factors:
#     f₁ = c − ρ − A y,   f₂ = −Aᵀρ,
# and solves for the correction with the T factors, applied in Float64:
#     h = R⁻ᵀ f₂;  (d₁, d₂) = Qᵀ f₁, d₁ its first n entries;  δy = R⁻¹(d₁ − h);  δρ = Q (h, d₂),
# then y += δy, ρ += δρ. The start is ρ = 0, y = 0, whose correction is the QR solve itself (y₀ = R⁻¹d₁ and
# ρ₀ = Q (0, d₂), from d = Qᵀc): the solve computes y₀ as without refinement, and ρ₀ from the same d. The rate of
# convergence is taken from the references, about eps(T)·κ(A) per step.
#
# Not these, which look alike: re-solving with the residual as the right-hand side (δy = R⁻¹Qᵀρ, through the same T
# reflectors that caused the error), and the corrected seminormal equations alone (δy = (RᵀR)⁻¹Aᵀρ with ρ = c − A y,
# which contracts only by eps(T)·κ²).
#
# The number of steps is a starting value (rules.md 26; improvements.md item 5).
const QR_REFINEMENT_STEPS = 1

# The refinement's Float64 vectors, allocated with the scratch: ρ and f₁ (m + n: the m rows of J S, then the n damping
# rows) and h (n). None for Float64, and none for the empty scratch of a system solved otherwise (m = 0: the normal
# equations, LSMR).
struct QRRefinement
    rho::Vector{Float64}
    f::Vector{Float64}
    h::Vector{Float64}
end
qr_refinement(::Type{Float64}, m::Integer, n::Integer) = nothing
qr_refinement(::Type, m::Integer, n::Integer) = m == 0 ? nothing : QRRefinement(zeros(m + n), zeros(m + n), zeros(n))

# z = U⁻ᵀ z for the upper triangle U = R[1:n, 1:n] (a forward substitution with Uᵀ), column-oriented, in Float64
function forwardsubstitute_transpose!(z::AbstractVector{Float64}, R::Matrix, n::Integer)
    @inbounds for j in 1:n
        t = z[j]
        for i in 1:j-1
            t -= Float64(R[i, j]) * z[i]
        end
        z[j] = t / Float64(R[j, j])
    end
    return z
end

# `steps` steps of refinement of the solve whose solution, in the scaled variables, is `y` (the QR solve's y₀),
# for the J (m × n, in T) and r of the system, the scale `s`, and the damping `lambda`, `scaling`. `Jr`, `tau` hold
# the QR of J S (reflectors below R₁) and `qv` = Q₁ᵀr; for a damped solve `Sm`, `tau2` hold the QR of the stacked
# matrix and `qv2` its transform of [q; 0], and `qv2` is the work vector of length 2n for the second stage. With
# k = min(m, n), the stacked matrix's rows k+1:n are zero rows; refinement treats them as rows of A with c = 0 there,
# whose residual is zero, and drops what a transform leaves in them.
function refine_qr!(rf::QRRefinement, y::AbstractVector{Float64}, J::Matrix, r::Vector{Float64}, s::Vector{Float64},
                    lambda::Float64, scaling, Jr::Matrix, tau::Vector, Sm::Matrix, tau2::Vector,
                    qv::Vector{Float64}, qv2::Vector{Float64}, steps::Integer)
    m, n = size(J)
    k = min(m, n)
    damped = lambda > 0
    R = damped ? Sm : Jr
    rho, f, h = rf.rho, rf.f, rf.h
    w = qv2
    @inbounds begin
        # ρ₀ = Q (0, d₂), d = Qᵀc
        if damped
            for i in 1:n
                w[i] = 0.0
            end
            apply_reflectors_reverse!(w, Sm, tau2, n)            # w[n+1:2n] still holds d₂'s second-stage part
            for i in 1:k
                f[i] = w[i]
            end
            for i in k+1:m
                f[i] = qv[i]
            end
            for i in 1:n
                rho[m + i] = w[n + i]
            end
        else
            for i in 1:n
                f[i] = 0.0
            end
            for i in n+1:m
                f[i] = qv[i]
            end
            for i in 1:n
                rho[m + i] = 0.0
            end
        end
        apply_reflectors_reverse!(view(f, 1:m), Jr, tau, k)
        for i in 1:m
            rho[i] = f[i]
        end
        for _ in 1:steps
            # f₁ = c − ρ − A y and f₂ = −Aᵀρ (into h), in Float64 from J, the scale and the damping
            for i in 1:m
                f[i] = r[i] - rho[i]
            end
            for j in 1:n
                xj = s[j] * y[j]
                for i in 1:m
                    f[i] -= Float64(J[i, j]) * xj
                end
            end
            for j in 1:n
                t = 0.0
                for i in 1:m
                    t += Float64(J[i, j]) * rho[i]
                end
                h[j] = -(s[j] * t)
            end
            if damped
                for j in 1:n
                    dj = sqrt(damping_entry(lambda, scaling, j)) * s[j]
                    f[m + j] = -rho[m + j] - dj * y[j]
                    h[j] -= dj * rho[m + j]
                end
            end
            forwardsubstitute_transpose!(h, R, n)                  # h = R⁻ᵀ f₂
            # d = Qᵀ f₁
            apply_reflectors!(view(f, 1:m), Jr, tau, k)
            d = f
            if damped
                for i in 1:k
                    w[i] = f[i]
                end
                for i in k+1:n
                    w[i] = 0.0
                end
                for i in 1:n
                    w[n + i] = f[m + i]
                end
                apply_reflectors!(w, Sm, tau2, n)
                d = w
            end
            # δy = R⁻¹(d₁ − h), into d[1:n]; y += δy
            for i in 1:n
                d[i] -= h[i]
            end
            backsubstitute!(view(d, 1:n), R, n)
            for i in 1:n
                y[i] += d[i]
            end
            # δρ = Q (h, d₂); ρ += δρ
            for i in 1:n
                d[i] = h[i]
            end
            if damped
                apply_reflectors_reverse!(w, Sm, tau2, n)
                for i in 1:k
                    f[i] = w[i]
                end
                for i in 1:n
                    rho[m + i] += w[n + i]
                end
            end
            apply_reflectors_reverse!(view(f, 1:m), Jr, tau, k)
            for i in 1:m
                rho[i] += f[i]
            end
        end
    end
    return y
end
