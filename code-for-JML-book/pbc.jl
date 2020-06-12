# Performance over Biased Classifier metric

# In both the EvaluateMetric function and the pbc one, pm = performance metric to be used:
# 1 = A.R., 2 = F1 for class 0 (first class), 3 = F1 for class 1 (second class), 4 = Weighted F1

function ConfusionMatrix(yy::Array{<:Any, 1}, y::Array{<:Any, 1}, N::Int64 = length(y))
    Q = sort(unique(y)) # unique classes (sorted)
    q = length(Q) # number of classes
    M = zeros(Int64, q, q) # confusion matrix

    for i = 1:N
        ind1 = (1:q)[y[i] .== Q][1] # ground truth class index

        if y[i] == yy[i]
            M[ind1, ind1] += 1
        else
            ind2 = (1:q)[yy[i] .== Q][1] # classifier's class index
            M[ind1, ind2] += 1
        end
    end

    return M
end

function EvaluateMetric(M::Array{Int64, 2}, pm::Int64, N::Int64 = sum(M))
    z = 0.0

    if pm == 1
        z = (M[1,1] + M[2,2]) / N
    elseif pm == 2
        if M[1,1] != 0
            P = M[1,1] / sum(M[:,1])
            R = M[1,1] / sum(M[1,:])
            z = 2*P*R / (P + R)
        end
    elseif pm == 3
        if M[2,2] != 0
            P = M[2,2] / sum(M[:,2])
            R = M[2,2] / sum(M[2,:])
            z = 2*P*R / (P + R)
        end
    elseif pm == 4
        w = sum(M, dims = 2)
        f1 = zeros(2)

        if M[1,1] != 0
            P0 = M[1,1] / sum(M[:,1])
            R0 = M[1,1] / w[1]
            f1[1] = 2*P0*R0 / (P0 + R0)
        end

        if M[2,2] != 0
            P1 = M[2,2] / sum(M[:,2])
            R1 = M[2,2] / w[2]
            f1[2] = 2*P1*R1 / (P1 + R1)
        end

        w /= N
        z = (w' * f1)[1]
    else
        error("Performance metric needs to be an integer between 1 and 4, inclusive!")
    end

    return z
end

rm(x::Real, n::Int64) = x*ones(n)

function pbc(yy::Array{<:Any, 1}, y::Array{<:Any, 1}, pm::Int64 = 1)
    N = length(y)
    Q = sort(unique(y))
    q = length(Q)
    nc = Array{Int64}(undef, q)

    for i = 1:q
        nc[i] = sum(y .== Q[i])
    end

    ind = argmax(nc) # majority class

    if (pm == 1) || (pm == 4)
        b = rm(Q[ind], N) # output of a super biased classifier, geared towards the majority class
    elseif pm == 2
        b = rm(Q[1], N) # output of a super baised classifier, geared towards the first class (class 0)
    elseif pm == 3
        b = rm(Q[2], N) # output of a super baised classifier, geared towards the second class (class 1)
    end

    zb = EvaluateMetric(ConfusionMatrix(b, y, N), pm, N)
    z = EvaluateMetric(ConfusionMatrix(yy, y, N), pm, N)
    return (z - zb) / (1 - zb)
end
