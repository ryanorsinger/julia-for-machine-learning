# Fuzzy K Nearest Neighbor classifier
# -----------------------------------
# Coded by Zacharias Voulgaris, as part of the Julia for Machine Learning book.
# Code is not optimized as it's written for educational purposes primarily.
#
# Usage:
# Training: M, Q = fknn_train(X, y, k, m), where
#                  X is the features matrix (each feature is a column),
#                  y is the class labels vector, and
#                  k is the number of neighbors to consider
#                  m is a parameter of the similarity function related to the severity of the penalty of farther neighbors (default = 3)
# Testing: Y, C = fknn_apply(XX, X, k, M, Q[, L]), where
#                 XX is the features matrix for the TEST set (each feature is a column)
#                 X is the features matrix for the training set
#                 k is the number of neighbors to consider
#                 M is the membership matrix from the previous method's outputs
#                 Q is the distinct class labels vector (sorted in ascending order)
#                 Y is the predicted labels vector
#                 C is the confidence vector corresponding to Y (confidence takes values between 0 and 1)

max_z = 1e99 # maximum allowed value for z

# AUXILIARY METHODS
function similarity(x::Array{<:Real, 1}, y::Array{<:Real, 1}, m::Int64 = 3)
    d = x .- y
    sd = sum(d.^2)
    z = sd ^ (1 / (1-m))
    return min(z, max_z) # it's good to put a ceiling to the values this metric takes so that we don't get Infs in the equations
end

function FindKNearestNeighborsTraining(X::Array{<:Real, 2}, j::Int64, k::Int64, N::Int64, m::Int64 = 3)
    # Initialization
    ind = Array{Int64}(undef, k)
    x = X[j,:] # data point under consideration
    s = Array{Float64}(undef, N) # similarities vector

    # Find indexes that are not j (to avoid taking into account the original data point itself)
    indexes = collect(1:N)
    deleteat!(indexes, j)

    # Calculate distances of x to all other data points in X
    for i in indexes; s[i] = similarity(x, X[i,:], m); end

    # Finding nearest neighbors of x
    for i = 1:k
        ind[i] = argmax(s)
        s[ind[i]] = -Inf
    end

    return ind
end

function FindKNearestNeighborsTesting(X::Array{<:Real, 2}, x::Array{<:Real, 1}, k::Int64, N::Int64, m::Int64 = 3)
    ind = Array{Int64}(undef, k)
    sn = Array{Float64}(undef, k)
    s = Array{Float64}(undef, N)

    for i = 1:N; s[i] = similarity(x, X[i,:], m); end

    for i = 1:k
        sn[i], ind[i] = findmax(s)
        s[ind[i]] = -Inf
    end

    return ind, sn
end

function FindNeighborsInEachClass(ind::Array{Int64, 1}, y::Array{<:Any, 1}, Q::Array{<:Any, 1}, k::Int64, L::Int64)
    IND = BitArray(undef, k, L) # binary matrix depicting placements of neighbors across the various classes

    for n = 1:k # repeat for each neighbor in ind
        for c = 1:L # repeat for each class in Q
            IND[n,c] = (y[ind[n]] .== Q[c]) # does neighbor n belong to class c or not?
        end
    end

    return IND, sum(IND, dims=1)
end

function MembershipTraining(X::Array{<:Real, 2}, y::Array{<:Any, 1}, k::Int64, m::Int64 = 3)
    N = size(X,1) # size of TRAINING set
    Q = sort(unique(y)) # distinct class labels
    L = length(Q) # number of labels
    M = Array{Float64}(undef, N, L) # membership matrix
    n = Array{Int64}(undef, L) # number of neighbors in each class

    for j = 1:N # repeat for all data points in training set X, y
        ind = FindKNearestNeighborsTraining(X, j, k, N, m)
        n = FindNeighborsInEachClass(ind, y, Q, k, L)[2]

        for i = 1:L # repeat for all classes in Q
            M[j,i] = n[i] / k
        end
    end

    return M
end

function MembershipTesting(XX::Array{<:Real, 2}, X::Array{<:Real, 2}, M::Array{Float64, 2}, k::Int64, Q::Array{<:Any, 1}, L::Int64, m::Int64 = 3)
    N = size(X, 1) # length of TRAINING set
    NN = size(XX, 1) # length of TESTING set
    MM = zeros(NN, L)

    for i = 1:NN
        x = XX[i,:] # test data point under consideration
        ind, sn = FindKNearestNeighborsTesting(X, x, k, N, m)
        ssn = sum(sn)
        IND = FindNeighborsInEachClass(ind, y, Q, k, L)[1]

        for c = 1:L
            Mind = ind[IND[:,c]] # numeric indexes of neighbors for class c, applicable to X and M matrices
            DNind = (1:k)[IND[:,c]] # numeric indexes of neighbor for class c, applicable to dn vector

            if !isempty(Mind)
                for j = 1:length(Mind)
                    MM[i,c] += M[Mind[j],c] * sn[DNind[j]]
                end

                MM[i,c] /= ssn
            end
        end
    end

    return MM
end

function DecisionAndConfidence(M::Array{Float64, 1}, Q::Array{<:Any, 1}, L::Int64, mc::Int64)
    sM = sum(M)

    if sM == 0.0
        return Q[mc], 1 / L
    else
        M_best, ind = findmax(M)
        return Q[ind], M_best / sum(M)
    end
end

# MAIN METHODS
function fknn_train(X::Array{<:Real, 2}, y::Array{<:Any, 1}, k::Int64, m::Int64 = 3)
    M = MembershipTraining(X, y, k, m)
    Q = sort(unique(y))
    L = length(Q)
    CL = Array{Int64}(undef, L)
    for i = 1:L; CL[i] = sum(y .== Q[i]); end
    mc = argmax(CL) # majority class
    return M, Q, mc
end

function fknn_apply(XX::Array{<:Real, 2}, X::Array{<:Real, 2}, k::Int64, M::Array{Float64, 2}, Q::Array{<:Any, 1}, mc::Int64, m::Int64 = 3, L::Int64 = length(Q))
    NN = size(XX, 1)
    Y = Array{eltype(y)}(undef, NN)
    C = Array{Float64}(undef, NN)

    MM = MembershipTesting(XX, X, M, k, Q, L, m)

    for i = 1:NN
        Y[i], C[i] = DecisionAndConfidence(MM[i,:], Q, L, mc)
    end

    return Y, C
end
