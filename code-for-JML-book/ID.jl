# Index of Discernibility functions

using Statistics

# Auxiliary function for DID
function findin(x::Array{<:Any, 1}, x0::Any)
	ind = (x .== x0) .* (1:length(x))
	return ind[ind .> 0]
end

# Auxiliary function for DID
within(x::Real, a::Real, b::Real) = (x >= a)&&(x <= b)

# Auxiliary function for DID (number of classes)
function nc(O::Array{<:Any,1})
	Q = unique(O)
	q = length(Q)
	C = Array{Any}(undef, q)
	N = Array{Int64}(undef, q)
	n = length(O)

	for i = 1:q
		C[i] = findin(O, Q[i])
		N[i] = length(C[i])
	end

	return q, Q, C, N
end

# Auxiliary generic function
rm(x::Real, n::Int64) = x*ones(n)

rm(X::Array{<:Real,1}, n::Int64) = ones(n)*X'

# Simple normalization method using min-max
function mmn(x::Array{<:Real, 1})
    m, M = extrema(x)

    if M != m
        return (x .- m) / (M .- m)
    else
        return rm(0.5, length(x))
    end
end

# Basic sampling method
function sample(I::Array{<:Real, 2}, O::Array{<:Any, 1}, s::Int64 = 0)
	N, nf = size(I)
	c, _, C, cvp = nc(O) # cvp = class values population
	if s == 0; s = div(N, 10); end # if no sample size is defined, use 1/10th of the original size
	S = Array{Int64}(undef, s) # Sample indexes
	sc = Array{Float64}(undef, c)
	z = 0

	for i = 1:c
		sc[i] = s*cvp[i] / N # samples of class i to be taken
		temp = round(Int64, sc[i])

		if temp > 0
			S[(z+1):(z+temp)] = C[i][randperm(cvp[i])][1:temp]
			z += temp
			sc[i] = NaN
		end
	end

	while z < s
		M, ind = findmax(sc)
		z += 1
		S[z] = C[ind][rand(1:cvp[ind])]
		sc[ind] = NaN
	end

	S = S[randperm(s)]
	return I[S,:], O[S], S
end

function nom2bin(F::Array{<:Any, 1})
	uv = sort(unique(F))
	n = length(uv)
	N = length(F)
	X = zeros(Int64, N, n)

	for i = 1:N
		ind = (1:n)[F[i] .== uv]
		X[i, ind] = 1
	end

	return X, uv
end

function nom2bin(I::Array{<:Any, 2})
	N, na = size(I)
	J = Array{Int64}(undef, N, 1)  # binary features Array
	K = Array{AbstractString}(undef, 0)  # names Array

	for i = 1:na
		X, uv = nom2bin(I[:,i])
		J = hcat(J, X)

		for x in uv
			y = replace(string(x), ",", "")
			y = replace(y, " ", "_")
			push!(K, string("is_", y))
		end
	end

	return J[:, 2:end], K
end

function CT(X::Array{<:Any, 1}, Y::Array{<:Any, 1})
	NX = length(X)
	NY = length(Y)

	if NX == NY
		QX = unique(X)
		qx = length(QX)
		QY = unique(Y)
		qy = length(QY)
		C = Array{Int64}(undef, qx, qy)

		for i = 1:qx
			ind_x = findin(X, [QX[i]])

			for j = 1:qy
				ind_y = findin(Y, [QY[j]])
				C[i,j] = length(intersect(ind_x, ind_y))
			end
		end

		return C
	end
end

# Distance-based Index of Discernibility
function DID(I::Array{<:Real,2}, O::Array{<:Any,1}, s::Int64 = 0)
	I = Float64.(I)
	N, na = size(I)

	if N < na
		N, na = na, N
	end

	if s == 0
		s = N
	end

	if na > 1
		for i = 1:na
			X = I[:,i]
			mp = (maximum(X) + minimum(X)) / 2
			if !within(mp, 0.45, 0.55)  # feature is not normalized
				I[:,i] = mmn(X)
			end
		end
	end

	q, _, C, n = nc(O)
	c = zeros(q, na) # Centers of hyperspheres
	R = zeros(q) # Radii of hyperspheres
	ICD = zeros(q, q) # Inter Class Discernibility
	sc = zeros(Int64, q) # Samples of each Class
	S = Array{Any}(undef, q) # actual Samples

	for i = 1:q
		c[i,:] = mean(I[C[i],:], dims=1) # centers of classes
		cc = rm(c[i,:],n[i])
		D = sum((cc - I[C[i],:]).^2 , dims = 2) # distances to center of each class
		R[i] = max(eps(), sqrt(maximum(D))) # Radius of each hypersphere
		sc[i] = round(Int64, s*n[i]/N) # size of sample of class i
		x = copy(C[i]) # indexes of class i
		z = n[i] # number of patters in class i
		temp = zeros(sc[i], na)

		r = sort(unique(rand(1:sc[i], z)))
		lr = length(r)
		temp[1:lr,:] = I[x[r],:]
		deleteat!(x, r)
		z -= lr

		for j = (lr+1):sc[i]
			r = ceil(Int64,z*rand())
			temp[j,:] = I[x[r],:]
			deleteat!(x, r)
			z -= 1
		end

		S[i] = copy(temp)
	end

	for i = 1:(q-1)
		for j = (i+1):q
			TEMP = 0.0

			for k = 1:sc[i]
				temp = sqrt(sum((S[i][k,:] - c[j,:]).^2, dims=1)[1]) / R[j]
				TEMP += min(temp, 1)
			end

			for k = 1:sc[j]
				temp = sqrt(sum((S[j][k,:] - c[i,:]).^2, dims=1)[1]) / R[i]
				TEMP += min(temp, 1)
			end

			ICD[i,j] = TEMP[1] /  (sc[i] + sc[j])
		end
	end

	ICD = ICD + ICD'
	y = sum(ICD) / (q^2 - q)
	return y, ICD
end

function DID(I::Array{<:Real,1}, O::Array{<:Any,1}, s::Int64 = 0)
	J = I[:,:]
	return DID(J,O,s)
end


# Nominal Index of Discernibility
function NID(I1::Array{<:Any,2}, O1::Array{<:Any,1}, s::Int64 = 0)
	N, na = size(I1)

	if s == 0
		s = N
	end

	Y = Array{Float64}(undef, na) # discentibilities of various features

	if s < N
		I, O = sample(I1, O1)
	else
		I = copy(I1)
		O = copy(O1)
	end

	q, Q, CI, CC = nc(O)
	CP = CC / s # class probabilities

	for i = 1:na
		F = I[:,i]
		C = CT(F, O)
		q = size(C, 2)
		n = size(C, 2) # number of feature values
		z = Array{Float64}(undef, n)
		CP = sum(C, dims=1) / s # class probabilities
		E = Array{Float64}(undef, n, q) # expected probabilities for various classes
		P = Array{Float64}(undef, n, q) # actual probabilities

		for j = 1:q
			E[:, j] .= CP[j] / n
			P[:, j] .= C[:, j] / CC[j]
		end

		Z = abs.(P ./ E - ones(n, q))
		M = 2(n-1)ones(n,q)
		Z ./= M
		z = maximum(Z, dims = 1) # largest deviations for the various class values
		Y[i] = sum(z.*CP) # weighted average
	end

	y, bf = findmax(Y)
	return y, Y, bf # y = discernibility of whole dataset, bf = best feature
end

function NID(I::Array{<:Any,1}, O::Array{<:Any,1}, s::Int64 = 0)
	J = I[:,:]
	return NID(J,O,s)
end
