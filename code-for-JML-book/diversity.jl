# Diverse pool of diversity functions!

using Statistics

function diversity(x::Array{<:Real, 1}, n::Int64 = length(x))
	sx = sort(x)
	dmax = (sx[end] - sx[1]) / (n - 1)
	if dmax == 0; return 0.0; end
	return 1 - abs(median(diff(sx) .- dmax) / dmax)
end

function diversity(X::Array{<:Real, 2})
	n, m = size(X)
	d = Array{Float64}(undef, m)

	for i = 1:m
		d[i] = diversity(X[:,i], n)
	end

	return mean(d), d
end

