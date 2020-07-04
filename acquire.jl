using CSV, DataFrames, StatsBase

function partition(ind::Union{Array{Int64, 1}, UnitRange{Int64}}, r::Float64, shuffle::Bool = true)
    if typeof(ind) == UnitRange{Int64}; ind = collect(ind); end
    N = length(ind) # total number of data points in sample
    n = round(Int64, N*r) # number of data points in training set (train)
    train = [0, 0] # initialize train output
    test = [0, 0] # initialize test output
    
    if shuffle        
        ind_ = ind[randperm(N)]
    else
        ind_ = ind
    end
    
    train = ind_[1:n]
    test = ind_[(n+1):end]
    return train, test
end


df = CSV.read("localization.csv", header = false);

old_names = names(df)

new_names = map(Symbol, ["WiFi1", "WiFi2", "WiFi3", "WiFi4", "WiFi5", "WiFi6", "WiFi7", "Room"])

for i = 1:8
    rename!(df, old_names[i] => new_names[i])
end

df[:RegressionTarget] = Matrix(df[[1, 4, 6, 7]]) * [5, 10, 15, 20] + 0.01*randn(2000)

XX = map(Float64, Matrix(df[1:7

X = StatsBase.standardize(ZScoreTransform, XX, dims=2)

df2 = DataFrame(hcat(X, df[:Room]))

for i = 1:8
    rename!(df2, names(df2)[i] => new_names[i])
end

X = df2[1:7]

XX = map(Float64, Matrix(X));


