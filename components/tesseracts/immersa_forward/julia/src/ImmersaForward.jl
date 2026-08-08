module ImmersaForward

export radius_squared

function radius_squared(x::Float64)
    return sum(x .^ 2)
end

end # module ImmersaForward
