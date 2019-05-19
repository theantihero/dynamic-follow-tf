def norm(data, min_max=[]):
    if min_max==[]:
        d_min = min(data)
        d_max = max(data)
        return [(i - d_min) / (d_max - d_min) for i in data], [d_min, d_max]
    else:
        return (data - min_max[0]) / (min_max[1] - min_max[0])

'''def denorm(data, min_max):
    return [(min_max[0] - i) * (min_max[0] - min_max[1]) for i in data]'''