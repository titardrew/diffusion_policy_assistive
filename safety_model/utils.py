def normalize(normalizer, x):
    if normalizer is None:
        return x
    else:
        return normalizer.normalize(x)


def unnormalize(normalizer, x):
    if normalizer is None:
        return x
    else:
        return normalizer.unnormalize(x)


def multi_repeat(x, times):
    # x - [Batch, Horizon, Dim]
    def _repeat(x):
        if len(x.shape) == 2:
            # => [Times, Batch, 1]  (for len)
            return x.repeat([times, 1, 1])
        # => [Times, Batch, Horizon, Dim]
        return x.repeat([times, 1, 1, 1])
    if isinstance(x, dict):
        a = {key: _repeat(val) for key, val in x.items()}
        return a
    else:
        return _repeat(x)

