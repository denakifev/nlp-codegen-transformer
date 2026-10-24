import torch


def _norm(vector):
    scale = vector.abs().max() if vector.numel() else vector.new_zeros(())
    if not bool(scale):
        return scale
    return scale * torch.linalg.vector_norm(vector / scale)


def _order(count, generator):
    device = generator.device if generator is not None else "cpu"
    return torch.randperm(count, generator=generator, device=device).cpu().tolist()


def _cosines(vectors):
    units = []
    for vector in vectors:
        norm = _norm(vector)
        units.append(vector / norm if bool(norm) else None)
    values = [float(torch.dot(units[first], units[second])) for first in range(len(units))
              for second in range(first) if units[first] is not None and units[second] is not None]
    return values


@torch.no_grad()
def project_gradients(gradients, mode="orthogonal", generator=None):
    if mode not in {"orthogonal", "pcgrad", "none"}:
        raise ValueError("Unknown gradient projection mode")
    gradients = list(gradients)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.float32 for value in gradients):
        raise ValueError("Shared gradients must be flat FP32 tensors")
    if gradients and any(value.shape != gradients[0].shape or value.device != gradients[0].device for value in gradients):
        raise ValueError("Shared gradients must use the same shape and device")
    if any(not bool(torch.isfinite(value).all()) for value in gradients):
        raise ValueError("Shared gradients must be finite")
    projected = [value.detach().clone() for value in gradients]
    order, projections = list(range(len(projected))), 0
    if mode == "orthogonal":
        order = _order(len(projected), generator)
        basis = []
        for index in order:
            vector = projected[index]
            original_norm = _norm(vector)
            for _ in range(2):
                for unit in basis:
                    coefficient = torch.dot(vector, unit)
                    vector.sub_(coefficient * unit)
                    projections += 1
            norm = _norm(vector)
            if not bool(original_norm) or bool(norm <= original_norm * 1e-6):
                vector.zero_()
            else:
                basis.append(vector / norm)
    elif mode == "pcgrad":
        norms = [_norm(value) for value in gradients]
        references = [value / norm if bool(norm) else None for value, norm in zip(gradients, norms)]
        for index, vector in enumerate(projected):
            for other in _order(len(gradients), generator):
                if other == index or references[other] is None:
                    continue
                coefficient = torch.dot(vector, references[other])
                if bool(coefficient < 0):
                    vector.sub_(coefficient * references[other])
                    projections += 1
    if any(not bool(torch.isfinite(value).all()) for value in projected):
        raise ValueError("Gradient projection overflowed")
    before, after = _cosines(gradients), _cosines(projected)
    metrics = {"mode": mode, "tasks": len(gradients), "order": order, "projections": projections,
               "conflicts_before": sum(value < -1e-6 for value in before),
               "conflicts_after": sum(value < -1e-6 for value in after),
               "max_abs_cosine_after": max(map(abs, after), default=0.0),
               "zero_inputs": sum(not bool(_norm(value)) for value in gradients),
               "zero_outputs": sum(not bool(_norm(value)) for value in projected)}
    return projected, metrics
