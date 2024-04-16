def inference(pointcloud, pos_scale=10, weight=None):
    
    if weight is not None:
        model = weight
    else:
        raise Exception('model not loaded')
    inputs = pos_scale * pointcloud
    inputs = inputs.to("cuda")

    Q, H, center, feature = model(inputs)

    Q = Q.to("cpu").detach().numpy()
    labels = H.to("cpu").detach().numpy()
    feature = feature.to("cpu").detach().numpy()
    center = center.to("cpu").detach().numpy()

    return Q, labels, center, feature