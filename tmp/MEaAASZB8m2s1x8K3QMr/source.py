def is_located_in(points, bboxes):
    
    assert points.size(1) == 2
    assert bboxes.size(1) == 4
    return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & \
           (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))